"""
src/evaluation/self_patching.py
================================
Oracle headroom analysis using the self-patching mechanism.

After full training (50 epochs), this module:
1. Loads the trained model checkpoint
2. Uses (l_s_best, l_t_best) from layer_profile.json
3. For each P_gen example, patches h_E^{l_t}(P_gen) ← h_E^{l_s}(P_mem)
4. Measures A_oracle = accuracy with the oracle patch
5. Computes headroom = A_oracle − A_gen (gap the training is closing)
6. Saves oracle_results.json alongside eval_results.json

This is the "Phase G" post-training oracle analysis from implementation_plan.md.
It answers: *how much better could this model be if it had perfect alignment?*

Usage:
    python -m src.evaluation.self_patching \
        --checkpoint    outputs/runs/qwen3.5-1.5b/stark_prime/.../checkpoint_epoch50 \
        --model_id      Qwen/Qwen3.5-1.5B \
        --profile_path  data/processed/layer_profile_qwen3.5-1.5b.json \
        --data_path     data/processed/stark_prime_qa.jsonl
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ.setdefault("HF_HOME",            HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE",  HF_CACHE)

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.paired_dataloader import get_dataloader
from src.models.hooks import RepresentationCache, register_hooks
from src.profiling.self_patch_scan import PatchHook, _get_layer, count_layers
from src.evaluation.metrics import (
    generalization_accuracy, memorization_accuracy, headroom,
)


# ─────────────────────────────────────────────────────────────────────────────
# Oracle accuracy with best (l_s, l_t) patch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def oracle_accuracy(
    model,
    loader,
    l_s:    int,
    l_t:    int,
    device: torch.device,
    dtype:  torch.dtype,
    batch_size: int = 8,
    verbose:    bool = True,
) -> tuple[float, list[int], list[int]]:
    """
    Compute A_oracle: accuracy on P_gen after injecting oracle h_E^{l_s}(P_mem)
    into h_E^{l_t}(P_gen).

    Returns:
        (a_oracle, oracle_preds, targets)
    """
    model.eval()
    all_preds   = []
    all_targets = []

    if verbose:
        print(f"  Oracle patch: l_s={l_s} → l_t={l_t}")

    for batch in loader:
        mem_ids  = batch["mem_input_ids"].to(device)
        gen_ids  = batch["gen_input_ids"].to(device)
        mem_span = [(int(s[0]), int(s[1])) for s in batch["mem_span"]]
        gen_span = [(int(s[0]), int(s[1])) for s in batch["gen_span"]]
        tgt_ids  = batch["target_ids"]   # (B, max_entity_len)

        # Step 1: Extract h_E^{l_s}(P_mem) — the oracle patch
        cache   = RepresentationCache()
        handles = register_hooks(model, [l_s], cache, mem_span)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
            model(mem_ids)
        for h in handles:
            h.remove()

        if l_s not in cache.cache:
            # Hooks missed — fall back to no-patch inference
            with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
                logits = model(gen_ids).logits
            preds = logits[:, -1, :].argmax(dim=-1).cpu()
        else:
            patch = cache.cache[l_s]   # (B, D)

            # Step 2: Run P_gen with the patch injected at l_t
            span_start = [s[0] for s in gen_span]
            span_end   = [s[1] for s in gen_span]
            hook = PatchHook(patch, span_start, span_end)
            hook.register(model, l_t)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
                logits = model(gen_ids).logits
            hook.remove()

            preds = logits[:, -1, :].argmax(dim=-1).cpu()

        B = tgt_ids.size(0)
        for b in range(B):
            valid  = tgt_ids[b][tgt_ids[b] != -100]
            target = valid[0].item() if len(valid) > 0 else -1
            all_targets.append(target)
            all_preds.append(preds[b].item())

    a_oracle = generalization_accuracy(all_preds, all_targets)
    return a_oracle, all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
# Baseline accuracy (no patch) — for headroom computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def baseline_gen_accuracy(
    model,
    loader,
    device: torch.device,
    dtype:  torch.dtype,
) -> tuple[float, list[int], list[int]]:
    """Run P_gen inference with no patching. Returns (a_gen, preds, targets)."""
    model.eval()
    all_preds   = []
    all_targets = []

    for batch in loader:
        gen_ids = batch["gen_input_ids"].to(device)
        tgt_ids = batch["target_ids"]

        with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
            logits = model(gen_ids).logits
        preds = logits[:, -1, :].argmax(dim=-1).cpu()

        B = tgt_ids.size(0)
        for b in range(B):
            valid  = tgt_ids[b][tgt_ids[b] != -100]
            target = valid[0].item() if len(valid) > 0 else -1
            all_targets.append(target)
            all_preds.append(preds[b].item())

    return generalization_accuracy(all_preds, all_targets), all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
# Full oracle headroom analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_oracle_analysis(
    checkpoint:   str,
    base_model_id: str,
    profile_path: str,
    data_path:    str,
    device:       torch.device,
    dtype:        torch.dtype,
    batch_size:   int  = 8,
    hf_cache:     str  = "./hf_cache",
    out_dir:      str  | None = None,
    verbose:      bool = True,
) -> dict:
    """
    Full oracle headroom pipeline:
      1. Load checkpoint
      2. Get (l_s_best, l_t_best) from profile
      3. Measure A_gen (no patch) and A_oracle (with oracle patch)
      4. Compute headroom, save oracle_results.json
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Oracle Headroom Analysis")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Profile:    {profile_path}")
        print(f"{'='*60}\n")

    # ── Load profile ───────────────────────────────────────────────────────────
    with open(profile_path) as f:
        profile = json.load(f)

    l_s_best = profile.get("l_s_best")
    l_t      = profile.get("l_t")

    if l_s_best is None or l_t is None:
        raise ValueError(
            f"Profile missing l_s_best or l_t: {profile_path}. "
            "Run self_patch_scan first."
        )

    if verbose:
        print(f"  Using l_s_best={l_s_best}, l_t={l_t} from profile.")

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt_path = Path(checkpoint)
    is_peft   = (ckpt_path / "adapter_config.json").exists()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, cache_dir=hf_cache, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, cache_dir=hf_cache, torch_dtype=dtype,
            device_map="cuda" if device.type == "cuda" else "cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(ckpt_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path), cache_dir=hf_cache, torch_dtype=dtype,
            device_map="cuda" if device.type == "cuda" else "cpu",
            trust_remote_code=True,
        )

    model.eval()
    L = count_layers(model)
    if verbose:
        print(f"  Model loaded. L={L}  l_t={l_t}  l_s={l_s_best}")

    loader = get_dataloader(data_path, tokenizer, batch_size=batch_size, shuffle=False)

    # ── Baseline A_gen ─────────────────────────────────────────────────────────
    if verbose:
        print("\n  Running baseline A_gen (no patch) …")
    a_gen, gen_preds, gen_targets = baseline_gen_accuracy(model, loader, device, dtype)

    # ── Oracle A_oracle ────────────────────────────────────────────────────────
    if verbose:
        print("\n  Running oracle A_oracle (with patch) …")
    a_oracle, oracle_preds, _ = oracle_accuracy(
        model, loader, l_s_best, l_t, device, dtype, batch_size, verbose
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    gap = headroom(a_oracle, a_gen)

    if verbose:
        print(f"\n  ── Oracle Headroom Results ──────────────────────────")
        print(f"  A_gen   (no patch) = {a_gen:.3f}")
        print(f"  A_oracle (patched) = {a_oracle:.3f}")
        print(f"  Headroom           = {gap:.3f}")
        frac = gap / max(a_oracle, 1e-8)
        print(f"  Frac unclosed      = {frac:.1%}")
        print(f"  ────────────────────────────────────────────────────")

    results = {
        "checkpoint":  str(checkpoint),
        "model_id":    base_model_id,
        "l_s_best":    l_s_best,
        "l_t":         l_t,
        "A_gen":       round(a_gen, 4),
        "A_oracle":    round(a_oracle, 4),
        "headroom":    round(gap, 4),
        "n_examples":  len(gen_targets),
    }

    # ── Save ───────────────────────────────────────────────────────────────────
    save_dir = Path(out_dir) if out_dir else ckpt_path
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "oracle_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\n  Saved → {out_path}")

    # Free GPU memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Oracle headroom analysis")
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to trained checkpoint (epoch50 or LoRA adapter)")
    parser.add_argument("--model_id",     required=True)
    parser.add_argument("--profile_path", required=True,
                        help="layer_profile_<model_key>.json with l_s_best and l_t")
    parser.add_argument("--data_path",    default="data/processed/stark_prime_qa.jsonl")
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--hf_cache",     default="./hf_cache")
    parser.add_argument("--out_dir",      default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    run_oracle_analysis(
        args.checkpoint, args.model_id, args.profile_path,
        args.data_path, device, dtype,
        args.batch_size, args.hf_cache, args.out_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
