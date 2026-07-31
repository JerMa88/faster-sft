"""
src/evaluation/evaluator.py
============================
Checkpoint evaluator — runs a trained model on P_mem and P_gen sets,
computes A_mem and A_gen, builds the accuracy-vs-epoch curve from a
run directory, and saves eval_results.json.

Usage (standalone):
    python -m src.evaluation.evaluator \
        --run_dir  outputs/runs/qwen3.5-1.5b/stark_prime/baseline-... \
        --model_id Qwen/Qwen3.5-1.5B \
        --data_path data/processed/stark_prime_qa.jsonl

Design:
  - Each training run saves checkpoints at epochs {1, 3, 5, 10, 15, 20, 30, 50}
  - Evaluator loads them in order, runs greedy first-token prediction,
    records A_mem and A_gen, then writes one JSON summary per run.
  - Works with both base models and PEFT/LoRA adapters (auto-detected).
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ.setdefault("HF_HOME",            HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE",  HF_CACHE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.paired_dataloader import get_dataloader
from src.evaluation.metrics import (
    memorization_accuracy, generalization_accuracy,
    convergence_epoch, auc_curve, compute_all_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(
    checkpoint_path: str,
    base_model_id:   str,
    device:          torch.device,
    dtype:           torch.dtype,
    hf_cache:        str,
):
    """
    Load model from checkpoint_path.
    Auto-detects LoRA adapters (adapter_config.json present).
    """
    ckpt = Path(checkpoint_path)
    is_peft = (ckpt / "adapter_config.json").exists()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, cache_dir=hf_cache, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft:
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft not installed — cannot load LoRA checkpoint.")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, cache_dir=hf_cache,
            torch_dtype=dtype,
            device_map="cuda" if device.type == "cuda" else "cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(ckpt))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), cache_dir=hf_cache,
            torch_dtype=dtype,
            device_map="cuda" if device.type == "cuda" else "cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt), cache_dir=hf_cache, trust_remote_code=True
        )

    model.eval()
    return model, tokenizer


@torch.no_grad()
def _run_inference(
    model,
    loader,
    device: torch.device,
    dtype:  torch.dtype,
    kind:   str = "gen",   # "mem" or "gen"
) -> tuple[list[int], list[int]]:
    """
    Greedy first-token decoding on all examples in loader.
    Returns (predictions, targets) both as lists of token IDs.
    """
    id_key   = f"{kind}_input_ids"
    span_key = f"{kind}_span"

    all_preds   = []
    all_targets = []

    for batch in loader:
        input_ids = batch[id_key].to(device)
        tgt_ids   = batch["target_ids"]     # (B, max_entity_len)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
            logits = model(input_ids).logits  # (B, seq, vocab)

        preds = logits[:, -1, :].argmax(dim=-1).cpu()  # (B,)

        for b in range(tgt_ids.size(0)):
            valid = tgt_ids[b][tgt_ids[b] != -100]
            target = valid[0].item() if len(valid) > 0 else -1
            all_targets.append(target)
            all_preds.append(preds[b].item())

    return all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
# Single-checkpoint evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: str,
    base_model_id:   str,
    data_path:       str,
    device:          torch.device,
    dtype:           torch.dtype,
    batch_size:      int = 16,
    hf_cache:        str = "./hf_cache",
    verbose:         bool = True,
) -> dict:
    """
    Evaluate a single checkpoint. Returns metrics dict with A_mem and A_gen.
    """
    if verbose:
        print(f"  Evaluating: {Path(checkpoint_path).name} …")

    model, tokenizer = _load_model(
        checkpoint_path, base_model_id, device, dtype, hf_cache
    )
    loader = get_dataloader(data_path, tokenizer, batch_size=batch_size, shuffle=False)

    mem_preds, mem_targets = _run_inference(model, loader, device, dtype, kind="mem")
    gen_preds, gen_targets = _run_inference(model, loader, device, dtype, kind="gen")

    a_mem = memorization_accuracy(mem_preds, mem_targets)
    a_gen = generalization_accuracy(gen_preds, gen_targets)

    if verbose:
        print(f"    A_mem={a_mem:.3f}  A_gen={a_gen:.3f}")

    # Free GPU memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint":  str(checkpoint_path),
        "A_mem":       round(a_mem, 4),
        "A_gen":       round(a_gen, 4),
        "n_examples":  len(mem_preds),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full run evaluation (all checkpoints → accuracy curve)
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_EPOCHS = [1, 3, 5, 10, 15, 20, 30, 50]


def evaluate_run(
    run_dir:       str,
    base_model_id: str,
    data_path:     str,
    device:        torch.device,
    dtype:         torch.dtype,
    batch_size:    int   = 16,
    hf_cache:      str   = "./hf_cache",
    threshold:     float = 0.95,
    verbose:       bool  = True,
) -> dict:
    """
    Evaluate all saved checkpoints in a run directory.
    Returns a summary dict with the full accuracy curve and convergence metrics.
    """
    run_dir  = Path(run_dir)
    checkpts = sorted(
        [d for d in run_dir.glob("checkpoint_epoch*") if d.is_dir()],
        key=lambda d: int(d.name.replace("checkpoint_epoch", ""))
    )

    if not checkpts:
        raise FileNotFoundError(f"No checkpoint_epoch* directories in {run_dir}")

    if verbose:
        print(f"\n  Run: {run_dir.name}")
        print(f"  Found {len(checkpts)} checkpoints: "
              f"{[c.name for c in checkpts]}")

    epoch_results = []
    for ckpt in checkpts:
        epoch_num = int(ckpt.name.replace("checkpoint_epoch", ""))
        result    = evaluate_checkpoint(
            str(ckpt), base_model_id, data_path,
            device, dtype, batch_size, hf_cache, verbose
        )
        result["epoch"] = epoch_num
        epoch_results.append(result)

    # Build accuracy curves
    epochs    = [r["epoch"] for r in epoch_results]
    a_mem_curve = [r["A_mem"] for r in epoch_results]
    a_gen_curve = [r["A_gen"] for r in epoch_results]

    # Convergence on A_gen
    t_conv = convergence_epoch(a_gen_curve, threshold, start_epoch=epochs[0])
    auc    = auc_curve(a_gen_curve)

    # Check baseline gate: A_mem ≥ 0.978 at epoch 3 checkpoint
    gate_result = None
    for r in epoch_results:
        if r["epoch"] == 3:
            gate_result = r["A_mem"] >= 0.978
            if verbose:
                status = "✅ PASS" if gate_result else "❌ FAIL"
                print(f"\n  Baseline gate (A_mem @ epoch3 ≥ 0.978): "
                      f"{r['A_mem']:.3f} → {status}")

    summary = {
        "run_dir":        str(run_dir),
        "base_model_id":  base_model_id,
        "data_path":      data_path,
        "epochs":         epochs,
        "A_mem_curve":    [round(a, 4) for a in a_mem_curve],
        "A_gen_curve":    [round(a, 4) for a in a_gen_curve],
        "A_mem_final":    round(a_mem_curve[-1], 4),
        "A_gen_final":    round(a_gen_curve[-1], 4),
        "T_conv":         t_conv,
        "AUC":            round(auc, 4),
        "threshold":      threshold,
        "baseline_gate":  gate_result,
        "per_epoch":      epoch_results,
    }

    # Save
    out_path = run_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"\n  Saved → {out_path}")
        print(f"  A_gen_final={summary['A_gen_final']:.3f}  "
              f"T_conv={t_conv}  AUC={auc:.3f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Cross-run comparison (for the paper table)
# ─────────────────────────────────────────────────────────────────────────────

def compare_runs(
    run_dirs:      list[str],
    base_model_id: str,
    data_path:     str,
    device:        torch.device,
    dtype:         torch.dtype,
    out_path:      str = "outputs/comparison.json",
    **eval_kwargs,
) -> dict:
    """
    Evaluate multiple run directories and produce a comparison table.
    Saves to out_path. Returns a dict keyed by run_dir name.
    """
    comparison = {}
    for run_dir in run_dirs:
        name = Path(run_dir).name
        try:
            summary = evaluate_run(
                run_dir, base_model_id, data_path, device, dtype, **eval_kwargs
            )
            comparison[name] = {
                "A_gen_final": summary["A_gen_final"],
                "T_conv":      summary["T_conv"],
                "AUC":         summary["AUC"],
                "A_mem_final": summary["A_mem_final"],
            }
        except Exception as e:
            comparison[name] = {"error": str(e)}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved → {out_path}")
    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT run checkpoints")
    parser.add_argument("--run_dir",    required=True,
                        help="Path to a run directory (outputs/runs/.../)")
    parser.add_argument("--model_id",   required=True,
                        help="Base HF model ID (for tokenizer + PEFT base)")
    parser.add_argument("--data_path",  default="data/processed/stark_prime_qa.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold",  type=float, default=0.95)
    parser.add_argument("--hf_cache",   default="./hf_cache")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    evaluate_run(
        args.run_dir, args.model_id, args.data_path,
        device, dtype, args.batch_size, args.hf_cache, args.threshold,
        verbose=True,
    )


if __name__ == "__main__":
    main()
