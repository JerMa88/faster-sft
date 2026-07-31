"""
Phase 1.5 — Empirical Layer Profiling for Qwen2.5-1.5B (or any model)
========================================================================
Runs three metrics to determine l_s_early, l_s_late, l_t:
  1. Per-layer linear probe accuracy (sklearn logistic regression)
  2. Logit-lens KL divergence between consecutive layers
  3. Self-patching gain scan (requires --checkpoint if not just probing)

Saves: data/processed/layer_profile.json

Usage:
    python scripts/run_profiling.py --model_id Qwen/Qwen2.5-1.5B
    python scripts/run_profiling.py --model_id Qwen/Qwen2.5-1.5B \\
        --checkpoint outputs/runs/.../checkpoint_epoch3
"""

import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ["HF_HOME"]            = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"]  = HF_CACHE

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.models.hooks import RepresentationCache, register_hooks
from src.data.paired_dataloader import get_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_layers(model) -> int:
    """
    Count decoder layers regardless of model architecture.

    IMPORTANT: Do NOT use model.base_model — transformers.PreTrainedModel exposes
    a base_model @property on ALL models, not just PEFT wrappers. Using it causes
    double-unwrapping (ForCausalLM → inner Model → AttributeError on .model).
    """
    # Only unwrap real PEFT wrappers (they have peft_config attribute)
    base = model
    if hasattr(base, "peft_config"):          # PeftModel
        base = base.base_model.model
    # Standard: ForCausalLM → .model → .layers
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return len(base.model.layers)
    # Bare inner model with direct .layers
    if hasattr(base, "layers"):
        return len(base.layers)
    raise ValueError(
        f"Cannot determine layer count for {type(model).__name__}. "
        f"Visible attrs: {[a for a in dir(base) if not a.startswith('_')]}"
    )


def collect_hidden_states(model, tokenizer, data_path, n_samples, device, dtype):
    """
    Run the model on first n_samples of data_path (mem prompts only),
    collect hidden states from ALL layers at the entity span positions.

    Returns:
        hidden_per_layer: dict[int -> np.ndarray (n_samples, hidden_size)]
        labels:           list[str] of target_entity strings
    """
    L = count_layers(model)
    loader = get_dataloader(data_path, tokenizer, batch_size=8, shuffle=False)

    all_layers  = list(range(L))
    cache       = RepresentationCache()
    hiddens     = {l: [] for l in all_layers}
    labels      = []
    total       = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if total >= n_samples:
                break
            mem_ids  = batch["mem_input_ids"].to(device)
            mem_span = [(int(s[0]), int(s[1])) for s in batch["mem_span"]]

            cache.clear()
            handles = register_hooks(model, all_layers, cache, mem_span)
            with torch.amp.autocast("cuda", dtype=dtype,
                                    enabled=(device.type == "cuda")):
                model(mem_ids)
            for h in handles:
                h.remove()

            bsz = mem_ids.size(0)
            for l in all_layers:
                if l in cache.cache:
                    rep = cache.cache[l].float().cpu().numpy()  # (B, D)
                    hiddens[l].append(rep)

            # Use tokenizer to decode target entities for labels
            for b in range(bsz):
                tgt = batch["target_ids"][b]
                valid = tgt[tgt != -100]
                labels.append(tokenizer.decode(valid, skip_special_tokens=True))

            total += bsz
            if total % 50 == 0:
                print(f"  Collected {total}/{n_samples} samples …")

    # Stack into arrays
    hiddens_np = {}
    for l in all_layers:
        if hiddens[l]:
            hiddens_np[l] = np.concatenate(hiddens[l], axis=0)[:n_samples]

    return hiddens_np, labels[:n_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1: Linear probe accuracy
# ─────────────────────────────────────────────────────────────────────────────

def run_linear_probe(hiddens_np: dict, labels: list) -> dict:
    """
    For each layer, train logistic regression on 80% and eval on 20%.
    Returns probe_acc[layer_idx] = float.
    """
    print("\n  [Metric 1] Running per-layer linear probes …")
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    n  = len(y)
    split = int(0.8 * n)

    probe_acc = {}
    for l, X in sorted(hiddens_np.items()):
        # Normalise to unit sphere (cosine-style)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms

        X_train, y_train = X_norm[:split], y[:split]
        X_test,  y_test  = X_norm[split:], y[split:]

        if len(np.unique(y_train)) < 2:
            probe_acc[l] = 0.0
            continue

        clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs",
                                 multi_class="auto", n_jobs=-1)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        probe_acc[l] = float(acc)
        print(f"    Layer {l:3d}: probe_acc = {acc:.3f}")

    return probe_acc


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2: Logit-lens KL divergence
# ─────────────────────────────────────────────────────────────────────────────

def run_logit_lens(model, tokenizer, data_path, n_samples, device, dtype) -> dict:
    """
    Project residual stream at each layer through the unembedding matrix.
    Compute KL(p^l || p^{l-1}) and p^l(y*).
    Returns dict with 'kl_per_layer' and 'p_correct_per_layer'.
    """
    print("\n  [Metric 2] Running logit-lens KL sweep …")

    # Get the unembedding matrix (lm_head or embed_out)
    base = model
    if hasattr(base, "base_model"):
        base = base.base_model.model
    # Access lm_head weight
    lm_head = None
    for name in ["lm_head", "embed_out"]:
        if hasattr(model, name):
            lm_head = getattr(model, name)
            break
    if lm_head is None:
        print("  [WARNING] lm_head not found; skipping logit-lens metric.")
        return {"kl_per_layer": {}, "p_correct_per_layer": {}}

    # Also need layer norm before lm_head if present
    final_ln = None
    for name in ["model.norm", "norm", "final_layer_norm", "model.final_layer_norm"]:
        parts = name.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            final_ln = obj
            break
        except AttributeError:
            pass

    L = count_layers(model)
    loader = get_dataloader(data_path, tokenizer, batch_size=4, shuffle=False)
    cache = RepresentationCache()

    kl_per_layer      = {l: [] for l in range(1, L)}
    p_correct_layer   = {l: [] for l in range(L)}
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if total >= n_samples:
                break
            mem_ids   = batch["mem_input_ids"].to(device)
            mem_spans = [(int(s[0]), int(s[1])) for s in batch["mem_span"]]
            tgt_ids   = batch["target_ids"]  # (B, max_entity_len)

            cache.clear()
            handles = register_hooks(model, list(range(L)), cache, mem_spans)
            with torch.amp.autocast("cuda", dtype=dtype,
                                    enabled=(device.type == "cuda")):
                model(mem_ids)
            for h in handles:
                h.remove()

            bsz = mem_ids.size(0)
            # Compute logit distributions at each layer
            prev_probs = None
            for l in range(L):
                if l not in cache.cache:
                    continue
                h = cache.cache[l].float()  # (B, D)
                # Apply final layer norm
                if final_ln is not None:
                    try:
                        h = final_ln(h)
                    except Exception:
                        pass
                logits = lm_head(h)           # (B, vocab)
                probs  = torch.softmax(logits, dim=-1)  # (B, vocab)

                # p_correct: probability of the first answer token
                for b in range(bsz):
                    valid = tgt_ids[b][tgt_ids[b] != -100]
                    if len(valid) > 0:
                        ans_tok = valid[0].item()
                        p_correct_layer[l].append(probs[b, ans_tok].item())

                if prev_probs is not None:
                    # KL(p^l || p^{l-1})
                    kl = (probs * (probs.log() - prev_probs.log().clamp(min=-1e9))).sum(-1)
                    kl_per_layer[l].extend(kl.cpu().tolist())

                prev_probs = probs

            total += bsz

    kl_mean  = {l: float(np.mean(v)) for l, v in kl_per_layer.items() if v}
    p_c_mean = {l: float(np.mean(v)) for l, v in p_correct_layer.items() if v}
    return {"kl_per_layer": kl_mean, "p_correct_per_layer": p_c_mean}


# ─────────────────────────────────────────────────────────────────────────────
# Selection rules
# ─────────────────────────────────────────────────────────────────────────────

def select_layers(probe_acc: dict, kl_data: dict, L: int,
                  theta_early: float = 0.6,
                  theta_late:  float = 0.85) -> tuple[int, int, int]:
    """
    Apply selection rules from implementation_plan.md:
      l_s_early = first layer > theta_early probe acc
      l_s_late  = last  layer > theta_late  probe acc
      l_t       = estimated as ~0.5L (override when self-patching scan runs)
    Cross-check l_s_early with logit-lens p_correct.
    """
    layers_sorted = sorted(probe_acc.keys())

    # l_s_early
    l_s_early_candidates = [l for l in layers_sorted if probe_acc[l] > theta_early]
    if l_s_early_candidates:
        l_s_early = l_s_early_candidates[0]
    else:
        # fall back: first local peak
        best = max(probe_acc, key=probe_acc.get)
        l_s_early = best
        print(f"  [WARN] No layer exceeds theta_early={theta_early}; "
              f"using peak layer {l_s_early} (acc={probe_acc[l_s_early]:.3f})")

    # Cross-validate with logit-lens
    p_c = kl_data.get("p_correct_per_layer", {})
    if p_c:
        ll_candidates = [l for l in sorted(p_c) if p_c.get(l, 0) > 0.5]
        if ll_candidates:
            l_s_early_ll = ll_candidates[0]
            if abs(l_s_early_ll - l_s_early) > 2:
                print(f"  [WARN] Logit-lens suggests l_s_early={l_s_early_ll}, "
                      f"probe suggests {l_s_early}. Difference > 2 layers; "
                      f"using probe value (logit-lens as cross-check).")

    # l_s_late
    l_s_late_candidates = [l for l in layers_sorted if probe_acc[l] > theta_late]
    if l_s_late_candidates:
        l_s_late = l_s_late_candidates[-1]
    else:
        l_s_late = layers_sorted[-1]
        print(f"  [WARN] No layer exceeds theta_late={theta_late}; "
              f"using last layer {l_s_late}.")

    # l_t: reasoning bottleneck — default to ~0.5L, will be overridden by self-patching
    l_t = int(0.50 * L)
    print(f"\n  ── Selection results ──────────────────────────────")
    print(f"  l_s_early = {l_s_early}  (probe_acc={probe_acc.get(l_s_early, 'N/A'):.3f})")
    print(f"  l_s_late  = {l_s_late}   (probe_acc={probe_acc.get(l_s_late, 'N/A'):.3f})")
    print(f"  l_t       = {l_t}  (estimated; run self-patching scan to refine)")
    return l_s_early, l_s_late, l_t


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1.5: Layer profiling")
    parser.add_argument("--model_id", type=str, required=True,
                        help="HF model ID (e.g. Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--data_path", type=str,
                        default="data/processed/stark_prime_qa.jsonl")
    parser.add_argument("--n_probe_samples", type=int, default=200,
                        help="Samples for probe + logit-lens (160 train / 40 test)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Warmup checkpoint path for self-patching scan "
                             "(skipped if not provided)")
    parser.add_argument("--out_path", type=str,
                        default="data/processed/layer_profile.json")
    parser.add_argument("--hf_cache", type=str, default="./hf_cache")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache

    # ── Device ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}  dtype: {dtype}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading {args.model_id} for profiling …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, cache_dir=args.hf_cache, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_path = args.checkpoint if args.checkpoint else args.model_id

    # Nanbeige4.2-3B has a rope_scaling dict without the required 'type' key.
    # Patch the config before loading to avoid KeyError in _init_rope().
    extra_kwargs = {}
    if "nanbeige" in args.model_id.lower():
        from transformers import AutoConfig
        nb_cfg = AutoConfig.from_pretrained(
            model_path, cache_dir=args.hf_cache, trust_remote_code=True
        )
        if (
            hasattr(nb_cfg, "rope_scaling")
            and isinstance(nb_cfg.rope_scaling, dict)
            and "type" not in nb_cfg.rope_scaling
        ):
            nb_cfg.rope_scaling["type"] = "linear"
            print("  [Nanbeige] Patched rope_scaling: added type='linear'")
        extra_kwargs["config"] = nb_cfg

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.hf_cache,
        dtype=dtype,                                  # replaces deprecated torch_dtype
        device_map="cuda" if device.type == "cuda" else "cpu",
        trust_remote_code=True,
        **extra_kwargs,
    )
    L = count_layers(model)
    print(f"  Model layers (L) = {L}")

    # ── Collect hidden states ──────────────────────────────────────────────────
    hiddens, labels = collect_hidden_states(
        model, tokenizer, args.data_path,
        args.n_probe_samples, device, dtype
    )

    # ── Metric 1: Linear probe ────────────────────────────────────────────────
    probe_acc = run_linear_probe(hiddens, labels)

    # ── Metric 2: Logit-lens ──────────────────────────────────────────────────
    kl_data = run_logit_lens(
        model, tokenizer, args.data_path,
        min(100, args.n_probe_samples), device, dtype
    )

    # ── Selection rules ────────────────────────────────────────────────────────
    l_s_early, l_s_late, l_t = select_layers(probe_acc, kl_data, L)

    # ── Save profile ───────────────────────────────────────────────────────────
    profile = {
        "model":                  args.model_id,
        "L":                      L,
        "l_s_early":              l_s_early,
        "l_s_late":               l_s_late,
        "l_t":                    l_t,
        "l_t_source":             "heuristic_0.5L (run self-patching to refine)",
        "probe_accuracy_per_layer":   {str(k): v for k, v in probe_acc.items()},
        "logit_lens_kl_per_layer":    {str(k): v for k, v in kl_data["kl_per_layer"].items()},
        "logit_lens_p_correct_layer": {str(k): v for k, v in
                                       kl_data["p_correct_per_layer"].items()},
    }

    out_path = ROOT / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n  Layer profile saved → {out_path}")
    print(f"  l_s_early={l_s_early}, l_s_late={l_s_late}, l_t={l_t}")
    print("  NOTE: l_t is an estimate. For the empirical optimum, run the "
          "self-patching scan after the 3-epoch warmup checkpoint is available.")


if __name__ == "__main__":
    main()
