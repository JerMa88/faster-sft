"""
Self-Patching Gain Scan (Metric 2 from implementation_plan.md §Part 1)
=======================================================================
Runs an L×L layer-pair scan on a post-warmup checkpoint to identify l_t:
the layer where injecting the correct entity representation maximally
improves multi-hop reasoning (generalization) accuracy.

For each pair (l_s, l_t):
    Patch h_E^{l_t}(P_gen) ← h_E^{l_s}(P_mem)   [oracle injection]
    Measure ΔAcc = patched_acc - baseline_acc

l_t = argmax_{l_t}  max_{l_s}  A[l_s, l_t]

This directly operationalizes Figure 5 from arXiv:2607.08393.

Usage (standalone):
    python -m src.profiling.self_patch_scan \
        --checkpoint outputs/runs/qwen3.5-1.5b/.../checkpoint_epoch3 \
        --model_id Qwen/Qwen3.5-1.5B \
        --data_path data/processed/stark_prime_qa.jsonl \
        --profile_path data/processed/layer_profile_qwen3.5-1.5b.json \
        --n_samples 100
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ.setdefault("HF_HOME",            HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE",  HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE",   HF_CACHE)

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.hooks import RepresentationCache, register_hooks
from src.data.paired_dataloader import get_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# Patching hook
# ─────────────────────────────────────────────────────────────────────────────

class PatchHook:
    """
    A forward hook that replaces entity-span hidden states at layer l_t
    with a precomputed patch tensor.

    The hook is registered on the l_t-th decoder layer output. After the
    layer computes its output, we overwrite entity positions with the patch.
    """

    def __init__(
        self,
        patch:      torch.Tensor,   # (B, D) — precomputed patch from P_mem at l_s
        span_start: list[int],      # per-example start indices (entity span)
        span_end:   list[int],      # per-example end indices (entity span, exclusive)
    ):
        self.patch      = patch          # (B, D) — mean-pooled entity rep
        self.span_start = span_start
        self.span_end   = span_end
        self.handle: Optional[torch.utils.hooks.RemovableHook] = None

    def __call__(self, module, input, output):
        """
        Called after l_t forward pass. output may be a tuple (hidden, attn, ...).
        We modify the hidden states in-place for the entity span positions.
        """
        if isinstance(output, tuple):
            hidden = output[0]            # (B, seq_len, D)
            rest   = output[1:]
        else:
            hidden = output
            rest   = None

        hidden = hidden.clone()
        B = hidden.size(0)
        patch_2d = self.patch.to(hidden.device, hidden.dtype)  # (B, D)

        for b in range(min(B, len(self.span_start))):
            s, e = self.span_start[b], self.span_end[b]
            if s >= e or e > hidden.size(1):
                continue
            # Write the patch to all entity-span token positions
            hidden[b, s:e, :] = patch_2d[b].unsqueeze(0).expand(e - s, -1)

        if rest is not None:
            return (hidden,) + rest
        return hidden

    def register(self, model, layer_idx: int) -> "PatchHook":
        """Register this hook on the layer_idx-th decoder layer."""
        layer = _get_layer(model, layer_idx)
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_layer(model, idx: int):
    """Return the idx-th decoder layer regardless of model wrapper."""
    base = model
    if hasattr(base, "base_model"):
        base = base.base_model.model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers[idx]
    if hasattr(base, "layers"):
        return base.layers[idx]
    raise ValueError(f"Cannot locate layer {idx} in model.")


def count_layers(model) -> int:
    base = model
    if hasattr(base, "base_model"):
        base = base.base_model.model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return len(base.model.layers)
    if hasattr(base, "layers"):
        return len(base.layers)
    raise ValueError("Cannot count layers.")


def _greedy_first_token(model, input_ids: torch.Tensor, device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
    """Run a forward pass and return the argmax of the last-position logits."""
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype,
                                enabled=(device.type == "cuda")):
            logits = model(input_ids).logits   # (B, seq_len, vocab)
    return logits[:, -1, :].argmax(dim=-1)     # (B,)


def _extract_entity_reps(model, input_ids, spans, layer_indices,
                         device, dtype) -> dict[int, torch.Tensor]:
    """
    Run the model with no_grad, collect mean-pooled entity-span reps
    at each requested layer.

    Returns: {layer_idx: tensor (B, D)}
    """
    cache   = RepresentationCache()
    handles = register_hooks(model, layer_indices, cache,
                             [(int(s[0]), int(s[1])) for s in spans])
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype,
                                enabled=(device.type == "cuda")):
            model(input_ids.to(device))
    for h in handles:
        h.remove()
    return {l: cache.cache[l].clone() for l in layer_indices if l in cache.cache}


# ─────────────────────────────────────────────────────────────────────────────
# Core scan
# ─────────────────────────────────────────────────────────────────────────────

def run_self_patch_scan(
    model,
    tokenizer,
    loader,
    L:        int,
    device:   torch.device,
    dtype:    torch.dtype,
    n_samples: int = 100,
    verbose:  bool = True,
) -> np.ndarray:
    """
    Full L×L self-patching scan.

    For each (l_s, l_t) pair:
        1. Get baseline acc on P_gen (no patching)
        2. Patch h_E^{l_t}(P_gen) ← h_E^{l_s}(P_mem) via forward hook
        3. Measure patched acc
        ΔAcc = patched_acc - baseline_acc

    Returns:
        gain_matrix: np.ndarray (L, L), gain_matrix[l_s, l_t] = mean ΔAcc
    """
    model.eval()

    # ── Collect all examples up to n_samples ──────────────────────────────────
    all_mem_ids  = []
    all_gen_ids  = []
    all_mem_span = []
    all_gen_span = []
    all_tgt_tok  = []   # first answer token per example

    for batch in loader:
        bsz = batch["mem_input_ids"].size(0)
        for b in range(bsz):
            if len(all_mem_ids) >= n_samples:
                break
            all_mem_ids.append(batch["mem_input_ids"][b])
            all_gen_ids.append(batch["gen_input_ids"][b])
            all_mem_span.append((int(batch["mem_span"][b][0]),
                                 int(batch["mem_span"][b][1])))
            all_gen_span.append((int(batch["gen_span"][b][0]),
                                 int(batch["gen_span"][b][1])))
            tids = batch["target_ids"][b]
            valid = tids[tids != -100]
            all_tgt_tok.append(valid[0].item() if len(valid) > 0 else 0)
        if len(all_mem_ids) >= n_samples:
            break

    N = len(all_mem_ids)
    if N == 0:
        raise RuntimeError("No samples collected from loader.")
    if verbose:
        print(f"  Self-patch scan: {N} examples, {L}×{L} = {L*L} layer pairs")

    # Stack into tensors (work example-by-example for large L)
    # Precompute baseline accuracy (no patching)
    baseline_correct = np.zeros(N, dtype=float)
    for i in range(N):
        gen_ids = all_gen_ids[i].unsqueeze(0).to(device)
        pred    = _greedy_first_token(model, gen_ids, device, dtype)
        baseline_correct[i] = float(pred[0].item() == all_tgt_tok[i])

    baseline_acc = baseline_correct.mean()
    if verbose:
        print(f"  Baseline gen acc (no patch): {baseline_acc:.3f}")

    # Precompute entity reps at all source layers for all P_mem examples
    # Shape: {l_s: (N, D)}
    if verbose:
        print("  Precomputing P_mem representations at all layers …")

    all_layers = list(range(L))
    # Process in mini-batches to avoid OOM
    mem_reps: dict[int, list] = {l: [] for l in all_layers}

    BATCH = 8
    for start in range(0, N, BATCH):
        end  = min(start + BATCH, N)
        mids = torch.stack(all_mem_ids[start:end]).to(device)
        spans = all_mem_span[start:end]
        reps  = _extract_entity_reps(model, mids, spans, all_layers, device, dtype)
        for l in all_layers:
            if l in reps:
                mem_reps[l].append(reps[l].cpu())
            else:
                mem_reps[l].append(torch.zeros(end - start,
                                               model.config.hidden_size))

    mem_reps_t: dict[int, torch.Tensor] = {
        l: torch.cat(vs, dim=0) for l, vs in mem_reps.items()
    }   # {l_s: (N, D)}

    # ── Main L×L scan ──────────────────────────────────────────────────────────
    gain_matrix = np.zeros((L, L), dtype=float)   # [l_s, l_t]

    for l_t in range(L):
        if verbose and l_t % 4 == 0:
            print(f"  Scanning l_t={l_t}/{L-1} …")

        for l_s in range(L):
            patch_src = mem_reps_t[l_s]   # (N, D)
            correct = np.zeros(N, dtype=float)

            for start in range(0, N, BATCH):
                end   = min(start + BATCH, N)
                b_sz  = end - start

                gen_ids = torch.stack(all_gen_ids[start:end]).to(device)
                patch   = patch_src[start:end].to(device)
                spans   = all_gen_span[start:end]

                span_start = [s[0] for s in spans]
                span_end   = [s[1] for s in spans]

                hook = PatchHook(patch, span_start, span_end)
                hook.register(model, l_t)

                try:
                    pred = _greedy_first_token(model, gen_ids, device, dtype)
                finally:
                    hook.remove()

                for b in range(b_sz):
                    correct[start + b] = float(pred[b].item() == all_tgt_tok[start + b])

            gain_matrix[l_s, l_t] = correct.mean() - baseline_acc

    return gain_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Selection rule: l_t from the heatmap
# ─────────────────────────────────────────────────────────────────────────────

def select_l_t(gain_matrix: np.ndarray) -> tuple[int, int, float]:
    """
    l_t = argmax_{l_t} max_{l_s} A[l_s, l_t]

    Returns: (l_s_best, l_t_best, max_gain)
    """
    max_per_lt = gain_matrix.max(axis=0)   # (L,) — best gain achievable at each l_t
    l_t_best   = int(np.argmax(max_per_lt))
    l_s_best   = int(np.argmax(gain_matrix[:, l_t_best]))
    max_gain   = float(gain_matrix[l_s_best, l_t_best])
    return l_s_best, l_t_best, max_gain


# ─────────────────────────────────────────────────────────────────────────────
# Update layer profile
# ─────────────────────────────────────────────────────────────────────────────

def update_profile(
    profile_path: str,
    l_t_empirical: int,
    l_s_best: int,
    max_gain: float,
    gain_matrix: np.ndarray,
    checkpoint: str,
):
    """Update layer_profile.json in-place with empirical l_t from self-patch scan."""
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
    else:
        profile = {}

    profile["l_t"]              = l_t_empirical
    profile["l_t_source"]       = f"self_patching_scan (checkpoint: {checkpoint})"
    profile["l_s_best"]         = l_s_best
    profile["max_patch_gain"]   = round(max_gain, 4)
    profile["self_patch_heatmap"] = gain_matrix.tolist()

    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"  Updated profile: l_t={l_t_empirical} "
          f"(was heuristic; max_gain={max_gain:.3f}) → {profile_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot (optional — skipped gracefully if matplotlib not available)
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(profile_path: str, out_dir: str):
    """Generate ProbeAcc(l), KL(l), and heatmap figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [INFO] matplotlib not available — skipping plots.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(profile_path) as f:
        profile = json.load(f)

    # ── ProbeAcc(l) ─────────────────────────────────────────────────────────
    probe_acc = profile.get("probe_accuracy_per_layer", {})
    if probe_acc:
        layers = sorted(probe_acc.keys(), key=int)
        accs   = [probe_acc[l] for l in layers]
        l_s_early = profile.get("l_s_early")
        l_s_late  = profile.get("l_s_late")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([int(l) for l in layers], accs, linewidth=2, color="#4C72B0")
        ax.set_xlabel("Layer"); ax.set_ylabel("Probe Accuracy")
        ax.set_title(f"Linear Probe Accuracy — {profile.get('model', '')}")
        if l_s_early is not None:
            ax.axvline(l_s_early, color="green",  linestyle="--",
                       label=f"l_s_early={l_s_early}")
        if l_s_late is not None:
            ax.axvline(l_s_late,  color="orange", linestyle="--",
                       label=f"l_s_late={l_s_late}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "probe_acc.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'probe_acc.png'}")

    # ── KL(l) ───────────────────────────────────────────────────────────────
    kl_data = profile.get("logit_lens_kl_per_layer", {})
    if kl_data:
        layers = sorted(kl_data.keys(), key=int)
        kls    = [kl_data[l] for l in layers]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([int(l) for l in layers], kls, linewidth=2, color="#DD8452")
        ax.set_xlabel("Layer"); ax.set_ylabel("KL Divergence")
        ax.set_title(f"Logit-Lens KL — {profile.get('model', '')}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "logit_lens_kl.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'logit_lens_kl.png'}")

    # ── Self-patching heatmap ────────────────────────────────────────────────
    heatmap = profile.get("self_patch_heatmap")
    if heatmap:
        A    = np.array(heatmap)   # (L, L) — [l_s, l_t]
        l_t  = profile.get("l_t")

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(A, aspect="auto", cmap="RdYlGn",
                       vmin=-0.1, vmax=max(0.1, A.max()))
        plt.colorbar(im, ax=ax, label="ΔAcc (patched − baseline)")
        ax.set_xlabel("l_t (target layer)")
        ax.set_ylabel("l_s (source layer)")
        ax.set_title(f"Self-Patching Gain A[l_s, l_t] — {profile.get('model', '')}")
        if l_t is not None:
            ax.axvline(l_t, color="white", linewidth=2, linestyle="--",
                       label=f"l_t={l_t}")
            ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / "self_patch_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir / 'self_patch_heatmap.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Self-patching gain scan (Phase 1.5 Metric 2)")
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to warmup checkpoint (e.g. checkpoint_epoch3)")
    parser.add_argument("--model_id",      required=True,
                        help="Base HF model ID (for tokenizer)")
    parser.add_argument("--data_path",     default="data/processed/stark_prime_qa.jsonl")
    parser.add_argument("--profile_path",  default=None,
                        help="layer_profile_<model_key>.json to update in-place")
    parser.add_argument("--n_samples",     type=int, default=100)
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--plots_dir",     default="outputs/plots",
                        help="Directory for profile figures")
    parser.add_argument("--hf_cache",      default="./hf_cache")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}  dtype: {dtype}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, cache_dir=args.hf_cache, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        cache_dir=args.hf_cache,
        torch_dtype=dtype,
        device_map="cuda" if device.type == "cuda" else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    L = count_layers(model)
    print(f"Model layers L={L}")

    loader = get_dataloader(
        args.data_path, tokenizer, batch_size=args.batch_size, shuffle=False
    )

    # Scan
    gain_matrix = run_self_patch_scan(
        model, tokenizer, loader, L, device, dtype,
        n_samples=args.n_samples, verbose=True
    )

    l_s_best, l_t_best, max_gain = select_l_t(gain_matrix)
    print(f"\n  ── Self-patching results ──────────────────────────────")
    print(f"  l_t (empirical) = {l_t_best}")
    print(f"  l_s_best        = {l_s_best}")
    print(f"  max ΔAcc        = {max_gain:.3f}")
    print(f"  ──────────────────────────────────────────────────────")

    if gain_matrix.max() <= 0:
        warnings.warn(
            "All ΔAcc values are ≤ 0 — the checkpoint may not have memorized "
            "enough facts yet (need A_mem > 0.95). Consider running more warmup epochs.",
            UserWarning
        )

    # Update profile
    if args.profile_path and os.path.exists(args.profile_path):
        update_profile(
            args.profile_path, l_t_best, l_s_best, max_gain,
            gain_matrix, args.checkpoint
        )
        # Generate figures
        save_plots(args.profile_path, args.plots_dir)
    else:
        # Just save the raw heatmap as JSON
        out_path = ROOT / "data" / "processed" / "self_patch_heatmap.json"
        with open(out_path, "w") as f:
            json.dump({
                "l_t": l_t_best, "l_s_best": l_s_best,
                "max_gain": max_gain, "heatmap": gain_matrix.tolist()
            }, f, indent=2)
        print(f"  Heatmap saved → {out_path}")


if __name__ == "__main__":
    main()
