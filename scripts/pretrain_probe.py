"""
Step A — Pretrain the Linear Probe φ* for ProbeLoss
=====================================================
Trains a linear probe φ*: R^d → R^|V| on late-layer entity representations
extracted from the un-finetuned (or warmup-checkpoint) model.

Procedure (per implementation_plan.md Part 2, Loss Variant 2):
  1. Load model at l_s_late from layer_profile_<model_key>.json
  2. Run all P_mem examples through the model, cache h_E^{l_s_late}
  3. Train nn.Linear(hidden_size, vocab_size) for 10 epochs with CE loss
     where labels = first answer token id of each example
  4. Evaluate accuracy on held-out 20% split
  5. Save probe state dict + metadata to data/processed/probe_phi_<model_key>.pt

The saved probe is loaded by train_sft.py when --loss_variant probe or hybrid.

Usage:
    python scripts/pretrain_probe.py --model_key qwen3.5-1.5b
    python scripts/pretrain_probe.py --model_key llama3.2-3b --data_path data/processed/stark_prime_qa.jsonl
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
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.hooks import RepresentationCache, register_hooks
from src.data.paired_dataloader import get_dataloader

# ── Model key → HF ID map (mirrors run_sft.sh) ───────────────────────────────
MODEL_IDS = {
    "llama3.2-3b":    "meta-llama/Llama-3.2-3B-Instruct",
    "qwen3.5-2b":   "Qwen/Qwen3.5-2B",
    "gemma2-2b":       "google/gemma-2-2b-it",
    "antares-1b":     "fdtn-ai/antares-1b",
    "nanbeige4.2-3b": "Nanbeige/Nanbeige4.2-3B",
    "lfm2.5-1.2b":    "LiquidAI/LFM2.5-1.2B-Base",
}


def count_layers(model) -> int:
    base = model
    if hasattr(base, "base_model"):
        base = base.base_model.model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return len(base.model.layers)
    if hasattr(base, "layers"):
        return len(base.layers)
    raise ValueError("Cannot detect layer count.")


def collect_probe_data(model, tokenizer, loader, l_s_late: int,
                       device: torch.device, dtype: torch.dtype):
    """
    Run all P_mem examples through the model (no_grad), collect:
      - h_mem: hidden state at l_s_late for the entity span  (N, D)
      - y_tok: first valid answer token id per example       (N,)
    """
    cache = RepresentationCache()
    all_h = []
    all_y = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            mem_ids  = batch["mem_input_ids"].to(device)
            mem_span = [(int(s[0]), int(s[1])) for s in batch["mem_span"]]
            tgt_ids  = batch["target_ids"]   # (B, max_entity_len)

            cache.clear()
            handles = register_hooks(model, [l_s_late], cache, mem_span)
            with torch.amp.autocast("cuda", dtype=dtype,
                                    enabled=(device.type == "cuda")):
                model(mem_ids)
            for h in handles:
                h.remove()

            if l_s_late not in cache.cache:
                continue

            h = cache.cache[l_s_late].float().cpu()   # (B, D)
            all_h.append(h)

            # First valid answer token per example
            for b in range(tgt_ids.size(0)):
                valid = tgt_ids[b][tgt_ids[b] != -100]
                tok   = valid[0].item() if len(valid) > 0 else 0
                all_y.append(tok)

    X = torch.cat(all_h, dim=0)       # (N, D)
    y = torch.tensor(all_y, dtype=torch.long)  # (N,)
    return X, y


def train_probe(X: torch.Tensor, y: torch.Tensor,
                vocab_size: int, epochs: int = 10,
                lr: float = 1e-3, val_frac: float = 0.2,
                device: torch.device = torch.device("cpu")):
    """
    Train a linear probe nn.Linear(D, vocab_size) on (X, y).
    Returns the trained probe (on CPU) and the final validation accuracy.
    """
    N   = X.size(0)
    D   = X.size(1)
    n_val   = max(1, int(N * val_frac))
    n_train = N - n_val

    # Shuffle
    perm      = torch.randperm(N)
    X, y      = X[perm], y[perm]

    X_train, y_train = X[:n_train].to(device), y[:n_train].to(device)
    X_val,   y_val   = X[n_train:].to(device), y[n_train:].to(device)

    # Normalise to unit sphere (matching the profiling convention)
    norms = X_train.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    X_train_n = X_train / norms
    norms_v   = X_val.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    X_val_n   = X_val / norms_v

    probe = nn.Linear(D, vocab_size, bias=True).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train_n, y_train)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    print(f"\n  Training probe: D={D}, vocab_size={vocab_size}, "
          f"n_train={n_train}, n_val={n_val}")

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            logits = probe(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        # Val accuracy
        probe.eval()
        with torch.no_grad():
            preds   = probe(X_val_n).argmax(dim=-1)
            val_acc = (preds == y_val).float().mean().item()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{epochs} | "
                  f"train_loss={total_loss/n_train:.4f} | "
                  f"val_acc={val_acc:.3f}")

    print(f"\n  Final probe val_acc = {val_acc:.3f}")
    if val_acc < 0.60:
        print("  [WARN] val_acc < 0.60 — probe is underperforming. "
              "Check data quality or increase epochs.")
    else:
        print("  [OK] val_acc ≥ 0.60 — probe ready to freeze.")

    return probe.cpu(), val_acc


def main():
    parser = argparse.ArgumentParser(description="Pretrain ProbeLoss φ*")
    parser.add_argument("--model_key", required=True,
                        choices=list(MODEL_IDS.keys()),
                        help="Short model key (e.g. qwen3.5-1.5b)")
    parser.add_argument("--model_id", default=None,
                        help="Override HF model ID (default: looked up from model_key)")
    parser.add_argument("--data_path", default="data/processed/stark_prime_qa.jsonl")
    parser.add_argument("--layer_profile",
                        default=None,
                        help="Path to layer_profile_<model_key>.json. "
                             "Auto-detected if not given.")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional SFT warmup checkpoint path. "
                             "Uses base model if not provided.")
    parser.add_argument("--probe_epochs", type=int, default=10)
    parser.add_argument("--probe_lr",     type=float, default=1e-3)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--hf_cache",     default="./hf_cache")
    parser.add_argument("--out_dir",      default="data/processed")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache

    # ── Resolve model ID ───────────────────────────────────────────────────────
    model_id = args.model_id or MODEL_IDS[args.model_key]
    print(f"\n{'='*60}")
    print(f"  Pretrain probe φ* for: {args.model_key}")
    print(f"  Model ID : {model_id}")
    print(f"  Data     : {args.data_path}")
    print(f"{'='*60}\n")

    # ── Device ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"  Device: {device}  dtype: {dtype}")

    # ── Layer profile ──────────────────────────────────────────────────────────
    profile_path = args.layer_profile or \
        str(ROOT / "data" / "processed" / f"layer_profile_{args.model_key}.json")

    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        l_s_late = profile["l_s_late"]
        print(f"  l_s_late={l_s_late} (from {profile_path})")
    else:
        print(f"  [WARN] No layer profile at {profile_path}. "
              f"Using heuristic l_s_late = 0.85 × L (will detect after model load).")
        l_s_late = None   # resolved after model load

    # ── Load model ─────────────────────────────────────────────────────────────
    model_path = args.checkpoint or model_id
    print(f"  Loading model: {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=args.hf_cache, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.hf_cache,
        torch_dtype=dtype,
        device_map="cuda" if device.type == "cuda" else "cpu",
        trust_remote_code=True,
    )

    if l_s_late is None:
        L        = count_layers(model)
        l_s_late = max(1, int(0.85 * L))
        print(f"  Heuristic l_s_late={l_s_late} for L={L}")

    vocab_size = model.config.vocab_size
    print(f"  vocab_size={vocab_size}  hidden_size={model.config.hidden_size}")

    # ── Collect representations ────────────────────────────────────────────────
    print(f"\n  Collecting h_E^{{l_s_late={l_s_late}}} for all P_mem examples …")
    loader = get_dataloader(
        args.data_path, tokenizer,
        batch_size=args.batch_size, shuffle=False
    )
    X, y = collect_probe_data(model, tokenizer, loader, l_s_late, device, dtype)
    print(f"  Collected {X.size(0)} examples  (X: {tuple(X.shape)}, y: {tuple(y.shape)})")

    # Free GPU memory before training the probe
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Train probe ────────────────────────────────────────────────────────────
    probe_device = device   # probe is tiny — keep on same device
    probe, val_acc = train_probe(
        X, y,
        vocab_size=vocab_size,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        device=probe_device,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_phi_{args.model_key}.pt"

    torch.save({
        "state_dict":  probe.state_dict(),
        "model_key":   args.model_key,
        "model_id":    model_id,
        "l_s_late":    l_s_late,
        "hidden_size": X.size(1),
        "vocab_size":  vocab_size,
        "val_acc":     val_acc,
        "probe_epochs": args.probe_epochs,
        "data_path":   args.data_path,
    }, out_path)

    print(f"\n  φ* saved → {out_path}")
    print(f"  val_acc = {val_acc:.3f} | hidden_size = {X.size(1)} | vocab_size = {vocab_size}")
    print("  Load with:  torch.load(out_path)['state_dict']")


if __name__ == "__main__":
    main()
