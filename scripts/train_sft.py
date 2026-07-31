"""
Alignment-Aware SFT Training Script
=====================================
Usage:
    python scripts/train_sft.py --model_id Qwen/Qwen3.5-2B \
        --data_path data/processed/stark_prime_qa.jsonl \
        --loss_variant contrastive \
        --epochs 50 --lambda_align 0.1 --warmup_epochs 3

Supports 6 models via --model_id; meant to be launched via SLURM.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.paired_dataloader import get_dataloader
from src.models.hooks import RepresentationCache, register_hooks
from src.training.losses import contrastive_loss, rep_distill_loss

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Alignment-aware SFT training")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B",
                   help="HuggingFace model ID (e.g. Qwen/Qwen3.5-2B)")
    p.add_argument("--data_path", type=str,
                   default="data/processed/stark_prime_qa.jsonl")
    p.add_argument("--loss_variant", type=str,
                   choices=["baseline", "rep_distill", "contrastive",
                             "probe", "hybrid"],
                   default="contrastive",
                   help="Which alignment loss variant to use")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lambda_align", type=float, default=0.1,
                   help="Weight of the alignment loss (λ)")
    p.add_argument("--warmup_epochs", type=int, default=3,
                   help="Epochs before alignment loss activates (K)")
    p.add_argument("--batch_size", type=int, default=0,
                   help="Batch size per GPU (0=auto-select from VRAM)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--layer_profile", type=str,
                   default="data/processed/layer_profile.json",
                   help="Path to layer_profile.json from Phase 1.5")
    p.add_argument("--probe_path", type=str, default=None,
                   help="Path to probe_phi_<model_key>.pt (for probe/hybrid variants). "
                        "Auto-detected from --model_key if not given.")
    p.add_argument("--model_key", type=str, default=None,
                   help="Short model key (e.g. qwen3.5-2b) for auto-finding probe file")
    p.add_argument("--out_dir", type=str, default="outputs/runs",
                   help="Directory for checkpoints and metrics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf_cache", type=str, default="./hf_cache",
                   help="Local HuggingFace model cache directory")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# VRAM-aware batch size selection
# ─────────────────────────────────────────────────────────────────────────────

def auto_batch_size(device: torch.device, model_param_bytes: int) -> int:
    """
    Heuristic: on an A100-80G, a 1.5B bfloat16 model uses ~3 GB of weights.
    Leave 20 GB for activations and allow batch_size ∝ remaining VRAM.
    Dual forward pass doubles the activation memory, so we halve the estimate.
    """
    if device.type != "cuda":
        return 4
    total_vram = torch.cuda.get_device_properties(device).total_memory
    weight_gb   = model_param_bytes / 1e9
    avail_gb    = total_vram / 1e9 - weight_gb - 4.0   # 4 GB headroom
    # Each 512-token bfloat16 sequence at hidden_size 1536 ≈ ~0.15 GB per batch item (dual pass)
    est_batch = max(8, int(avail_gb / 0.30))
    # Cap at 64 for numerical stability
    return min(est_batch, 64)


# ─────────────────────────────────────────────────────────────────────────────
# Layer index loading
# ─────────────────────────────────────────────────────────────────────────────

# Reasonable defaults for each model family (L = total layers, 0-indexed)
# These are used when layer_profile.json has not been produced yet.
# Format: model_family_key -> (l_s_early, l_s_late, l_t)
LAYER_DEFAULTS = {
    # Qwen3.5-2B (estimated layers)
    "qwen3.5-2b": (4, 24, 13),
    # Llama-3.2-3B: 28 layers
    "llama-3.2-3b":  (4, 24, 13),
    # Gemma-4-E4B: 34 layers
    "gemma-4-e4b":   (5, 29, 16),
    # Antares-1B (Granite-based): 24 layers
    "antares-1b":    (3, 20, 11),
    # Nanbeige4.2-3B: Looped Transformer, 32 effective layers
    "nanbeige4.2-3b":(5, 27, 15),
    # LFM2.5-1.2B (Liquid state-space hybrid): 24 layers
    "lfm2.5-1.2b":  (3, 20, 11),
}

def resolve_layer_indices(model_id: str, profile_path: str):
    """Return (l_s_early, l_s_late, l_t) from profile file or defaults."""
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        l_s_early = profile["l_s_early"]
        l_s_late  = profile["l_s_late"]
        l_t       = profile["l_t"]
        print(f"  Loaded layer profile from {profile_path}: "
              f"l_s_early={l_s_early}, l_s_late={l_s_late}, l_t={l_t}")
        return l_s_early, l_s_late, l_t

    # Fallback: match by model_id substring
    mid_lower = model_id.lower()
    for key, vals in LAYER_DEFAULTS.items():
        if key in mid_lower or any(part in mid_lower for part in key.split("-")):
            print(f"  [WARNING] No layer_profile.json found. Using defaults for "
                  f"'{key}': l_s_early={vals[0]}, l_s_late={vals[1]}, l_t={vals[2]}")
            return vals

    # Generic fallback: use ~10%, ~85%, ~50% of detected depth
    print("  [WARNING] No layer defaults matched; using 0.1L/0.85L/0.5L heuristic.")
    return None  # signal to detect from model after loading


def detect_layer_count(model) -> int:
    """Count decoder layers in a loaded model (handles PeftModel wrapper)."""
    base = model
    if hasattr(base, "base_model"):
        base = base.base_model.model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return len(base.model.layers)
    if hasattr(base, "layers"):
        return len(base.layers)
    raise ValueError("Cannot detect layer count from model architecture.")


# ─────────────────────────────────────────────────────────────────────────────
# LoRA config per model family
# ─────────────────────────────────────────────────────────────────────────────

def get_lora_target_modules(model_id: str) -> list[str]:
    """Return the right projection names for each architecture."""
    mid = model_id.lower()
    if "llama" in mid or "qwen" in mid or "nanbeige" in mid:
        return ["q_proj", "v_proj", "o_proj", "k_proj"]
    if "gemma" in mid:
        return ["q_proj", "v_proj", "o_proj", "k_proj"]
    if "antares" in mid or "granite" in mid:
        return ["q_proj", "v_proj", "o_proj"]
    if "lfm" in mid or "liquid" in mid:
        # LFM uses attention-like projections
        return ["q_proj", "v_proj", "o_proj"]
    # Fallback
    return ["q_proj", "v_proj", "o_proj"]


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype  = torch.bfloat16
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("  [WARNING] CUDA not available — falling back to CPU / float32.")
        device = torch.device("cpu")
        dtype  = torch.float32

    # ── Output directory ───────────────────────────────────────────────────────
    model_slug = args.model_id.replace("/", "--")
    run_name   = f"{model_slug}_{args.loss_variant}_lam{args.lambda_align}_seed{args.seed}"
    out_dir    = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*64}")
    print(f"  Model      : {args.model_id}")
    print(f"  Loss       : {args.loss_variant}  λ={args.lambda_align}  K={args.warmup_epochs}")
    print(f"  Data       : {args.data_path}")
    print(f"  Output     : {out_dir}")
    print(f"{'='*64}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    os.environ["HF_HOME"]              = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"]   = args.hf_cache
    os.environ["HF_DATASETS_CACHE"]    = args.hf_cache

    print(f"Loading tokenizer: {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.hf_cache,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_id} ...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_id, cache_dir=args.hf_cache, trust_remote_code=True)
    
    # --- MONKEY PATCH FOR NANBEIGE ---
    import transformers
    if hasattr(transformers, "DynamicCache"):
        if not hasattr(transformers.DynamicCache, "from_legacy_cache"):
            transformers.DynamicCache.from_legacy_cache = lambda past_key_values: transformers.DynamicCache()
        if not hasattr(transformers.DynamicCache, "to_legacy_cache"):
            transformers.DynamicCache.to_legacy_cache = lambda self: ()
    # ---------------------------------

    if args.model_key and "nanbeige" in args.model_key.lower():
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            if "type" not in config.rope_scaling:
                config.rope_scaling["type"] = "linear"
            if "factor" not in config.rope_scaling:
                config.rope_scaling["factor"] = 1.0

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        cache_dir=args.hf_cache,
        torch_dtype=dtype,
        device_map="cuda" if device.type == "cuda" else "cpu",
        trust_remote_code=True,
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    target_modules = get_lora_target_modules(args.model_id)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,      # standard: alpha = 2r
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Layer indices ──────────────────────────────────────────────────────────
    indices = resolve_layer_indices(args.model_id, args.layer_profile)
    if indices is None:
        L = detect_layer_count(model)
        l_s_early = max(1, int(0.10 * L))
        l_s_late  = max(1, int(0.85 * L))
        l_t       = max(1, int(0.50 * L))
        print(f"  Heuristic layers for L={L}: "
              f"l_s_early={l_s_early}, l_s_late={l_s_late}, l_t={l_t}")
    else:
        l_s_early, l_s_late, l_t = indices

    # ── Batch size ────────────────────────────────────────────────────────────
    if args.batch_size == 0:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        batch_size  = auto_batch_size(device, param_bytes)
    else:
        batch_size = args.batch_size
    print(f"  Batch size : {batch_size}")

    # ── Load φ* probe (for probe / hybrid variants) ───────────────────────────
    phi_probe    = None
    phi_probe_fn = None   # callable: h_gen (B,D) → loss scalar
    if args.loss_variant in ("probe", "hybrid"):
        # Auto-detect probe path from model_key
        probe_path = args.probe_path
        if probe_path is None and args.model_key:
            probe_path = str(ROOT / "data" / "processed" /
                             f"probe_phi_{args.model_key}.pt")
        if probe_path and os.path.exists(probe_path):
            ckpt = torch.load(probe_path, map_location=device)
            hidden_size = ckpt["hidden_size"]
            vocab_size  = ckpt["vocab_size"]
            phi_probe = torch.nn.Linear(hidden_size, vocab_size, bias=True).to(device)
            phi_probe.load_state_dict(ckpt["state_dict"])
            phi_probe.eval()
            for p in phi_probe.parameters():
                p.requires_grad_(False)   # frozen
            print(f"  φ* probe loaded: {probe_path} "
                  f"(val_acc={ckpt.get('val_acc', 'N/A'):.3f})")

            _probe_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            def phi_probe_fn(h_gen_local, tgt_ids_local):
                """CE loss through frozen φ* on first valid answer token."""
                # h_gen_local: (B, D); tgt_ids_local: (B, max_entity_len)
                h_norm = torch.nn.functional.normalize(h_gen_local.float(), dim=-1)
                logits = phi_probe(h_norm)   # (B, vocab)
                # label = first valid token per example
                labels = tgt_ids_local[:, 0].to(device)  # (B,)
                labels = labels.masked_fill(labels == -100, -100)
                return _probe_loss_fn(logits, labels)
        else:
            print(f"  [WARN] φ* probe not found at '{probe_path}'. "
                  f"Falling back to rep_distill for probe/hybrid variants. "
                  f"Run: python scripts/pretrain_probe.py --model_key {args.model_key or '<key>'}")

    # Enable gradient checkpointing to save VRAM (trades compute for memory)
    if device.type == "cuda":
        model.gradient_checkpointing_enable()

    # ── DataLoader ────────────────────────────────────────────────────────────
    loader = get_dataloader(args.data_path, tokenizer, batch_size=batch_size)
    print(f"  Dataset    : {len(loader.dataset)} pairs  "
          f"({len(loader)} batches of {batch_size})")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    # ── AMP scaler (only for float16/bfloat16 on CUDA) ───────────────────────
    use_amp   = (device.type == "cuda")
    scaler    = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

    # ── Cache ─────────────────────────────────────────────────────────────────
    rep_cache = RepresentationCache()

    # ── Metrics ───────────────────────────────────────────────────────────────
    all_metrics   = []
    ckpt_epochs   = {1, 3, 5, 10, 15, 20, 30, 50}

    model.train()
    t_train_start = time.time()

    for epoch in range(args.epochs):
        epoch_loss       = 0.0
        epoch_ce_loss    = 0.0
        epoch_align_loss = 0.0
        t_epoch_start    = time.time()
        align_active     = (epoch >= args.warmup_epochs) and \
                           (args.loss_variant != "baseline")

        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)

            mem_ids  = batch["mem_input_ids"].to(device)
            gen_ids  = batch["gen_input_ids"].to(device)
            mem_span = batch["mem_span"]
            gen_span = batch["gen_span"]

            # ── Pass 1: P_mem → h_mem at l_s_early and l_s_late (no grad) ──
            rep_cache.clear()
            mem_spans_list = [(int(s[0]), int(s[1])) for s in mem_span]
            handles_mem = register_hooks(
                model, [l_s_early, l_s_late], rep_cache, mem_spans_list
            )
            with torch.no_grad():
                model(mem_ids)
            for h in handles_mem:
                h.remove()
            h_mem_early = rep_cache.cache.get(l_s_early)  # (B, D) or None
            h_mem_late  = rep_cache.cache.get(l_s_late)

            # ── Pass 2: P_gen → CE loss + h_gen at l_t ───────────────────────
            rep_cache.clear()
            gen_spans_list = [(int(s[0]), int(s[1])) for s in gen_span]
            handles_gen = register_hooks(
                model, [l_t], rep_cache, gen_spans_list
            )

            # Mask labels: only supervise the answer portion (after entity span)
            labels = gen_ids.clone()
            for b_idx in range(gen_ids.size(0)):
                s_start, s_end = int(gen_span[b_idx][0]), int(gen_span[b_idx][1])
                labels[b_idx, :s_start] = -100
                labels[b_idx, s_end:]   = -100

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                outputs  = model(gen_ids, labels=labels)
                ce_loss  = outputs.loss

            for h in handles_gen:
                h.remove()
            h_gen = rep_cache.cache.get(l_t)  # (B, D)

            # ── Alignment loss ────────────────────────────────────────────────
            if align_active and h_gen is not None and h_mem_late is not None:
                tgt_ids = batch["target_ids"]  # (B, max_entity_len)
                if args.loss_variant == "rep_distill":
                    # Average RepDist over both source layers
                    align_loss  = rep_distill_loss(h_mem_early, h_gen) * 0.5
                    align_loss += rep_distill_loss(h_mem_late, h_gen)  * 0.5
                elif args.loss_variant == "contrastive":
                    # InfoNCE — average over both source layers, τ=0.07
                    align_loss  = contrastive_loss(h_mem_early, h_gen,
                                                   temperature=0.07) * 0.5
                    align_loss += contrastive_loss(h_mem_late, h_gen,
                                                   temperature=0.07) * 0.5
                elif args.loss_variant == "probe":
                    if phi_probe_fn is not None:
                        align_loss = phi_probe_fn(h_gen, tgt_ids)
                    else:
                        align_loss = rep_distill_loss(h_mem_late, h_gen)  # fallback
                elif args.loss_variant == "hybrid":
                    # α·RepDist + (1-α)·ProbeLoss, α=0.5
                    rd_loss = (rep_distill_loss(h_mem_early, h_gen) * 0.5
                               + rep_distill_loss(h_mem_late, h_gen) * 0.5)
                    if phi_probe_fn is not None:
                        pl_loss = phi_probe_fn(h_gen, tgt_ids)
                    else:
                        pl_loss = contrastive_loss(h_mem_late, h_gen, temperature=0.07)
                    align_loss = 0.5 * rd_loss + 0.5 * pl_loss
                else:
                    align_loss = rep_distill_loss(h_mem_late, h_gen)  # safety fallback

                total_loss = ce_loss + args.lambda_align * align_loss
                epoch_align_loss += align_loss.item()
            else:
                total_loss = ce_loss

            # ── Backward ──────────────────────────────────────────────────────
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss    += total_loss.item()
            epoch_ce_loss += ce_loss.item()

            if step % 20 == 0:
                n_steps = len(loader)
                print(f"  Epoch {epoch+1:3d}/{args.epochs} "
                      f"Step {step:4d}/{n_steps} | "
                      f"Loss {total_loss.item():.4f} "
                      f"(CE {ce_loss.item():.4f}"
                      f"{f', Align {align_loss.item():.4f}' if align_active and h_gen is not None else ''})"
                      f"  LR {scheduler.get_last_lr()[0]:.2e}")

        # ── Epoch summary ──────────────────────────────────────────────────────
        n = len(loader)
        epoch_secs = time.time() - t_epoch_start
        summary = {
            "epoch":       epoch + 1,
            "total_loss":  epoch_loss    / n,
            "ce_loss":     epoch_ce_loss / n,
            "align_loss":  epoch_align_loss / n if align_active else 0.0,
            "align_active": align_active,
            "epoch_secs":  round(epoch_secs, 1),
        }
        all_metrics.append(summary)
        print(f"\n  ── Epoch {epoch+1} done in {epoch_secs:.0f}s │ "
              f"Avg loss: {summary['total_loss']:.4f} ──\n")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) in ckpt_epochs:
            ckpt_path = out_dir / f"checkpoint_epoch{epoch+1}"
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            print(f"  Checkpoint saved → {ckpt_path}")

        # ── Save running metrics ───────────────────────────────────────────────
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

    total_time = time.time() - t_train_start
    print(f"\n{'='*64}")
    print(f"  Training complete in {total_time/3600:.2f} hours")
    print(f"  Metrics → {metrics_path}")
    print(f"  Model   → {out_dir}")
    print(f"{'='*64}\n")

    return all_metrics


if __name__ == "__main__":
    args = parse_args()
    train(args)
