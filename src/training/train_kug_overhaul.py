"""
Fast Decoupled Trainer for KUG Experiments — v2 Completion-Only Loss
======================================================================
Trains Qwen/Qwen2.5-1.5B across 3 experimental regimes:
  1. --method baseline  : L = L_CE(P_mem)              for 50 epochs
  2. --method two_stage : L = L_CE(P_mem) Epochs 1-15, then L_CE(P_gen) Epochs 16-50
  3. --method joint     : L = L_CE(P_mem) + L_CE(P_gen) for 50 epochs

KEY CHANGE from v1: Cross-entropy loss is computed ONLY on answer completion tokens
(labels != -100), matching the Mem2Gen-71FF DataCollatorForCompletionOnlyLM.
This is what drives A_mem >= 95% convergence by epoch 10-15 as in Figure 7.

Decoupled execution:
  - Performs pure gradient optimization without inline generation loops.
  - Saves adapter weights at EVERY epoch (`checkpoint-epoch-X`).
  - Logs L_mem, L_gen, total loss, num_active_tokens, weight norms, update norms,
    and grad norms to W&B at every optimizer step and every epoch.
  - Writes `run_metadata.json` containing `wandb_run_id` for post-training eval.
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
import wandb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.data.paired_dataloader import get_kug_dataloader


def compute_weight_norm(model: nn.Module) -> float:
    """Compute L2 norm of trainable parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm_sq += p.detach().norm(2).item() ** 2
    return math.sqrt(total_norm_sq)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute L2 norm of gradients."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm_sq += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total_norm_sq)


def compute_completion_loss(model, input_ids, attention_mask, labels):
    """
    Compute completion-only cross-entropy loss.

    Labels are pre-masked: -100 for all prompt tokens, active for completion tokens.
    This means 100% of gradient signal trains on target answer tokens only.

    Args:
        model: The language model (with or without PEFT adapters).
        input_ids:       (B, L) full tokenized sequence.
        attention_mask:  (B, L) attention mask.
        labels:          (B, L) with -100 on prompt tokens, active on completion.

    Returns:
        Scalar loss (mean over active completion tokens, skipped if all -100).
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    return outputs.loss


def train_kug(args):
    print(f"=== Starting KUG Training Overhaul v2 (Completion-Only Loss) ===")
    print(f"Method: {args.method}")
    print(f"Base Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}, Grad Accum: {args.gradient_accumulation_steps}")
    print(f"LR: {args.learning_rate}, LoRA r={args.lora_r}, alpha={args.lora_alpha}")

    out_dir = Path(args.output_dir) / f"{args.method}_qwen2.5-1.5b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=f"train_v2_{args.method}_qwen2.5-1.5b",
        config={
            "method": args.method,
            "base_model": args.model_name_or_path,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "loss_mode": "completion_only",
            "answer_sep": "\\nAnswer:",
        },
    )
    wandb_run_id = wandb_run.id
    print(f"W&B Run Initialized ID: {wandb_run_id}")

    # Save run metadata for post-training evaluation resumption
    meta_path = out_dir / "run_metadata.json"
    metadata = {
        "wandb_run_id": wandb_run_id,
        "wandb_project": args.wandb_project,
        "method": args.method,
        "base_model": args.model_name_or_path,
        "dataset_path": args.dataset_path,
        "num_epochs": args.num_epochs,
        "output_dir": str(out_dir.resolve()),
        "loss_mode": "completion_only",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata -> {meta_path}")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Completion-only loss: pad on right

    print("Loading base model onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,        # NOTE: use `dtype=` not deprecated `torch_dtype=`
        device_map="cuda",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing to trade compute for activation memory.
    # This allows larger batch sizes without OOM on backward pass.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

    # Dataset already creates completion-only labels
    dataloader = get_kug_dataloader(
        args.dataset_path,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )

    # ── Verify completion masking on first N samples directly (no shuffle bias) ──
    print("\nLabel masking verification (first 5 samples from dataset):")
    underlying_dataset = dataloader.dataset
    n_verify = min(5, len(underlying_dataset))
    total_active_verify = 0
    for v_idx in range(n_verify):
        sample = underlying_dataset[v_idx]
        mem_labels = sample["mem_labels"]
        active = (mem_labels != -100).sum().item()
        total_tokens = len(mem_labels)
        print(f"  Sample[{v_idx:02d}] task={sample.get('task_type','?'):15s}  "
              f"active={active:3d}/{total_tokens}  ({100*active/total_tokens:.2f}%)")
        total_active_verify += active
    assert total_active_verify > 0, (
        f"ERROR: All labels are -100 across first {n_verify} samples! "
        f"Completion masking is broken or all {n_verify} samples were filtered."
    )
    print(f"  -> {total_active_verify} total active tokens across {n_verify} samples. Masking OK.\n")

    # Only pass TRAINABLE (LoRA) parameters to optimizer.
    # AdamW creates float32 m+v states for all params passed to it.
    # Passing all 1.56B params -> 12+ GB of optimizer state for FROZEN weights.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizer will track {len(trainable_params)} trainable parameter tensors.")
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    initial_weight_norm = compute_weight_norm(model)
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss_mem = 0.0
        epoch_loss_gen = 0.0
        epoch_loss_total = 0.0
        epoch_active_tokens_mem = 0
        epoch_active_tokens_gen = 0
        num_batches = 0

        # Determine phase for 2-stage training
        if args.method == "baseline":
            use_mem, use_gen = True, False
        elif args.method == "two_stage":
            if epoch <= 15:
                use_mem, use_gen = True, False
            else:
                use_mem, use_gen = False, True
        elif args.method == "joint":
            use_mem, use_gen = True, True
        else:
            raise ValueError(f"Unknown method: {args.method}")

        print(f"\n[Epoch {epoch:02d}/{args.num_epochs}] Phase: use_mem={use_mem}, use_gen={use_gen}", flush=True)

        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            mem_ids = batch["mem_input_ids"].cuda()
            mem_mask = batch["mem_attention_mask"].cuda()
            mem_labels = batch["mem_labels"].cuda()  # Completion-only labels
            gen_ids = batch["gen_input_ids"].cuda()
            gen_mask = batch["gen_attention_mask"].cuda()
            gen_labels = batch["gen_labels"].cuda()  # Completion-only labels

            # Count active (non-masked) tokens for diagnostics
            n_active_mem = (mem_labels != -100).sum().item()
            n_active_gen = (gen_labels != -100).sum().item()

            if use_mem and n_active_mem > 0:
                loss_mem = compute_completion_loss(model, mem_ids, mem_mask, mem_labels)
            else:
                loss_mem = torch.tensor(0.0, device="cuda")

            if use_gen and n_active_gen > 0:
                loss_gen = compute_completion_loss(model, gen_ids, gen_mask, gen_labels)
            else:
                loss_gen = torch.tensor(0.0, device="cuda")

            if args.method == "joint":
                total_loss = loss_mem + loss_gen
            elif use_mem:
                total_loss = loss_mem
            else:
                total_loss = loss_gen

            scaled_loss = total_loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            epoch_loss_mem += loss_mem.item()
            epoch_loss_gen += loss_gen.item()
            epoch_loss_total += total_loss.item()
            epoch_active_tokens_mem += n_active_mem
            epoch_active_tokens_gen += n_active_gen
            num_batches += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = compute_grad_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                curr_weight_norm = compute_weight_norm(model)
                update_norm = abs(curr_weight_norm - initial_weight_norm)

                wandb.log({
                    "step": global_step,
                    "train/step_loss_mem": loss_mem.item(),
                    "train/step_loss_gen": loss_gen.item(),
                    "train/step_loss_total": total_loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/weight_norm": curr_weight_norm,
                    "train/update_norm": update_norm,
                    "train/active_tokens_mem": n_active_mem,
                    "train/active_tokens_gen": n_active_gen,
                })

        avg_loss_mem = epoch_loss_mem / max(1, num_batches)
        avg_loss_gen = epoch_loss_gen / max(1, num_batches)
        avg_loss_total = epoch_loss_total / max(1, num_batches)
        avg_active_mem = epoch_active_tokens_mem / max(1, num_batches)
        avg_active_gen = epoch_active_tokens_gen / max(1, num_batches)

        curr_weight_norm = compute_weight_norm(model)
        update_norm = abs(curr_weight_norm - initial_weight_norm)

        print(
            f"Epoch {epoch:02d}/{args.num_epochs:02d} | "
            f"L_mem: {avg_loss_mem:.4f} | L_gen: {avg_loss_gen:.4f} | "
            f"Total: {avg_loss_total:.4f} | "
            f"Avg active tokens mem/gen: {avg_active_mem:.1f}/{avg_active_gen:.1f}",
            flush=True,
        )

        wandb.log({
            "epoch": epoch,
            "train/epoch_loss_mem": avg_loss_mem,
            "train/epoch_loss_gen": avg_loss_gen,
            "train/epoch_loss_total": avg_loss_total,
            "train/epoch_weight_norm": curr_weight_norm,
            "train/epoch_update_norm": update_norm,
            "train/epoch_avg_active_tokens_mem": avg_active_mem,
            "train/epoch_avg_active_tokens_gen": avg_active_gen,
        })

        # Save per-epoch adapter weights
        ckpt_dir = out_dir / f"checkpoint-epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Saved adapter checkpoint -> {ckpt_dir}")

    wandb.finish()
    print("=== Training Completed Successfully ===")


def main():
    parser = argparse.ArgumentParser(description="Train KUG overhaul models v2 (completion-only loss)")
    parser.add_argument("--method", choices=["baseline", "two_stage", "joint"], required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset_path", type=str, default="data/processed/kug_dataset_all.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/kug_overhaul_v2")
    parser.add_argument("--wandb_project", type=str, default="kug_overhaul_qwen1.5b")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    train_kug(args)


if __name__ == "__main__":
    main()
