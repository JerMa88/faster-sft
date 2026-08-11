"""
Fast Decoupled Trainer for KUG Experiments
============================================
Trains Qwen/Qwen2.5-1.5B across 3 experimental regimes:
  1. --method baseline  : L = L_CE(P_mem) for 50 epochs
  2. --method two_stage : L = L_CE(P_mem) for Epochs 1-15, then L = L_CE(P_gen) for Epochs 16-50
  3. --method joint     : L = L_CE(P_mem) + L_CE(P_gen) for 50 epochs

Decoupled execution:
  - Performs pure gradient optimization without inline generation loops.
  - Saves adapter weights at EVERY epoch (`checkpoint-epoch-X`).
  - Logs L_mem, L_gen, total loss, weight norms, update norms, and grad norms to W&B.
  - Writes `run_metadata.json` containing `wandb_run_id` for post-training evaluation resumption.
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


def compute_sequence_loss(model, input_ids, attention_mask):
    """Compute token-level causal language modeling cross-entropy loss."""
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


def train_kug(args):
    print(f"=== Starting KUG Training Overhaul ===")
    print(f"Method: {args.method}")
    print(f"Base Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}, Grad Accum: {args.gradient_accumulation_steps}")

    out_dir = Path(args.output_dir) / f"{args.method}_qwen2.5-1.5b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=f"train_{args.method}_qwen2.5-1.5b",
        config={
            "method": args.method,
            "base_model": args.model_name_or_path,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        }
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
    }
    with open(meta_path, "w") as f:
        json.dumps(metadata, f, indent=2)
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata -> {meta_path}")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataloader = get_kug_dataloader(args.dataset_path, tokenizer, batch_size=args.batch_size, max_length=args.max_length, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    initial_weight_norm = compute_weight_norm(model)
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss_mem = 0.0
        epoch_loss_gen = 0.0
        epoch_loss_total = 0.0
        num_batches = 0

        # Determine phase for 2-stage training
        # Stage 1 (1-15): pure mem; Stage 2 (16-50): pure gen
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

        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            mem_ids = batch["mem_input_ids"].cuda()
            mem_mask = batch["mem_attention_mask"].cuda()
            gen_ids = batch["gen_input_ids"].cuda()
            gen_mask = batch["gen_attention_mask"].cuda()

            loss_mem = compute_sequence_loss(model, mem_ids, mem_mask) if use_mem else torch.tensor(0.0, device="cuda")
            loss_gen = compute_sequence_loss(model, gen_ids, gen_mask) if use_gen else torch.tensor(0.0, device="cuda")

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
            num_batches += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = compute_grad_norm(model)
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
                })

        avg_loss_mem = epoch_loss_mem / max(1, num_batches)
        avg_loss_gen = epoch_loss_gen / max(1, num_batches)
        avg_loss_total = epoch_loss_total / max(1, num_batches)

        curr_weight_norm = compute_weight_norm(model)
        update_norm = abs(curr_weight_norm - initial_weight_norm)

        print(f"Epoch {epoch:02d}/{args.num_epochs:02d} | Loss Mem: {avg_loss_mem:.4f} | Loss Gen: {avg_loss_gen:.4f} | Total: {avg_loss_total:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/epoch_loss_mem": avg_loss_mem,
            "train/epoch_loss_gen": avg_loss_gen,
            "train/epoch_loss_total": avg_loss_total,
            "train/epoch_weight_norm": curr_weight_norm,
            "train/epoch_update_norm": update_norm,
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
    parser = argparse.ArgumentParser(description="Train KUG overhaul models")
    parser.add_argument("--method", choices=["baseline", "two_stage", "joint"], required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset_path", type=str, default="data/processed/kug_dataset_all.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/kug_overhaul")
    parser.add_argument("--wandb_project", type=str, default="kug_overhaul_qwen1.5b")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
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
