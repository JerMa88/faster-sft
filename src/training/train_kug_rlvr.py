"""
KUG 2-Stage RLVR Trainer (GRPO with Verifiable Rewards)
========================================================
Stage 2 Optimization on P_gen Queries (Epochs 16–50)
Starting from Stage 1 Checkpoint (Epoch 15).

Algorithm: Group Relative Policy Optimization (GRPO)
  - Zero supervised answer tokens are provided in prompts or loss.
  - Samples K=4 rollouts per query prompt at temperature T=0.7.
  - Evaluates verifiable binary reward R in {0, 1} via `rlvr_verifier`.
  - Computes group-normalized advantages A_i = (R_i - mean(R)) / (std(R) + eps).
  - Optimizes clipped policy gradient objective.
  - Computes token-level KL penalty against Stage 1 reference adapter (pi_ref).
  - Saves adapter checkpoints at every epoch (checkpoint-epoch-16 to checkpoint-epoch-50).
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import wandb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.training.rlvr_verifier import compute_verifiable_reward


ANSWER_SEP = "\nAnswer:"


class RLVRQueryDataset(Dataset):
    """
    Dataset supplying prompt-only inputs for RLVR rollouts and target verification.
    """

    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Extract prompt-only text for P_gen
                    p_gen_full = item.get("p_gen", "")
                    sep_pos = p_gen_full.rfind(ANSWER_SEP)
                    if sep_pos != -1:
                        p_gen_prompt = p_gen_full[: sep_pos + len(ANSWER_SEP)]
                    else:
                        p_gen_prompt = p_gen_full

                    self.data.append({
                        "id": item.get("id", ""),
                        "task_type": item.get("task_type", "chaining"),
                        "p_gen_prompt": p_gen_prompt,
                        "target_entity": item.get("target_entity", ""),
                        "bridge_entity": item.get("bridge_entity", ""),
                        "chain_hops": item.get("chain_hops", []),
                        "fc_label": item.get("fc_label", ""),
                    })

        print(f"[RLVRQueryDataset] Loaded {len(self.data)} prompts from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def collate_rlvr_batch(batch: List[dict]) -> dict:
    return {
        "id": [b["id"] for b in batch],
        "task_type": [b["task_type"] for b in batch],
        "p_gen_prompt": [b["p_gen_prompt"] for b in batch],
        "target_entity": [b["target_entity"] for b in batch],
        "bridge_entity": [b["bridge_entity"] for b in batch],
        "chain_hops": [b["chain_hops"] for b in batch],
        "fc_label": [b["fc_label"] for b in batch],
    }


def compute_grad_norm(model: nn.Module) -> float:
    """Compute L2 norm of gradients."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm_sq += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total_norm_sq)


def train_rlvr(args):
    print("=== Starting KUG 2-Stage RLVR Training (GRPO) ===")
    print(f"Base Model: {args.model_name_or_path}")
    print(f"Init Checkpoint (Stage 1): {args.init_checkpoint}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Start Epoch: {args.start_epoch}, End Epoch: {args.end_epoch}")
    print(f"Batch Size: {args.batch_size}, Grad Accum: {args.gradient_accumulation_steps}")
    print(f"Rollouts per prompt (K): {args.num_rollouts}, Temperature: {args.temperature}")
    print(f"KL Penalty Beta: {args.kl_beta}, Clip Epsilon: {args.clip_eps}")
    print(f"LR: {args.learning_rate}")

    out_dir = Path(args.output_dir) / f"{args.method}_qwen2.5-1.5b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=f"train_v2_{args.method}_qwen2.5-1.5b",
        config={
            "method": args.method,
            "base_model": args.model_name_or_path,
            "init_checkpoint": args.init_checkpoint,
            "start_epoch": args.start_epoch,
            "end_epoch": args.end_epoch,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_rollouts": args.num_rollouts,
            "temperature": args.temperature,
            "kl_beta": args.kl_beta,
            "clip_eps": args.clip_eps,
            "learning_rate": args.learning_rate,
        },
    )
    wandb_run_id = wandb_run.id
    print(f"W&B Run Initialized ID: {wandb_run_id}")

    meta_path = out_dir / "run_metadata.json"
    metadata = {
        "wandb_run_id": wandb_run_id,
        "wandb_project": args.wandb_project,
        "method": args.method,
        "base_model": args.model_name_or_path,
        "init_checkpoint": args.init_checkpoint,
        "num_epochs": args.end_epoch,
        "output_dir": str(out_dir.resolve()),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata -> {meta_path}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left-padding required for batched autoregressive generation

    print("Loading base model onto GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    # Load Stage 1 LoRA checkpoint as trainable "default" policy
    print(f"Loading Stage 1 LoRA checkpoint from: {args.init_checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.init_checkpoint, adapter_name="default", is_trainable=True)

    # Also load the identical Stage 1 checkpoint as frozen "reference" policy for KL penalty
    print("Loading frozen reference adapter for KL divergence calculation...")
    model.load_adapter(args.init_checkpoint, adapter_name="reference")
    # Freeze reference adapter parameters
    for name, param in model.named_parameters():
        if "reference" in name:
            param.requires_grad = False

    model.set_adapter("default")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Policy model initialized with {len(trainable_params)} trainable parameter tensors.")

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Dataloader
    dataset = RLVRQueryDataset(args.dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_rlvr_batch,
        num_workers=0,
        pin_memory=True,
    )

    global_step = 0
    K = args.num_rollouts

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        model.train()
        epoch_rewards = []
        epoch_rewards_by_task = {"chaining": [], "intersection": [], "fact_checking": []}
        epoch_policy_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        print(f"\n[RLVR Epoch {epoch:02d}/{args.end_epoch}] Sampling rollouts and computing verifiable rewards...", flush=True)

        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            prompts = batch["p_gen_prompt"]
            targets = batch["target_entity"]
            task_types = batch["task_type"]
            bridge_entities = batch["bridge_entity"]
            chain_hops_list = batch["chain_hops"]
            fc_labels = batch["fc_label"]
            B = len(prompts)

            # Tokenize prompts (left-padded for batch generation)
            tokenizer.padding_side = "left"
            prompt_enc = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_prompt_length, return_tensors="pt")
            prompt_ids = prompt_enc.input_ids.cuda()  # (B, L_prompt)
            prompt_mask = prompt_enc.attention_mask.cuda()

            # Repeat each prompt K times for group rollout generation
            expanded_prompt_ids = prompt_ids.repeat_interleave(K, dim=0)    # (B*K, L_prompt)
            expanded_prompt_mask = prompt_mask.repeat_interleave(K, dim=0)  # (B*K, L_prompt)
            prompt_len = expanded_prompt_ids.shape[1]

            # 1. Generate K rollouts per prompt with trainable policy (sampling with T=0.85)
            model.eval()
            model.set_adapter("default")
            with torch.no_grad():
                generated_seqs = model.generate(
                    input_ids=expanded_prompt_ids,
                    attention_mask=expanded_prompt_mask,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )  # (B*K, L_total)

            model.train()
            # Extract completion tokens only (after prompt_len)
            completion_seqs = generated_seqs[:, prompt_len:]
            total_seq_len = generated_seqs.shape[1]

            # Decode completions to text and compute verifiable rewards with step-wise breadcrumbs
            rewards = []
            for i in range(B * K):
                prompt_idx = i // K
                completion_text = tokenizer.decode(completion_seqs[i], skip_special_tokens=True)
                r = compute_verifiable_reward(
                    completion=completion_text,
                    target_entity=targets[prompt_idx],
                    task_type=task_types[prompt_idx],
                    bridge_entity=bridge_entities[prompt_idx],
                    chain_hops=chain_hops_list[prompt_idx],
                    fc_label=fc_labels[prompt_idx],
                )
                rewards.append(r)
                task = task_types[prompt_idx]
                if task in epoch_rewards_by_task:
                    epoch_rewards_by_task[task].append(r)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device="cuda").view(B, K)
            epoch_rewards.extend(rewards)

            # 2. Compute Group Relative Advantages
            mean_r = rewards_tensor.mean(dim=1, keepdim=True)
            std_r = rewards_tensor.std(dim=1, keepdim=True) + 1e-4
            advantages = (rewards_tensor - mean_r) / std_r  # (B, K)
            flat_advantages = advantages.view(-1)          # (B*K,)

            # Build attention mask and response loss mask for full generated sequences
            gen_attention_mask = (generated_seqs != tokenizer.pad_token_id).long()
            # Response mask: 1 only for non-padding completion tokens
            response_mask = (generated_seqs[:, prompt_len:] != tokenizer.pad_token_id).float()
            num_response_tokens = response_mask.sum().clamp(min=1.0)

            # 3. Forward pass under Policy Model (default adapter) -> completion token log probs
            model.set_adapter("default")
            policy_outputs = model(input_ids=generated_seqs, attention_mask=gen_attention_mask)
            # Memory optimization: Slice logits ONLY for completion tokens (length <= 32) instead of entire prompt
            policy_comp_logits = policy_outputs.logits[:, prompt_len - 1 : -1, :]
            comp_targets = target_ids[:, prompt_len - 1 :]
            policy_comp_log_probs = F.log_softmax(policy_comp_logits, dim=-1).gather(
                dim=-1, index=comp_targets.unsqueeze(-1)
            ).squeeze(-1)
            del policy_outputs, policy_comp_logits

            # 4. Forward pass under Reference Model (reference adapter) -> ref completion token log probs
            model.set_adapter("reference")
            with torch.no_grad():
                ref_outputs = model(input_ids=generated_seqs, attention_mask=gen_attention_mask)
                ref_comp_logits = ref_outputs.logits[:, prompt_len - 1 : -1, :]
                ref_comp_log_probs = F.log_softmax(ref_comp_logits, dim=-1).gather(
                    dim=-1, index=comp_targets.unsqueeze(-1)
                ).squeeze(-1)
                del ref_outputs, ref_comp_logits

            # Switch back to trainable policy adapter
            model.set_adapter("default")

            # 5. Compute Policy Gradient Loss with Advantage Weighting
            # Token importance ratio (at generation time, pi_old is detached policy_comp_log_probs)
            log_ratio = policy_comp_log_probs - policy_comp_log_probs.detach()
            ratio = torch.exp(log_ratio)

            adv_expanded = flat_advantages.unsqueeze(1)  # (B*K, 1)
            surr1 = ratio * adv_expanded
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_expanded
            policy_loss_per_token = -torch.min(surr1, surr2) * response_mask
            policy_loss = policy_loss_per_token.sum() / num_response_tokens

            # 6. Compute Token-Level KL Penalty against Stage 1 Reference
            # KL(pi || ref) approx = exp(log_ref - log_pi) - (log_ref - log_pi) - 1
            log_diff = ref_comp_log_probs - policy_comp_log_probs
            kl_per_token = (torch.exp(log_diff) - log_diff - 1.0) * response_mask
            kl_loss = kl_per_token.sum() / num_response_tokens

            total_loss = policy_loss + args.kl_beta * kl_loss
            scaled_loss = total_loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            epoch_policy_loss += policy_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = compute_grad_norm(model)
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                step_reward = sum(rewards) / len(rewards)
                wandb.log({
                    "step": global_step,
                    "train/rlvr_step_reward": step_reward,
                    "train/policy_loss": policy_loss.item(),
                    "train/kl_loss": kl_loss.item(),
                    "train/total_loss": total_loss.item(),
                    "train/grad_norm": grad_norm,
                })

        avg_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))
        avg_policy = epoch_policy_loss / max(1, num_batches)
        avg_kl = epoch_kl_loss / max(1, num_batches)
        avg_total = epoch_total_loss / max(1, num_batches)

        chaining_r = sum(epoch_rewards_by_task["chaining"]) / max(1, len(epoch_rewards_by_task["chaining"]))
        inter_r = sum(epoch_rewards_by_task["intersection"]) / max(1, len(epoch_rewards_by_task["intersection"]))
        fc_r = sum(epoch_rewards_by_task["fact_checking"]) / max(1, len(epoch_rewards_by_task["fact_checking"]))

        print(
            f"Epoch {epoch:02d}/{args.end_epoch:02d} | "
            f"Reward: {avg_reward:.3%} | "
            f"Chaining: {chaining_r:.3%} | Inter: {inter_r:.3%} | FC: {fc_r:.3%} | "
            f"Loss: {avg_total:.4f} (Pol: {avg_policy:.4f}, KL: {avg_kl:.4f})",
            flush=True,
        )

        wandb.log({
            "epoch": epoch,
            "train/epoch_rlvr_reward": avg_reward,
            "train/epoch_reward_chaining": chaining_r,
            "train/epoch_reward_intersection": inter_r,
            "train/epoch_reward_fact_checking": fc_r,
            "train/epoch_policy_loss": avg_policy,
            "train/epoch_kl_loss": avg_kl,
            "train/epoch_total_loss": avg_total,
        })

        # Save per-epoch adapter weights
        ckpt_dir = out_dir / f"checkpoint-epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir, selected_adapters=["default"])
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Saved RLVR adapter checkpoint -> {ckpt_dir}")

    wandb.finish()
    print("=== RLVR Training Completed Successfully ===")


def main():
    parser = argparse.ArgumentParser(description="Train KUG 2-Stage RLVR with GRPO and Verifiable Rewards")
    parser.add_argument("--method", type=str, default="two_stage_breadcrumb_rlvr")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--init_checkpoint", type=str, default="outputs/kug_overhaul_v2/baseline_qwen2.5-1.5b/checkpoint-epoch-15")
    parser.add_argument("--dataset_path", type=str, default="data/processed/kug_dataset_all.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/kug_overhaul_v2")
    parser.add_argument("--wandb_project", type=str, default="kug_overhaul_qwen1.5b")
    parser.add_argument("--start_epoch", type=int, default=16)
    parser.add_argument("--end_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--kl_beta", type=float, default=0.04)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    train_rlvr(args)


if __name__ == "__main__":
    main()
