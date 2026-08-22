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
import numpy as np
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
from src.models.entity_span_extractor import find_token_span_for_substring, extract_entity_token_mask
from src.losses.kcr_loss import KnowledgeCircuitRoutingLoss


ANSWER_SEP = "\nAnswer:"


class RLVRQueryDataset(Dataset):
    """
    Dataset supplying prompt-only inputs for RLVR rollouts, paired memory prompts for OPRD, and target verification.
    """

    def __init__(self, jsonl_path: str, use_cot: bool = False, use_thinking: bool = False):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    task_type = item.get("task_type", "chaining")
                    p_gen_full = item.get("p_gen", "")
                    p_mem_full = item.get("p_mem", "")
                    sep_pos = p_gen_full.rfind(ANSWER_SEP)
                    sep_mem_pos = p_mem_full.rfind(ANSWER_SEP)

                    query = p_gen_full[: sep_pos].strip() if sep_pos != -1 else p_gen_full.strip()
                    query_mem = p_mem_full[: sep_mem_pos].strip() if sep_mem_pos != -1 else p_mem_full.strip()

                    if use_thinking:
                        p_gen_prompt = f"{query}\n<think>\n"
                        p_mem_prompt = f"{query_mem}\n<think>\n"
                    elif use_cot and task_type == "chaining":
                        p_gen_prompt = f"{query}\nThought: "
                        p_mem_prompt = f"{query_mem}\nAnswer:"
                    else:
                        if sep_pos != -1:
                            p_gen_prompt = p_gen_full[: sep_pos + len(ANSWER_SEP)]
                        else:
                            p_gen_prompt = p_gen_full

                        if sep_mem_pos != -1:
                            p_mem_prompt = p_mem_full[: sep_mem_pos + len(ANSWER_SEP)]
                        else:
                            p_mem_prompt = p_mem_full

                    self.data.append({
                        "id": item.get("id", ""),
                        "task_type": task_type,
                        "p_gen_prompt": p_gen_prompt,
                        "p_mem_prompt": p_mem_prompt,
                        "head_entity": item.get("head_entity", ""),
                        "target_entity": item.get("target_entity", ""),
                        "bridge_entity": item.get("bridge_entity", ""),
                        "chain_hops": item.get("chain_hops", []),
                        "fc_label": item.get("fc_label", ""),
                    })

        print(f"[RLVRQueryDataset] Loaded {len(self.data)} prompts from {jsonl_path} (use_cot={use_cot}, use_thinking={use_thinking})")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def collate_rlvr_batch(batch: List[dict]) -> dict:
    return {
        "id": [b["id"] for b in batch],
        "task_type": [b["task_type"] for b in batch],
        "p_gen_prompt": [b["p_gen_prompt"] for b in batch],
        "p_mem_prompt": [b["p_mem_prompt"] for b in batch],
        "head_entity": [b["head_entity"] for b in batch],
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
    print("=" * 70)
    print("  KUG 2-Stage RLVR Training (GRPO + OPRD + Curriculum Annealing)")
    print(f"  Base Model:       {args.model_name_or_path}")
    print(f"  Init Checkpoint:  {args.init_checkpoint}")
    print(f"  Dataset:          {args.dataset_path}")
    print(f"  Epochs:           {args.start_epoch} -> {args.end_epoch}")
    print(f"  Rollouts (K):     {args.num_rollouts}")
    print(f"  Batch Size:       {args.batch_size} (Grad Accum: {args.gradient_accumulation_steps})")
    print(f"  KL Beta:          {args.kl_beta}")
    print(f"  OPRD Weight:      {args.oprd_weight}")
    print(f"  Curriculum Anneal:{args.curriculum_anneal}")
    print(f"  Use CoT:          {args.use_cot}")
    print("=" * 70)

    out_dir = Path(args.output_dir) / f"{args.method}_{Path(args.model_name_or_path).name.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=f"train_v2_{args.method}_{Path(args.model_name_or_path).name.lower()}",
        config=vars(args),
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
    tokenizer.padding_side = "left"

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

    # Also load the identical Stage 1 checkpoint as frozen "reference" policy for KL penalty and OPRD
    print("Loading frozen reference adapter for KL divergence and OPRD calculation...")
    model.load_adapter(args.init_checkpoint, adapter_name="reference")
    # Freeze reference adapter parameters
    for name, param in model.named_parameters():
        if "reference" in name:
            param.requires_grad = False

    model.set_adapter("default")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled on policy model.")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Policy model initialized with {len(trainable_params)} trainable parameter tensors.")

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Dataloader
    dataset = RLVRQueryDataset(args.dataset_path, use_cot=args.use_cot, use_thinking=args.use_thinking)
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

    kcr_loss_fn = KnowledgeCircuitRoutingLoss(target_layer_ratio=0.50, source_layer_ratios=(0.10, 0.80)).to("cuda")

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        model.train()
        epoch_rewards = []
        epoch_rewards_by_task = {"chaining": [], "intersection": [], "fact_checking": []}
        epoch_policy_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_kcr_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        print(f"\n[RLVR Epoch {epoch:02d}/{args.end_epoch}] Sampling rollouts and computing verifiable rewards (Curriculum Phase)...", flush=True)

        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            prompts = batch["p_gen_prompt"]
            mem_prompts = batch["p_mem_prompt"]
            targets = batch["target_entity"]
            task_types = batch["task_type"]
            bridge_entities = batch["bridge_entity"]
            chain_hops_list = batch["chain_hops"]
            fc_labels = batch["fc_label"]
            B = len(prompts)

            # Tokenize prompts (left-padded for batch generation)
            tokenizer.padding_side = "left"
            prompt_enc = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_prompt_length, return_tensors="pt")
            prompt_ids = prompt_enc.input_ids.cuda()
            prompt_mask = prompt_enc.attention_mask.cuda()

            # Repeat each prompt K times for group rollout generation
            expanded_prompt_ids = prompt_ids.repeat_interleave(K, dim=0)
            expanded_prompt_mask = prompt_mask.repeat_interleave(K, dim=0)
            prompt_len = expanded_prompt_ids.shape[1]

            # 1. Generate K rollouts per prompt with trainable policy
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
                )

            model.train()
            completion_seqs = generated_seqs[:, prompt_len:]

            # Decode completions to text and compute verifiable rewards with Curriculum Annealing
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
                    epoch=epoch,
                    curriculum_anneal=args.curriculum_anneal,
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
            advantages = (rewards_tensor - mean_r) / std_r
            flat_advantages = advantages.view(-1)

            gen_attention_mask = (generated_seqs != tokenizer.pad_token_id).long()
            response_mask = (generated_seqs[:, prompt_len:] != tokenizer.pad_token_id).float()
            num_response_tokens = response_mask.sum().clamp(min=1.0)

            # 3. Forward pass under Policy Model (default adapter) with micro-batching
            target_ids = generated_seqs[:, 1:]
            comp_targets = target_ids[:, prompt_len - 1 :]
            model.set_adapter("default")

            micro_bs = 4
            policy_comp_log_probs_list = []
            prompt_hs_list = []

            for mb_start in range(0, B * K, micro_bs):
                mb_end = min(mb_start + micro_bs, B * K)
                mb_seqs = generated_seqs[mb_start:mb_end]
                mb_mask = gen_attention_mask[mb_start:mb_end]
                mb_targets = comp_targets[mb_start:mb_end]

                need_hs = (args.kcr_weight > 0.0 or args.oprd_weight > 0.0)
                mb_out = model(input_ids=mb_seqs, attention_mask=mb_mask, output_hidden_states=need_hs)
                mb_logits = mb_out.logits[:, prompt_len - 1 : -1, :]
                mb_lp = F.log_softmax(mb_logits, dim=-1).gather(dim=-1, index=mb_targets.unsqueeze(-1)).squeeze(-1)
                policy_comp_log_probs_list.append(mb_lp)

                if need_hs:
                    if not prompt_hs_list:
                        prompt_hs_list = [[] for _ in range(len(mb_out.hidden_states))]
                    for l_idx, hs in enumerate(mb_out.hidden_states):
                        prompt_hs_list[l_idx].append(hs[:, :prompt_len, :])
                del mb_out, mb_logits, mb_lp

            policy_comp_log_probs = torch.cat(policy_comp_log_probs_list, dim=0)

            if args.kcr_weight > 0.0 or args.oprd_weight > 0.0:
                all_prompt_hs = tuple(torch.cat(prompt_hs_list[l_idx], dim=0) for l_idx in range(len(prompt_hs_list)))
                policy_prompt_hidden_states = tuple(hs[::K] for hs in all_prompt_hs)
            else:
                policy_prompt_hidden_states = ()

            # 4. Forward pass under Reference Model (reference adapter) with micro-batching
            model.set_adapter("reference")
            ref_comp_log_probs_list = []
            with torch.no_grad():
                for mb_start in range(0, B * K, micro_bs):
                    mb_end = min(mb_start + micro_bs, B * K)
                    mb_seqs = generated_seqs[mb_start:mb_end]
                    mb_mask = gen_attention_mask[mb_start:mb_end]
                    mb_targets = comp_targets[mb_start:mb_end]

                    mb_out = model(input_ids=mb_seqs, attention_mask=mb_mask)
                    mb_logits = mb_out.logits[:, prompt_len - 1 : -1, :]
                    mb_lp = F.log_softmax(mb_logits, dim=-1).gather(dim=-1, index=mb_targets.unsqueeze(-1)).squeeze(-1)
                    ref_comp_log_probs_list.append(mb_lp)
                    del mb_out, mb_logits, mb_lp

            ref_comp_log_probs = torch.cat(ref_comp_log_probs_list, dim=0)

            # 5. Knowledge-Circuit Routing (KCR): Causal Head-Entity Layer-Pair Alignment
            if args.kcr_weight > 0.0:
                tokenizer.padding_side = "right"
                mem_enc = tokenizer(
                    mem_prompts, padding=True, truncation=True, max_length=args.max_prompt_length, return_tensors="pt"
                ).to("cuda")
                with torch.no_grad():
                    ref_mem_outputs = model(input_ids=mem_enc.input_ids, attention_mask=mem_enc.attention_mask, output_hidden_states=True)

                # Build token-span masks for head entity in P_gen and P_mem
                gen_masks_list = []
                mem_masks_list = []
                for b_idx in range(B):
                    head_ent = batch["head_entity"][b_idx]
                    p_gen_txt = prompts[b_idx]
                    p_mem_txt = mem_prompts[b_idx]

                    unpadded_gen_len = int(prompt_mask[b_idx].sum().item())
                    gen_pad_offset = prompt_len - unpadded_gen_len

                    gen_toks = tokenizer(p_gen_txt, return_offsets_mapping=True)
                    mem_toks = tokenizer(p_mem_txt, return_offsets_mapping=True)

                    span_gen = find_token_span_for_substring(p_gen_txt, head_ent, gen_toks.get("offset_mapping", []))
                    span_mem = find_token_span_for_substring(p_mem_txt, head_ent, mem_toks.get("offset_mapping", []))

                    gen_masks_list.append(extract_entity_token_mask(prompt_len, span_gen, pad_offset=gen_pad_offset))
                    mem_masks_list.append(extract_entity_token_mask(mem_enc.input_ids.shape[1], span_mem, pad_offset=0))

                gen_entity_masks = torch.tensor(gen_masks_list, device="cuda", dtype=torch.float32)
                mem_entity_masks = torch.tensor(mem_masks_list, device="cuda", dtype=torch.float32)

                kcr_loss = kcr_loss_fn(
                    policy_prompt_hidden_states,
                    ref_mem_outputs.hidden_states,
                    gen_entity_masks,
                    mem_entity_masks,
                )
                del ref_mem_outputs
            else:
                kcr_loss = torch.tensor(0.0, device="cuda")

            del policy_outputs

            # Switch back to trainable policy adapter
            model.set_adapter("default")

            # 6. Compute Policy Gradient Loss with Advantage Weighting
            log_ratio = policy_comp_log_probs - policy_comp_log_probs.detach()
            ratio = torch.exp(log_ratio)

            adv_expanded = flat_advantages.unsqueeze(1)
            surr1 = ratio * adv_expanded
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_expanded
            policy_loss_per_token = -torch.min(surr1, surr2) * response_mask
            policy_loss = policy_loss_per_token.sum() / num_response_tokens

            # 7. Compute Token-Level KL Penalty against Stage 1 Reference
            log_diff = ref_comp_log_probs - policy_comp_log_probs
            kl_per_token = (torch.exp(log_diff) - log_diff - 1.0) * response_mask
            kl_loss = kl_per_token.sum() / num_response_tokens

            total_loss = policy_loss + args.kl_beta * kl_loss + args.kcr_weight * kcr_loss
            scaled_loss = total_loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            epoch_policy_loss += policy_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_kcr_loss += kcr_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = compute_grad_norm(model)
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                step_reward = np.mean(rewards)
                if global_step % 1 == 0:
                    wandb.log({
                        "train/total_loss": total_loss.item(),
                        "train/policy_loss": policy_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/kcr_loss": kcr_loss.item(),
                        "train/rlvr_step_reward": step_reward,
                        "train/grad_norm": grad_norm,
                        "step": global_step,
                    })

        avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        avg_policy_loss = epoch_policy_loss / max(1, num_batches)
        avg_kl_loss = epoch_kl_loss / max(1, num_batches)
        avg_kcr_loss = epoch_kcr_loss / max(1, num_batches)
        avg_total_loss = epoch_total_loss / max(1, num_batches)

        reward_ch = np.mean(epoch_rewards_by_task["chaining"]) if epoch_rewards_by_task["chaining"] else 0.0
        reward_in = np.mean(epoch_rewards_by_task["intersection"]) if epoch_rewards_by_task["intersection"] else 0.0
        reward_fc = np.mean(epoch_rewards_by_task["fact_checking"]) if epoch_rewards_by_task["fact_checking"] else 0.0

        print(
            f"Epoch {epoch:02d}/{args.end_epoch} | Reward: {avg_epoch_reward*100:.3f}% | "
            f"Chaining: {reward_ch*100:.3f}% | Inter: {reward_in*100:.3f}% | FC: {reward_fc*100:.3f}% | "
            f"Loss: {avg_total_loss:.4f} (Pol: {avg_policy_loss:.4f}, KL: {avg_kl_loss:.4f}, KCR: {avg_kcr_loss:.4f})",
            flush=True
        )

        wandb.log({
            "epoch": epoch,
            "train/epoch_rlvr_reward": avg_epoch_reward,
            "train/epoch_reward_chaining": reward_ch,
            "train/epoch_reward_intersection": reward_in,
            "train/epoch_reward_fact_checking": reward_fc,
            "train/epoch_total_loss": avg_total_loss,
            "train/epoch_policy_loss": avg_policy_loss,
            "train/epoch_kl_loss": avg_kl_loss,
            "train/epoch_kcr_loss": avg_kcr_loss,
        })

        # Save per-epoch checkpoint
        ckpt_dir = out_dir / f"checkpoint-epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir), selected_adapters=["default"])
        tokenizer.save_pretrained(str(ckpt_dir))
        print(f"Saved RLVR adapter checkpoint -> {ckpt_dir}")

    wandb.finish()
    print("=== RLVR Training Completed Successfully ===")


def main():
    parser = argparse.ArgumentParser(description="Train KUG 2-Stage RLVR with GRPO, KCR, and Curriculum Annealing")
    parser.add_argument("--method", type=str, default="two_stage_kcr_curriculum_rlvr")
    parser.add_argument("--use_cot", action="store_true", default=True)
    parser.add_argument("--curriculum_anneal", action="store_true", default=True)
    parser.add_argument("--kcr_weight", type=float, default=0.15)
    parser.add_argument("--oprd_weight", type=float, default=0.0)
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
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--kl_beta", type=float, default=0.04)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--use_thinking", action="store_true", default=False, help="Use structured thinking traces (<think>...</think>) for reasoning models")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    train_rlvr(args)


if __name__ == "__main__":
    main()

