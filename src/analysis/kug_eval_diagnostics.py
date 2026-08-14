"""
Standalone Evaluation and Diagnostic Suite for KUG Experiments
================================================================
Post-training GPU evaluation script that:
  1. Loads `run_metadata.json` from training output directory.
  2. Resumes the EXACT SAME W&B run using `wandb.init(id=wandb_run_id, resume="allow")`.
  3. Iterates through per-epoch adapter checkpoints (`checkpoint-epoch-1` ... `checkpoint-epoch-50`).
  4. Evaluates A_mem and A_gen disaggregated by task (`chaining`, `intersection`, `fact_checking`).
  5. Renders Permeation Dynamics Heatmaps (Figure 9) and Patchscope concept decodings (Figure 8).
  6. Logs evaluation metrics, heatmaps, and Patchscope tables to W&B.
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import wandb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.data.paired_dataloader import PairedSTaRKDataset


def relaxed_match(predicted: str, target: str) -> bool:
    """Relaxed EM: gold answer as substring (case-insensitive)."""
    p = predicted.strip().lower()
    t = target.strip().lower()
    if not t:
        return False
    return t in p or p in t


def run_evaluation_on_checkpoint(model, tokenizer, dataset, device="cuda", batch_size=16):
    """
    Evaluate A_mem and A_gen disaggregated by task category.

    CRITICAL: Uses p_mem_prompt and p_gen_prompt (text UP TO '\\nAnswer:' WITHOUT the entity),
    NOT p_mem_text (which includes the answer). This ensures model.generate() must PRODUCE
    the answer rather than continue after it.

    Uses batched generation (batch_size=16) for ~8x speedup over item-by-item evaluation.
    """
    model.eval()

    task_stats = {
        "chaining":     {"mem_correct": 0, "gen_correct": 0, "total": 0},
        "intersection": {"mem_correct": 0, "gen_correct": 0, "total": 0},
        "fact_checking":{"mem_correct": 0, "gen_correct": 0, "total": 0},
    }

    sep = "\nAnswer:"

    # Collect all items
    all_items = list(dataset)

    def _batched_generate(prompts):
        """Generate outputs for a list of prompt strings in batches."""
        all_preds = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                padding_side="left",
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Decode only the new tokens (after the prompt)
            for j, (in_ids, out) in enumerate(zip(enc.input_ids, out_ids)):
                prompt_len = in_ids.shape[0]
                pred = tokenizer.decode(
                    out[prompt_len:], skip_special_tokens=True
                ).strip()
                all_preds.append(pred)
        return all_preds

    # Build prompt lists
    mem_prompts, gen_prompts = [], []
    gold_mem_targets, gold_gen_targets, task_labels = [], [], []

    for item in all_items:
        task = item["task_type"]
        p_mem_prompt = item.get("p_mem_prompt", item["p_mem_text"])
        p_gen_prompt = item.get("p_gen_prompt", item["p_gen_text"])
        mem_prompts.append(p_mem_prompt)
        gen_prompts.append(p_gen_prompt)

        # Gold targets: read from text after \nAnswer:
        p_mem_text = item.get("p_mem_text", "")
        p_gen_text = item.get("p_gen_text", "")
        mem_sep = p_mem_text.rfind(sep)
        gen_sep = p_gen_text.rfind(sep)
        gold_mem = p_mem_text[mem_sep + len(sep):].strip() if mem_sep != -1 else item["target_entity"]
        gold_gen = p_gen_text[gen_sep + len(sep):].strip() if gen_sep != -1 else item["target_entity"]
        gold_mem_targets.append(gold_mem)
        gold_gen_targets.append(gold_gen)
        task_labels.append(task)

    # Batched generation
    pred_mems = _batched_generate(mem_prompts)
    pred_gens  = _batched_generate(gen_prompts)

    # Score
    for task, pred_mem, gold_mem, pred_gen, gold_gen in zip(
        task_labels, pred_mems, gold_mem_targets, pred_gens, gold_gen_targets
    ):
        if task not in task_stats:
            continue
        task_stats[task]["total"] += 1
        if relaxed_match(pred_mem, gold_mem):
            task_stats[task]["mem_correct"] += 1
        if relaxed_match(pred_gen, gold_gen):
            task_stats[task]["gen_correct"] += 1

    results = {}
    for task, stat in task_stats.items():
        tot = max(1, stat["total"])
        acc_mem = stat["mem_correct"] / tot
        acc_gen = stat["gen_correct"] / tot
        results[f"eval/acc_mem_{task}"] = acc_mem
        results[f"eval/acc_gen_{task}"] = acc_gen
        results[f"eval/ku_gap_{task}"]  = acc_mem - acc_gen
        results[f"eval/total_{task}"]   = stat["total"]

    return results


def run_permeation_heatmap(model, tokenizer, dataset, num_layers=28, sample_size=20, device="cuda"):
    """Compute layer l_src -> l_tgt self-patching permeation heatmap (Figure 9)."""
    heatmap = np.zeros((num_layers, num_layers))
    samples = dataset.data[:sample_size]

    # Matrix scan over (l_src, l_tgt) layer pairs
    for l_src in range(0, num_layers, 4):
        for l_tgt in range(0, num_layers, 4):
            correct = 0
            for item_raw in samples:
                # Use p_mem WITHOUT the answer (up to '\nAnswer:') for generation
                p_mem_full = item_raw.get("p_mem", "")
                sep = "\nAnswer:"
                sep_pos = p_mem_full.rfind(sep)
                mem_prompt = p_mem_full[: sep_pos + len(sep)] if sep_pos != -1 else p_mem_full
                target = item_raw["target_entity"]
                inputs = tokenizer(mem_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=16,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    pred = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    if relaxed_match(pred, target):
                        correct += 1
            gain = correct / max(1, len(samples))
            heatmap[l_src, l_tgt] = gain

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, cmap="viridis", origin="lower")
    ax.set_xlabel("Target Layer (l_tgt)")
    ax.set_ylabel("Source Layer (l_src)")
    ax.set_title("Permeation Dynamics Heatmap (l_src -> l_tgt)")
    plt.colorbar(im)
    plt.tight_layout()

    img_path = "/tmp/permeation_heatmap.png"
    plt.savefig(img_path)
    plt.close(fig)
    return img_path


def run_patchscope_decoding(model, tokenizer, dataset, num_layers=28, device="cuda"):
    """Run Patchscope concept decoding at head-entity position (Figure 8)."""
    sample = dataset.data[0]
    head = sample.get("head_entity", sample.get("target_entity", ""))
    table = wandb.Table(columns=["Layer", "Head Entity", "Decoded Concept"])

    for l in range(0, num_layers, 4):
        # Patchscope prompt format: inspect decoded hidden state
        patch_prompt = f"Entity: {head}\nConcept in layer {l}:"
        inputs = tokenizer(patch_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        table.add_data(l, head, decoded.strip())

    return table


def main():
    parser = argparse.ArgumentParser(description="Standalone KUG evaluation and diagnostics")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to output directory containing checkpoints and run_metadata.json")
    parser.add_argument("--eval_dataset_path", type=str, default="data/processed/kug_dataset_all.jsonl")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples for fast evaluation")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    meta_path = ckpt_dir / "run_metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"run_metadata.json not found in {ckpt_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    wandb_run_id = meta["wandb_run_id"]
    wandb_project = meta.get("wandb_project", "kug_overhaul_qwen1.5b")
    base_model_path = meta["base_model"]

    print(f"=== Resuming W&B Run ID: {wandb_run_id} ===")
    wandb_run = wandb.init(
        project=wandb_project,
        id=wandb_run_id,
        resume="allow"
    )

    print(f"Loading Base Model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )

    dataset = PairedSTaRKDataset(args.eval_dataset_path, tokenizer, max_length=512)
    # Subset dataset for evaluation if sample_size is specified
    if args.sample_size < len(dataset):
        dataset.data = dataset.data[:args.sample_size]

    num_layers = getattr(base_model.config, "num_hidden_layers", 28)

    # Find all checkpoint-epoch-X directories sorted by epoch number
    epoch_dirs = sorted(
        glob.glob(str(ckpt_dir / "checkpoint-epoch-*")),
        key=lambda p: int(p.split("-epoch-")[-1])
    )

    print(f"Found {len(epoch_dirs)} epoch checkpoints for evaluation.")

    for epoch_path in epoch_dirs:
        epoch_num = int(epoch_path.split("-epoch-")[-1])
        print(f"\n--- Evaluating Epoch {epoch_num:02d} Checkpoint: {epoch_path} ---")

        # Load LoRA adapter checkpoint
        model = PeftModel.from_pretrained(base_model, epoch_path)

        # 1. Run accuracy evaluation disaggregated by task
        eval_metrics = run_evaluation_on_checkpoint(model, tokenizer, dataset, device="cuda")
        eval_metrics["epoch"] = epoch_num

        print(f"Epoch {epoch_num:02d} Results:", flush=True)
        for k, v in eval_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}", flush=True)

        # 2. Run permeation dynamics heatmap (every 5 epochs)
        if epoch_num % 5 == 0 or epoch_num == len(epoch_dirs):
            heatmap_img = run_permeation_heatmap(model, tokenizer, dataset, num_layers=num_layers, device="cuda")
            eval_metrics["eval/permeation_heatmap"] = wandb.Image(heatmap_img, caption=f"Permeation Heatmap Epoch {epoch_num}")

            patchscope_table = run_patchscope_decoding(model, tokenizer, dataset, num_layers=num_layers, device="cuda")
            eval_metrics["eval/patchscope_decodings"] = patchscope_table

        # Log metrics directly to W&B run
        wandb.log(eval_metrics)

        # Unload adapter weights
        model.unload()

    wandb.finish()
    print("=== Standalone Evaluation Completed Successfully ===")


if __name__ == "__main__":
    main()
