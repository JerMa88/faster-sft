import os
import glob
import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import wandb

from src.data.paired_dataloader import PairedSTaRKDataset
from src.analysis.kug_eval_diagnostics import run_evaluation_on_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Fast Accuracy Evaluator for KUG Checkpoints")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--eval_dataset_path", type=str, default="data/processed/kug_dataset_all.jsonl")
    parser.add_argument("--sample_size", type=int, default=200)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    meta_path = ckpt_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_metadata.json not found in {ckpt_dir}")

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    wandb_run_id = meta["wandb_run_id"]
    wandb_project = meta.get("wandb_project", "kug_overhaul_qwen1.5b")
    base_model_path = meta["base_model"]

    print(f"=== Fast Accuracy Evaluation for {ckpt_dir.name} ===")
    print(f"Resuming W&B Run ID: {wandb_run_id}")
    wandb.init(project=wandb_project, id=wandb_run_id, resume="allow")

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
    if args.sample_size < len(dataset):
        # Stratified sampling: ensure all 3 task types are represented proportionally.
        # Without this, the first N samples are all chaining (dataset is ordered by task),
        # giving 0 samples to intersection and fact_checking → artificially 0% accuracy.
        import random as _random
        _random.seed(42)
        by_task = {"chaining": [], "intersection": [], "fact_checking": []}
        for item in dataset.data:
            t = item.get("task_type", "chaining")
            if t in by_task:
                by_task[t].append(item)
        # Allocate sample_size proportionally but with a floor of min(20, available)
        total_available = sum(len(v) for v in by_task.values())
        stratified = []
        for task, items in by_task.items():
            n = max(min(20, len(items)),
                    int(args.sample_size * len(items) / max(1, total_available)))
            _random.shuffle(items)
            stratified.extend(items[:n])
        # Trim to exactly sample_size if over
        _random.shuffle(stratified)
        dataset.data = stratified[:args.sample_size]
        task_dist = {t: sum(1 for d in dataset.data if d.get("task_type") == t)
                     for t in by_task}
        print(f"Stratified eval sample: {task_dist} (total={len(dataset.data)})", flush=True)

    epoch_dirs = sorted(
        glob.glob(str(ckpt_dir / "checkpoint-epoch-*")),
        key=lambda p: int(p.split("-epoch-")[-1])
    )

    print(f"Found {len(epoch_dirs)} epoch checkpoints for fast accuracy evaluation.")

    for epoch_path in epoch_dirs:
        epoch_num = int(epoch_path.split("-epoch-")[-1])
        print(f"Evaluating Epoch {epoch_num:02d}...", flush=True)

        model = PeftModel.from_pretrained(base_model, epoch_path)
        eval_metrics = run_evaluation_on_checkpoint(model, tokenizer, dataset, device="cuda", batch_size=128)
        eval_metrics["epoch"] = epoch_num

        for k, v in eval_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}", flush=True)

        wandb.log(eval_metrics)
        model.unload()

    wandb.finish()
    print("=== Fast Accuracy Evaluation Completed Successfully ===")

if __name__ == "__main__":
    main()
