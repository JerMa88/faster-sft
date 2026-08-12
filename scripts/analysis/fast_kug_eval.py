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
        dataset.data = dataset.data[:args.sample_size]

    epoch_dirs = sorted(
        glob.glob(str(ckpt_dir / "checkpoint-epoch-*")),
        key=lambda p: int(p.split("-epoch-")[-1])
    )

    print(f"Found {len(epoch_dirs)} epoch checkpoints for fast accuracy evaluation.")

    for epoch_path in epoch_dirs:
        epoch_num = int(epoch_path.split("-epoch-")[-1])
        print(f"Evaluating Epoch {epoch_num:02d}...", flush=True)

        model = PeftModel.from_pretrained(base_model, epoch_path)
        eval_metrics = run_evaluation_on_checkpoint(model, tokenizer, dataset, device="cuda")
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
