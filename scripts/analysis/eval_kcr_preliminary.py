"""
Preliminary Evaluation Script for Knowledge-Circuit Routing (KCR) Checkpoints
Evaluates all existing epoch checkpoints (Stage 1 Epochs 1-15 + Stage 2 KCR Epochs 16+)
on a spare GPU on bcm-dgxa100-0005.
"""

import os
import sys
import glob
import json
import torch
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import wandb

from src.data.paired_dataloader import PairedSTaRKDataset
from src.analysis.kug_eval_diagnostics import run_evaluation_on_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Preliminary Evaluation for KCR Checkpoints")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/kug_overhaul_v2/two_stage_kcr_curriculum_rlvr_qwen2.5-1.5b")
    parser.add_argument("--eval_dataset_path", type=str, default="data/processed/kug_dataset_all.jsonl")
    parser.add_argument("--sample_size", type=int, default=399)
    parser.add_argument("--batch_size", type=int, default=64)
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

    print(f"================================================================")
    print(f"  KCR Preliminary Evaluation on bcm-dgxa100-0005 (GPU)")
    print(f"  Checkpoint Dir: {ckpt_dir}")
    print(f"  W&B Run ID:     {wandb_run_id}")
    print(f"  Sample Size:    {args.sample_size}")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"================================================================", flush=True)

    try:
        wandb.init(project=wandb_project, id=wandb_run_id, resume="allow")
    except Exception as e:
        print(f"Warning: wandb init failed ({e}), continuing without wandb resume.")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model into GPU memory...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )

    dataset = PairedSTaRKDataset(args.eval_dataset_path, tokenizer, max_length=512)
    if args.sample_size < len(dataset):
        import random as _random
        _random.seed(42)
        by_task = {"chaining": [], "intersection": [], "fact_checking": []}
        for item in dataset.data:
            t = item.get("task_type", "chaining")
            if t in by_task:
                by_task[t].append(item)
        total_available = sum(len(v) for v in by_task.values())
        stratified = []
        for task, items in by_task.items():
            n = max(min(20, len(items)),
                    int(args.sample_size * len(items) / max(1, total_available)))
            _random.shuffle(items)
            stratified.extend(items[:n])
        _random.shuffle(stratified)
        dataset.data = stratified[:args.sample_size]
        task_dist = {t: sum(1 for d in dataset.data if d.get("task_type") == t)
                     for t in by_task}
        print(f"Stratified eval sample distribution: {task_dist} (total={len(dataset.data)})", flush=True)

    epoch_dirs = sorted(
        glob.glob(str(ckpt_dir / "checkpoint-epoch-*")),
        key=lambda p: int(p.split("-epoch-")[-1])
    )

    print(f"Found {len(epoch_dirs)} epoch checkpoints to evaluate.")

    results = []

    for epoch_path in epoch_dirs:
        epoch_num = int(epoch_path.split("-epoch-")[-1])
        print(f"\nEvaluating Checkpoint Epoch {epoch_num:02d}...", flush=True)

        model = PeftModel.from_pretrained(base_model, epoch_path)
        eval_metrics = run_evaluation_on_checkpoint(model, tokenizer, dataset, device="cuda", batch_size=args.batch_size)
        eval_metrics["epoch"] = epoch_num

        results.append(eval_metrics)

        ch_mem = eval_metrics.get("eval/acc_mem_chaining", eval_metrics.get("accuracy/chaining_mem", 0)) * 100
        ch_gen = eval_metrics.get("eval/acc_gen_chaining", eval_metrics.get("accuracy/chaining_gen", 0)) * 100
        ch_gap = eval_metrics.get("eval/ku_gap_chaining", eval_metrics.get("kug_gap/chaining", 0)) * 100
        in_mem = eval_metrics.get("eval/acc_mem_intersection", eval_metrics.get("accuracy/intersection_mem", 0)) * 100
        in_gen = eval_metrics.get("eval/acc_gen_intersection", eval_metrics.get("accuracy/intersection_gen", 0)) * 100
        fc_mem = eval_metrics.get("eval/acc_mem_fact_checking", eval_metrics.get("accuracy/fact_checking_mem", 0)) * 100
        fc_gen = eval_metrics.get("eval/acc_gen_fact_checking", eval_metrics.get("accuracy/fact_checking_gen", 0)) * 100

        print(
            f"Epoch {epoch_num:02d} | "
            f"Chaining [A_mem={ch_mem:.1f}%, A_gen={ch_gen:.1f}%, KU-Gap={ch_gap:+.1f}%] | "
            f"Inter [A_mem={in_mem:.1f}%, A_gen={in_gen:.1f}%] | "
            f"FC [A_mem={fc_mem:.1f}%, A_gen={fc_gen:.1f}%]",
            flush=True
        )

        try:
            wandb.log(eval_metrics)
        except Exception:
            pass

        model.unload()

    # Save JSON summary
    out_json = ckpt_dir / "preliminary_eval_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved preliminary evaluation results to {out_json}")

    try:
        wandb.finish()
    except Exception:
        pass

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Epoch':<6} | {'Chaining A_mem':<14} | {'Chaining A_gen':<14} | {'Chaining KU-Gap':<15} | {'Inter A_gen':<12} | {'FC A_gen':<10} | {'Overall A_gen':<13}")
    print("-" * 105)
    for r in results:
        ep = r.get("epoch", 0)
        ch_mem = r.get("eval/acc_mem_chaining", r.get("accuracy/chaining_mem", 0)) * 100
        ch_gen = r.get("eval/acc_gen_chaining", r.get("accuracy/chaining_gen", 0)) * 100
        ch_gap = r.get("eval/ku_gap_chaining", r.get("kug_gap/chaining", 0)) * 100
        in_gen = r.get("eval/acc_gen_intersection", r.get("accuracy/intersection_gen", 0)) * 100
        fc_gen = r.get("eval/acc_gen_fact_checking", r.get("accuracy/fact_checking_gen", 0)) * 100
        ov_gen = (ch_gen * 178 + in_gen * 184 + fc_gen * 35) / 397
        print(f"{ep:<6d} | {ch_mem:>13.1f}% | {ch_gen:>13.1f}% | {ch_gap:>+14.1f}% | {in_gen:>11.1f}% | {fc_gen:>9.1f}% | {ov_gen:>12.1f}%")
    print("=" * 105)


if __name__ == "__main__":
    main()
