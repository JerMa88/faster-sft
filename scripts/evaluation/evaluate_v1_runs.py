"""
Batch Evaluator for Faster-SFT
================================
Runs evaluation on all trained checkpoints in `outputs/runs/`, 
computes A_mem and A_gen exact match metrics across datasets,
and generates a consolidated summary table.

Usage:
    python scripts/evaluate_all.py
"""

import os
import sys
import json
import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

from src.evaluation.evaluator import evaluate_run

MODEL_ID_MAP = {
    "llama3.2-3b":    "meta-llama/Llama-3.2-3B-Instruct",
    "qwen3.5-2b":     "Qwen/Qwen3.5-2B",
    "gemma2-2b":      "google/gemma-2-2b-it",       # legacy runs
    "gemma4-e4b":     "google/gemma-4-E4B-it",       # new
    "antares-1b":     "fdtn-ai/antares-1b",
    "nanbeige4.2-3b": "Nanbeige/Nanbeige4.2-3B",
    "lfm2.5-1.2b":    "LiquidAI/LFM2.5-1.2B-Base",
}

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default=None, help="Filter by model key")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size (reduce for CUDA-unstable models)")
    args = parser.parse_args()

    # Slurm sets CUDA_VISIBLE_DEVICES when --gres=gpu:1 is requested.
    # If it's set but torch can't see a GPU, something went wrong with CUDA
    # init (race condition on multi-job nodes). Abort loudly rather than
    # silently running 10+ hours on CPU.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={cuda_visible!r} but torch.cuda.is_available()=False. "
            "Likely a CUDA init race on a multi-job node. Resubmit the job."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}  Dtype: {dtype}  CUDA_VISIBLE_DEVICES={cuda_visible!r}")

    if args.model_key:
        runs = sorted(glob.glob(f"outputs/runs/{args.model_key}/*/*"))
    else:
        runs = sorted(glob.glob("outputs/runs/*/*/*"), key=lambda p: (0 if "qwen" in p else (2 if "nanbeige" in p else 1), p))
    print(f"Found {len(runs)} run directories to evaluate.\n")

    results = {}
    for run_dir in runs:
        path = Path(run_dir)
        model_key = path.parents[1].name
        dataset_name = path.parents[0].name
        
        base_model_id = MODEL_ID_MAP.get(model_key)
        if not base_model_id:
            print(f"Skipping unknown model key: {model_key}")
            continue

        data_file = ROOT / "data" / "processed" / f"stark_{dataset_name.replace('stark_', '')}_qa_v2.jsonl"
        if not data_file.exists():
            data_file = ROOT / "data" / "processed" / "stark_prime_qa_v2.jsonl"

        if (Path(run_dir) / "eval_results.json").exists():
            print(f"Skipping already evaluated run: {run_dir}")
            continue

        print(f"============================================================")
        print(f"Evaluating {model_key} | Dataset: {dataset_name}")
        print(f"Run dir: {run_dir}")
        print(f"============================================================")

        try:
            summary = evaluate_run(
                run_dir=run_dir,
                base_model_id=base_model_id,
                data_path=str(data_file),
                device=device,
                dtype=dtype,
                batch_size=args.batch_size,
                hf_cache=HF_CACHE,
                verbose=True,
            )
            results[f"{model_key}_{dataset_name}"] = summary
        except Exception as e:
            print(f"  [ERROR] Evaluation failed for {run_dir}: {e}")

    summary_file = ROOT / "outputs" / "final_evaluation_summary_v1_reeval.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n============================================================")
    print(f"✓ Final Evaluation Complete! Summary saved to {summary_file}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
