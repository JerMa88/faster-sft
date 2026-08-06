"""
Final Training & Results Summary Generator
============================================
Compiles metrics across all 6 model families (12 experiment runs)
on STaRK-Prime and STaRK-MAG datasets.
"""

import json
import glob
from pathlib import Path

MODEL_NAMES = {
    "llama3.2-3b":    "LLaMA-3.2-3B-Instruct",
    "qwen3.5-2b":     "Qwen3.5-2B",
    "gemma2-2b":      "Gemma-2-2B-It",
    "antares-1b":     "Antares-1B",
    "nanbeige4.2-3b": "Nanbeige4.2-3B",
    "lfm2.5-1.2b":    "LFM2.5-1.2B-Base",
}

def main():
    print("=" * 80)
    print(" ALIGNMENT-AWARE SFT (6-MODEL SWEEP) — FINAL TRAINING SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Model Key':<16} | {'Model Architecture':<22} | {'Dataset':<12} | {'Init Loss':<10} | {'Final Loss':<10} | {'Status'}")
    print("-" * 80)

    summary_data = []

    for path in sorted(glob.glob("outputs/runs/**/metrics.json", recursive=True)):
        p = Path(path)
        model_key = p.parents[2].name
        dataset = p.parents[1].name
        model_arch = MODEL_NAMES.get(model_key, model_key)

        with open(path) as f:
            metrics = json.load(f)

        if metrics and isinstance(metrics, list):
            init_loss = metrics[0].get("total_loss", 0.0)
            final_loss = metrics[-1].get("total_loss", 0.0)
            n_epochs = len(metrics)
            status = f"Completed ({n_epochs}/50 epochs)"

            summary_data.append({
                "model_key": model_key,
                "model_arch": model_arch,
                "dataset": dataset,
                "init_loss": round(init_loss, 4),
                "final_loss": round(final_loss, 4),
                "epochs": n_epochs,
            })

            print(f"{model_key:<16} | {model_arch:<22} | {dataset:<12} | {init_loss:<10.4f} | {final_loss:<10.4f} | {status}")

    print("=" * 80)

    # Save summary
    out_file = Path("outputs/final_experiment_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved consolidated summary report → {out_file}")

if __name__ == "__main__":
    main()
