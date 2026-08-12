"""
Dedicated Baseline SFT Replication Plot (Figure 7)
===================================================
Generates 3 separate panels, one per task type (chaining, intersection, fact_checking),
showing A_mem and A_gen across 50 epochs for Baseline SFT (Method 1).

Pass/Fail thresholds from Section C.2 of Dai et al. (2025):
  - Chaining:       A_mem >= 90% by epoch 10,  A_gen < 5% throughout
  - Intersection:   A_mem >= 90% by epoch 10,  A_gen > 90% by epoch 15
  - Fact Checking:  A_mem >= 90% by epoch 10,  A_gen ~ 50% throughout

Usage:
    python scripts/plotting/plot_baseline_sft.py
    python scripts/plotting/plot_baseline_sft.py --run_id <wandb_run_id>
    python scripts/plotting/plot_baseline_sft.py --ckpt_dir outputs/kug_overhaul_v2/baseline_qwen2.5-1.5b
"""

import os
import re
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

import wandb

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.5,
})

TASK_INFO = {
    "chaining": {
        "title": "Task A: Chaining (Multi-hop Reasoning)",
        "color": "#1a6faf",
        "threshold_mem": 0.90,
        "threshold_gen_max": 0.05,
        "threshold_gen_label": "$A_{gen}$ < 5%",
    },
    "intersection": {
        "title": "Task B: Intersection",
        "color": "#d45f00",
        "threshold_mem": 0.90,
        "threshold_gen_min": 0.90,
        "threshold_gen_label": "$A_{gen}$ > 90%",
    },
    "fact_checking": {
        "title": "Task C: Fact Checking",
        "color": "#1a8a4a",
        "threshold_mem": 0.90,
        "threshold_gen_target": 0.50,
        "threshold_gen_label": "$A_{gen}$ ≈ 50%",
    },
}


def fetch_wandb_history(run_id, entity="jerma88-smu", project="kug_overhaul_qwen1.5b"):
    """Fetch logged history from W&B API for a given run ID."""
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = list(run.scan_history())
        df = pd.DataFrame(history)
        if df.empty:
            return {}
        results = {}
        eval_cols = [c for c in df.columns if c.startswith("eval/")]
        for _, row in df.iterrows():
            if pd.notna(row.get("eval/acc_mem_chaining")):
                ep = row.get("epoch")
                if pd.isna(ep):
                    ep = row.get("eval/epoch")
                if pd.notna(ep):
                    epoch_val = int(ep)
                    metrics = {}
                    for col in eval_cols:
                        val = row.get(col)
                        if pd.notna(val) and isinstance(val, (int, float, np.number)):
                            metrics[col] = float(val)
                    if metrics:
                        results[epoch_val] = metrics
        return results
    except Exception as e:
        print(f"  W&B fetch error for {run_id}: {e}")
        return {}


def parse_eval_log(log_path):
    """Parse evaluation stdout log file as fallback."""
    if not os.path.exists(log_path):
        return {}
    with open(log_path, "r") as f:
        content = f.read()
    epoch_blocks = re.split(r"(?:---|)?Evaluating Epoch\s+", content)
    results = {}
    for block in epoch_blocks[1:]:
        lines = block.strip().split("\n")
        match = re.search(r"(\d+)", lines[0])
        if not match:
            continue
        epoch = int(match.group(1))
        metrics = {}
        for line in lines:
            if "eval/" in line and ":" in line:
                parts = line.strip().split(":")
                key = parts[0].strip()
                try:
                    val = float(parts[1].strip())
                    metrics[key] = val
                except ValueError:
                    pass
        if metrics:
            results[epoch] = metrics
    return results


def load_eval_data(ckpt_dir, run_id, project="kug_overhaul_qwen1.5b"):
    """Merge log file + W&B API data for a given checkpoint directory."""
    merged = {}

    # 1. Parse stdout log files
    log_patterns = [
        f"outputs/logs/*eval*baseline*.out",
        f"outputs/logs/*eval*v2*baseline*.out",
    ]
    for pattern in log_patterns:
        for log_path in glob.glob(pattern):
            data = parse_eval_log(log_path)
            for ep, metrics in data.items():
                if ep not in merged:
                    merged[ep] = {}
                merged[ep].update(metrics)

    # 2. Merge W&B API
    if run_id:
        data_wandb = fetch_wandb_history(run_id, project=project)
        for ep, metrics in data_wandb.items():
            if ep not in merged:
                merged[ep] = {}
            merged[ep].update(metrics)

    return merged


def check_threshold(epochs, data, task):
    """
    Check if pass/fail thresholds are met.
    Returns (mem_pass, gen_pass, epoch_90_mem) 
    """
    t = TASK_INFO[task]
    mem_threshold = t["threshold_mem"]

    epoch_90_mem = None
    for ep in sorted(epochs):
        v = data[ep].get(f"eval/acc_mem_{task}", 0.0)
        if v >= mem_threshold and epoch_90_mem is None:
            epoch_90_mem = ep

    # Final mem at epoch 10+
    eps_10plus = [ep for ep in sorted(epochs) if ep >= 10]
    if eps_10plus:
        mem_at_10 = data[eps_10plus[0]].get(f"eval/acc_mem_{task}", 0.0)
        mem_pass = mem_at_10 >= mem_threshold
    else:
        mem_pass = False

    # Check gen threshold
    if "threshold_gen_max" in t:
        gen_vals = [data[ep].get(f"eval/acc_gen_{task}", 1.0) for ep in sorted(epochs)]
        gen_pass = all(v < t["threshold_gen_max"] for v in gen_vals)
    elif "threshold_gen_min" in t:
        late_eps = [ep for ep in sorted(epochs) if ep >= 15]
        if late_eps:
            gen_at_15 = data[late_eps[0]].get(f"eval/acc_gen_{task}", 0.0)
            gen_pass = gen_at_15 >= t["threshold_gen_min"]
        else:
            gen_pass = False
    elif "threshold_gen_target" in t:
        target = t["threshold_gen_target"]
        mid_eps = [ep for ep in sorted(epochs) if 10 <= ep <= 40]
        if mid_eps:
            gen_vals = [data[ep].get(f"eval/acc_gen_{task}", 0.0) for ep in mid_eps]
            avg_gen = np.mean(gen_vals)
            gen_pass = 0.35 <= avg_gen <= 0.65  # ~50% within ±15%
        else:
            gen_pass = False
    else:
        gen_pass = False

    return mem_pass, gen_pass, epoch_90_mem


def plot_baseline_sft(eval_data, output_path="figures/baseline_sft_figure7_replication.png"):
    """
    Generate 3-panel Figure 7 replication for Baseline SFT.
    One panel per task: chaining, intersection, fact_checking.
    Includes pass/fail threshold annotations.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    epochs_all = sorted(eval_data.keys())

    for col_idx, task in enumerate(["chaining", "intersection", "fact_checking"]):
        ax = axes[col_idx]
        info = TASK_INFO[task]
        color = info["color"]

        if not epochs_all:
            ax.text(0.5, 0.5, "No evaluation data yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(info["title"])
            continue

        mem_accs = [eval_data[ep].get(f"eval/acc_mem_{task}", float("nan")) * 100 for ep in epochs_all]
        gen_accs = [eval_data[ep].get(f"eval/acc_gen_{task}", float("nan")) * 100 for ep in epochs_all]

        ax.plot(epochs_all, mem_accs, color=color, linestyle="-", linewidth=2.8, label="$A_{mem}$ (Memorization)")
        ax.plot(epochs_all, gen_accs, color=color, linestyle="--", linewidth=2.2, alpha=0.85, label="$A_{gen}$ (Generalization)")

        # Threshold line for A_mem >= 90%
        ax.axhline(y=90, color="grey", linestyle=":", linewidth=1.5, alpha=0.7, label="90% threshold")
        ax.axvline(x=10, color="grey", linestyle=":", linewidth=1.5, alpha=0.7, label="Epoch 10")

        # Task-specific threshold band
        if task == "chaining":
            ax.axhspan(0, 5, alpha=0.08, color="red", label="Target $A_{gen}$ zone (<5%)")
        elif task == "intersection":
            ax.axhspan(90, 100, alpha=0.08, color=color, label="Target $A_{gen}$ zone (>90%)")
        elif task == "fact_checking":
            ax.axhspan(40, 60, alpha=0.08, color=color, label="Target $A_{gen}$ zone (40–60%)")

        # Pass/fail annotation
        mem_pass, gen_pass, epoch_90 = check_threshold(epochs_all, eval_data, task)
        n_evaluated = len(epochs_all)
        status_lines = [
            f"Evaluated: {n_evaluated}/50 epochs",
            f"$A_{{mem}}$ ≥ 90% by ep10: {'✓ PASS' if mem_pass else '✗ FAIL/TBD'}",
            f"{info['threshold_gen_label']}: {'✓ PASS' if gen_pass else '✗ FAIL/TBD'}",
        ]
        if epoch_90:
            status_lines.append(f"$A_{{mem}}$ crossed 90% at epoch {epoch_90}")

        ax.text(
            0.02, 0.98, "\n".join(status_lines),
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

        ax.set_title(info["title"], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Training Epoch", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_xlim(1, max(50, max(epochs_all)))
        ax.set_ylim(-2, 107)
        ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.55)

    plt.suptitle(
        "Baseline SFT — Figure 7 Replication (Qwen 2.5 1.5B, MAG Dataset)\n"
        "Completion-Only Loss Masking | KUG Overhaul v2",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline SFT figure to: {output_path}")
    print(f"Saved baseline SFT PDF to:    {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Baseline SFT (Figure 7 Replication)")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/kug_overhaul_v2/baseline_qwen2.5-1.5b")
    parser.add_argument("--run_id", type=str, default=None, help="Override W&B run ID")
    parser.add_argument("--project", type=str, default="kug_overhaul_qwen1.5b")
    parser.add_argument("--output", type=str, default="figures/baseline_sft_figure7_replication.png")
    args = parser.parse_args()

    # Load run_id from metadata if not provided
    run_id = args.run_id
    meta_path = os.path.join(args.ckpt_dir, "run_metadata.json")
    if run_id is None and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        run_id = meta.get("wandb_run_id")
        print(f"Loaded W&B run ID from metadata: {run_id}")

    print(f"Loading evaluation data...")
    eval_data = load_eval_data(args.ckpt_dir, run_id, args.project)
    print(f"Loaded data for {len(eval_data)} epochs: {sorted(eval_data.keys())}")

    if not eval_data:
        print("WARNING: No evaluation data found. Plotting empty figure.")

    plot_baseline_sft(eval_data, args.output)


if __name__ == "__main__":
    main()
