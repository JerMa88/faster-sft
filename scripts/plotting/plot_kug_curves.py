import os
import re
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb

# Use clean publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def fetch_wandb_history(run_id, entity="jerma88-smu", project="kug_overhaul_qwen1.5b"):
    """
    Fetches logged history dataframe from W&B API for a given run ID.
    Returns dict: {epoch: {metric_name: value}}
    """
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = list(run.scan_history())
        df = pd.DataFrame(history)

        if df.empty:
            return {}

        results = {}
        eval_cols = [c for c in df.columns if c.startswith("eval/")]

        # Look for rows where eval/acc_mem_chaining exists
        for _, row in df.iterrows():
            if pd.notna(row.get("eval/acc_mem_chaining")):
                # Check epoch from row['epoch'] or row['eval/epoch'] or derive
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
        print(f"Error fetching W&B run {run_id}: {e}")
        return {}

def parse_eval_log(log_path):
    """
    Fallback parser for evaluation stdout log file.
    """
    if not os.path.exists(log_path):
        return {}

    with open(log_path, 'r') as f:
        content = f.read()

    # Split by Evaluating Epoch
    epoch_blocks = re.split(r"(?:---|)?Evaluating Epoch\s+", content)
    results = {}

    for block in epoch_blocks[1:]:
        lines = block.strip().split("\n")
        header = lines[0]
        match = re.search(r"(\d+)", header)
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

def get_run_metadata(ckpt_dir):
    meta_path = os.path.join(ckpt_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}

def plot_figure7_replication(eval_data_dict, output_path="figures/figure7_kug_curves_replication.png"):
    """
    Plots Figure 7 replication with 3 panels:
    Panel A: Method 1 (Baseline SFT)
    Panel B: Method 2 (2-Stage Mem-then-Gen SFT)
    Panel C: Method 3 (Joint Supervised SFT)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    methods = [
        ("baseline", "Method 1: Baseline SFT ($P_{mem}$ only)", axes[0]),
        ("two_stage", "Method 2: 2-Stage SFT (Mem $\\to$ Gen)", axes[1]),
        ("joint", "Method 3: Joint Supervised SFT ($P_{mem} + P_{gen}$)", axes[2])
    ]

    task_colors = {
        "chaining": "#1f77b4",        # Blue
        "intersection": "#ff7f0e",    # Orange
        "fact_checking": "#2ca02c"    # Green
    }

    task_names = {
        "chaining": "Chaining (Multi-hop)",
        "intersection": "Intersection",
        "fact_checking": "Fact Checking"
    }

    for method_key, title, ax in methods:
        data = eval_data_dict.get(method_key, {})
        epochs = sorted(data.keys())

        if not epochs:
            ax.set_title(f"{title}\n(Evaluating...)", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            continue

        for task in ["chaining", "intersection", "fact_checking"]:
            mem_accs = [data[e].get(f"eval/acc_mem_{task}", 0.0) * 100 for e in epochs]
            gen_accs = [data[e].get(f"eval/acc_gen_{task}", 0.0) * 100 for e in epochs]

            color = task_colors[task]
            label = task_names[task]

            # Solid line for Memorization Accuracy
            ax.plot(epochs, mem_accs, label=f"{label} ($A_{{mem}}$)", color=color, linestyle='-', linewidth=2.4)
            # Dashed line for Generalization Accuracy
            ax.plot(epochs, gen_accs, label=f"{label} ($A_{{gen}}$)", color=color, linestyle='--', linewidth=2.0, alpha=0.85)

        # Add vertical line for 2-stage transition at epoch 15
        if method_key == "two_stage":
            ax.axvline(x=15, color='red', linestyle=':', linewidth=1.8, label='Loss Switch (Epoch 15)')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Training Epochs", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(1, max(50, max(epochs) if epochs else 50))
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches='tight')
    print(f"Saved Figure 7 replication plot to {output_path}")

def plot_kug_gap_comparison(eval_data_dict, output_path="figures/kug_gap_comparison.png"):
    """
    Plots direct Knowledge Understanding Gap (Delta A = A_mem - A_gen) across epochs for each method.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    methods = [
        ("baseline", "Method 1: Baseline SFT", axes[0]),
        ("two_stage", "Method 2: 2-Stage SFT", axes[1]),
        ("joint", "Method 3: Joint Supervised SFT", axes[2])
    ]

    task_colors = {
        "chaining": "#1f77b4",
        "intersection": "#ff7f0e",
        "fact_checking": "#2ca02c"
    }

    for method_key, title, ax in methods:
        data = eval_data_dict.get(method_key, {})
        epochs = sorted(data.keys())

        if not epochs:
            ax.set_title(f"{title}\n(Evaluating...)", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            continue

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)

        for task in ["chaining", "intersection", "fact_checking"]:
            gaps = [data[e].get(f"eval/ku_gap_{task}", 0.0) * 100 for e in epochs]
            ax.plot(epochs, gaps, label=task.replace("_", " ").title(), color=task_colors[task], linewidth=2.4)

        if method_key == "two_stage":
            ax.axvline(x=15, color='red', linestyle=':', linewidth=1.8, label='Loss Switch (Epoch 15)')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Training Epochs", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(1, max(50, max(epochs) if epochs else 50))

    axes[0].set_ylabel("KUG Gap $\Delta A = A_{mem} - A_{gen}$ (%)", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches='tight')
    print(f"Saved KUG Gap comparison plot to {output_path}")

def main():
    dirs = {
        "baseline": "outputs/kug_overhaul/baseline_qwen2.5-1.5b",
        "two_stage": "outputs/kug_overhaul/two_stage_qwen2.5-1.5b",
        "joint": "outputs/kug_overhaul/joint_qwen2.5-1.5b"
    }

    eval_data = {}
    for key, path in dirs.items():
        meta = get_run_metadata(path)
        run_id = meta.get("wandb_run_id")
        project = meta.get("wandb_project", "kug_overhaul_qwen1.5b")

        print(f"Processing {key} (Run ID: {run_id})...")

        merged = {}
        # 1. Parse log files first
        log_matches = sorted(glob.glob(f"outputs/logs/*eval*{key}*.out"))
        for log_path in log_matches:
            data_log = parse_eval_log(log_path)
            for ep, metrics in data_log.items():
                if ep not in merged:
                    merged[ep] = {}
                merged[ep].update(metrics)

        # 2. Merge W&B API data
        if run_id:
            data_wandb = fetch_wandb_history(run_id, project=project)
            for ep, metrics in data_wandb.items():
                if ep not in merged:
                    merged[ep] = {}
                merged[ep].update(metrics)

        print(f"  Total merged epochs for {key}: {len(merged)} -> {sorted(merged.keys())}")
        eval_data[key] = merged

    plot_figure7_replication(eval_data, "figures/figure7_kug_curves_replication.png")
    plot_kug_gap_comparison(eval_data, "figures/kug_gap_comparison.png")

if __name__ == "__main__":
    main()
