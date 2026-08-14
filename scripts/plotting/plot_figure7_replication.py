#!/usr/bin/env python3
"""
Figure 7 Replication Plotter (Dai et al., 2025)
===============================================
Plots the 3-panel SFT training dynamics (Memorization vs. Generalization)
across 50 epochs with per-epoch training gradient norm on a secondary axis:
  - Panel 1: Chaining      (A_mem -> 98.9%, A_gen -> 3.4%, persistent KU-gap)
  - Panel 2: Intersection  (A_mem -> 92.4%, A_gen -> 93.5%, parallel convergence)
  - Panel 3: Fact Checking (A_mem -> 100.0%, A_gen -> 52.6%, ~0.50 binary baseline)

Outputs are saved directly to ./figures/ as both high-resolution PNG and PDF.

Usage:
  python scripts/plotting/plot_figure7_replication.py
  python scripts/plotting/plot_figure7_replication.py --log outputs/logs/fast_eval_475392.out
  python scripts/plotting/plot_figure7_replication.py --theme light --out figures/figure7.png
"""

import os
import re
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


PAPER_TARGETS = {
    "chaining":      {"mem": 0.95, "gen": 0.03, "label": "Chaining (Multi-Hop)"},
    "intersection":  {"mem": 0.95, "gen": 0.90, "label": "Intersection"},
    "fact_checking": {"mem": 0.95, "gen": 0.50, "label": "Fact Checking"},
}

TASK_ORDER = ["chaining", "intersection", "fact_checking"]


def find_latest_log() -> str:
    """Find the most relevant/latest fast_eval output log."""
    logs = glob.glob("outputs/logs/fast_eval_*.out")
    if not logs:
        logs = glob.glob("fast_eval_*.out")
    if logs:
        logs.sort(key=os.path.getmtime, reverse=True)
        return logs[0]
    return "outputs/logs/fast_eval_475392.out"


def load_training_gradient_stats(custom_path: str = None) -> dict:
    """Load per-epoch training gradient norm and loss stats."""
    cache_path = custom_path or "outputs/metrics/epoch_training_stats.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}

    # Try fetching from W&B if available
    try:
        import wandb
        import pandas as pd
        api = wandb.Api()
        run = api.run("jerma88-smu/kug_overhaul_qwen1.5b/c39m3zuh")
        history = list(run.scan_history())
        df = pd.DataFrame(history)
        steps_per_epoch = 70
        epoch_stats = {}
        for ep in range(1, 51):
            start_step = (ep - 1) * steps_per_epoch + 1
            end_step = ep * steps_per_epoch
            ep_df = df[(df["step"] >= start_step) & (df["step"] <= end_step)]
            if not ep_df.empty and "train/grad_norm" in ep_df:
                epoch_stats[ep] = {
                    "grad_norm": float(ep_df["train/grad_norm"].mean()),
                    "loss": float(ep_df["train/step_loss_total"].mean()),
                }
        if epoch_stats:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(epoch_stats, f, indent=2)
            return epoch_stats
    except Exception as e:
        print(f"Warning: Could not fetch training gradient stats from W&B: {e}")

    return {}


def parse_eval_log(log_path: str) -> dict:
    """Parse fast_eval stdout log into per-task per-epoch accuracy arrays."""
    results = {
        task: {"epoch": [], "mem": [], "gen": [], "gap": []}
        for task in TASK_ORDER
    }

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    current_epoch = None
    epoch_data = {}

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Evaluating Epoch (\d+)\.\.\.", line)
            if m:
                if current_epoch is not None and epoch_data:
                    for task in results:
                        mem_key = f"eval/acc_mem_{task}"
                        gen_key = f"eval/acc_gen_{task}"
                        gap_key = f"eval/ku_gap_{task}"
                        if mem_key in epoch_data:
                            results[task]["epoch"].append(current_epoch)
                            results[task]["mem"].append(epoch_data[mem_key])
                            results[task]["gen"].append(epoch_data.get(gen_key, 0.0))
                            results[task]["gap"].append(epoch_data.get(gap_key, 0.0))
                current_epoch = int(m.group(1))
                epoch_data = {}
                continue

            m = re.match(r"eval/(\S+):\s+([\d.]+)", line)
            if m:
                epoch_data[f"eval/{m.group(1)}"] = float(m.group(2))

    # Commit last epoch
    if current_epoch is not None and epoch_data:
        for task in results:
            mem_key = f"eval/acc_mem_{task}"
            gen_key = f"eval/acc_gen_{task}"
            gap_key = f"eval/ku_gap_{task}"
            if mem_key in epoch_data:
                results[task]["epoch"].append(current_epoch)
                results[task]["mem"].append(epoch_data[mem_key])
                results[task]["gen"].append(epoch_data.get(gen_key, 0.0))
                results[task]["gap"].append(epoch_data.get(gap_key, 0.0))

    return results


def plot_figure7(
    results: dict,
    train_stats: dict,
    out_path: str,
    theme: str = "light",
    title_suffix: str = "",
    stage_split: int = None,
    main_title: str = None,
):
    """
    Generate 3-panel Figure 7 matching paper publication style,
    with training gradient norm plotted at every epoch on secondary axis.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_completed = max(len(results[t]["epoch"]) for t in TASK_ORDER)
    if n_completed == 0:
        print("ERROR: No completed epoch data found to plot.")
        return

    # Style definitions
    if theme == "dark":
        bg_fig = "#0F1117"
        bg_ax = "#1A1D27"
        text_color = "#FFFFFF"
        grid_color = "#2E3346"
        spine_color = "#3A3D4A"
        color_mem = "#3B82F6"
        color_gen = "#EF4444"
        color_grad = "#A855F7"
        fill_color = "#3B82F6"
        legend_bg = "#222634"
        legend_edge = "#3A3D4A"
    else:
        bg_fig = "#FFFFFF"
        bg_ax = "#F8FAFC"
        text_color = "#0F172A"
        grid_color = "#E2E8F0"
        spine_color = "#CBD5E1"
        color_mem = "#1D4ED8"
        color_gen = "#DC2626"
        color_grad = "#7C3AED"
        fill_color = "#93C5FD"
        legend_bg = "#FFFFFF"
        legend_edge = "#E2E8F0"

    fig = plt.figure(figsize=(17, 5.5))
    fig.patch.set_facecolor(bg_fig)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    for col, task in enumerate(TASK_ORDER):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(bg_ax)

        epochs = results[task]["epoch"]
        mem = results[task]["mem"]
        gen = results[task]["gen"]

        if not epochs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=text_color, transform=ax.transAxes)
            continue

        ep = np.array(epochs)
        m = np.array(mem)
        g = np.array(gen)

        ax.fill_between(ep, g, m, alpha=0.16 if theme == "light" else 0.13, color=fill_color, label="_nolegend_")

        l1, = ax.plot(ep, m, color=color_mem, linewidth=2.8, label=r"Memorization ($A_{mem}$)", marker="o", markersize=4.5, markevery=max(1, len(ep) // 10), zorder=4)
        l2, = ax.plot(ep, g, color=color_gen, linewidth=2.8, label=r"Generalization ($A_{gen}$)", marker="s", markersize=4.5, markevery=max(1, len(ep) // 10), zorder=4)

        pt = PAPER_TARGETS[task]
        l3 = ax.axhline(pt["mem"], color=color_mem, linestyle="--", linewidth=1.2, alpha=0.55, label=f"Paper $A_{{mem}}$ Target ({pt['mem']:.0%})", zorder=3)
        l4 = ax.axhline(pt["gen"], color=color_gen, linestyle="--", linewidth=1.2, alpha=0.55, label=f"Paper $A_{{gen}}$ Target ({pt['gen']:.0%})", zorder=3)

        l_split = None
        if stage_split is not None:
            l_split = ax.axvline(
                x=stage_split, color="#F59E0B" if theme == "light" else "#FBBF24",
                linestyle="-.", linewidth=1.5, alpha=0.9,
                label=f"Stage Switch (Ep {stage_split})", zorder=3
            )

        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        grad_epochs = [e for e in ep if e in train_stats]
        grad_norms = [train_stats[e].get("grad_norm", train_stats[e].get("loss_kl", train_stats[e].get("loss_total", 0.0))) for e in grad_epochs]

        if grad_epochs and any(v > 0 for v in grad_norms):
            l5, = ax2.plot(grad_epochs, grad_norms, color=color_grad, linestyle=":", linewidth=1.8, alpha=0.85, marker="^", markersize=3.5, markevery=max(1, len(grad_epochs) // 8), label=r"Training Dynamics ($\|\nabla \mathcal{L}\|_2$ / $\mathcal{L}_{KL}$)", zorder=2)
            ax2.set_ylim(0, max(3.5, max(grad_norms) * 1.25 if grad_norms else 3.5))
            ax2.tick_params(colors=color_grad, labelsize=9)
            ax2.spines["right"].set_color(color_grad)
            if col == 2:
                ax2.set_ylabel(r"Training Dynamics ($\|\nabla \mathcal{L}\|_2$ / $\mathcal{L}_{KL}$)", color=color_grad, fontsize=11, fontweight="bold", labelpad=6)
        else:
            l5 = None

        if len(ep) > 0:
            final_ep = ep[-1]
            final_m = m[-1]
            final_g = g[-1]
            gap = final_m - final_g
            annotation_text = (
                f"Epoch {final_ep}\n"
                f"$A_{{mem}}$: {final_m:.1%}\n"
                f"$A_{{gen}}$: {final_g:.1%}\n"
                rf"$\Delta A$: {gap:+.1%}"
            )
            ax.text(
                0.03, 0.05, annotation_text,
                transform=ax.transAxes,
                fontsize=8.5,
                fontweight="medium",
                color=text_color,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_ax, edgecolor=spine_color, alpha=0.9)
            )

        task_title = PAPER_TARGETS[task]["label"]
        ax.set_title(f"Task {chr(65+col)}: {task_title}", color=text_color, fontsize=13.0, fontweight="bold", pad=12)
        ax.set_xlabel("Epochs", color=text_color, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Accuracy", color=text_color, fontsize=11, fontweight="bold")
        ax.set_xlim(0, max(50, max(epochs)))
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100)}%"))
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7, color=grid_color)

        for s in ax.spines.values():
            s.set_color(spine_color)
            s.set_linewidth(1.0)
        ax.tick_params(colors=text_color, labelsize=9.5)

        lines = [l1, l2, l3, l4]
        if l_split is not None:
            lines.append(l_split)
        if l5 is not None:
            lines.append(l5)
        labels = [l.get_label() for l in lines]
        ax.legend(
            lines, labels, fontsize=7.5,
            loc="lower right" if col != 1 else "center right",
            framealpha=0.92, facecolor=legend_bg, edgecolor=legend_edge,
            labelcolor=text_color
        )

    header = main_title or (
        r"Figure 7 Replication — SFT Training Dynamics & Gradient Norm (Qwen2.5-1.5B on STaRK)" + "\n" +
        r"Memorization ($A_{mem}$), Generalization ($A_{gen}$), and Gradient Norm $\|\nabla \mathcal{L}\|_2$ across 50 Epochs"
    )
    fig.suptitle(header + title_suffix, color=text_color, fontsize=13.5, fontweight="bold", y=1.03)

    png_path = str(out_path)
    pdf_path = str(out_path.with_suffix(".pdf"))

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"  Saved PNG (300 DPI) -> {png_path}")
    print(f"  Saved Vector PDF    -> {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Recreate Figure 7 replication plot with per-epoch gradient norm.")
    parser.add_argument("--log", type=str, default=None, help="Path to fast_eval log. Defaults to latest.")
    parser.add_argument("--stats", type=str, default=None, help="Path to training gradient stats JSON.")
    parser.add_argument("--out", type=str, default="figures/figure7.png", help="Output PNG path.")
    parser.add_argument("--theme", type=str, default="all", choices=["light", "dark", "all"],
                        help="Plot style theme: light (paper white), dark (sleek dark mode), or all.")
    parser.add_argument("--stage_split", type=int, default=None, help="Epoch where 2-stage switch occurs (e.g. 15).")
    parser.add_argument("--title", type=str, default=None, help="Custom super title for figure.")
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    print(f"Loading evaluation log: {log_path}")
    results = parse_eval_log(log_path)
    total_epochs = max(len(v["epoch"]) for v in results.values())
    print(f"Found {total_epochs} evaluated epochs across {len(results)} tasks.")

    train_stats = load_training_gradient_stats(args.stats)
    print(f"Loaded training gradient stats for {len(train_stats)} epochs.")

    out_base = Path(args.out)

    if args.theme in ["light", "all"]:
        light_path = out_base.parent / f"{out_base.stem}.png"
        print(f"Generating Publication (Light) Figure 7 with Gradient Norm...")
        plot_figure7(results, train_stats, light_path, theme="light", stage_split=args.stage_split, main_title=args.title)

    if args.theme in ["dark", "all"]:
        dark_path = out_base.parent / f"{out_base.stem}_dark.png"
        print(f"Generating Dark-Theme Figure 7 with Gradient Norm...")
        plot_figure7(results, train_stats, dark_path, theme="dark", stage_split=args.stage_split, main_title=args.title)

    print(f"All Figure 7 plots successfully saved for {out_base.stem} in ./figures/!")


if __name__ == "__main__":
    main()
