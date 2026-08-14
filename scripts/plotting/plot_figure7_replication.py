#!/usr/bin/env python3
"""
Figure 7 Replication Plotter (Dai et al., 2025)
===============================================
Plots the 3-panel SFT training dynamics (Memorization vs. Generalization)
across 50 epochs:
  - Panel 1: Chaining      (A_mem -> 98.9%, A_gen -> 3.4%, persistent KU-gap)
  - Panel 2: Intersection  (A_mem -> 92.4%, A_gen -> 93.5%, parallel convergence)
  - Panel 3: Fact Checking (A_mem -> 100.0%, A_gen -> 52.6%, ~0.50 binary baseline)

Outputs are saved directly to ./figures/ as both high-resolution PNG and PDF.

Usage:
  python scripts/plotting/plot_figure7_replication.py
  python scripts/plotting/plot_figure7_replication.py --log outputs/logs/fast_eval_475392.out
  python scripts/plotting/plot_figure7_replication.py --theme light --out figures/figure7_paper.png
"""

import os
import re
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


def plot_figure7(results: dict, out_path: str, theme: str = "light", title_suffix: str = ""):
    """
    Generate 3-panel Figure 7 matching paper publication style.
    theme: 'light' (white publication style) or 'dark' (sleek presentation style)
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
        subtext_color = "#A0A5B5"
        grid_color = "#2E3346"
        spine_color = "#3A3D4A"
        color_mem = "#3B82F6"   # Bright Blue
        color_gen = "#EF4444"   # Bright Red
        fill_color = "#3B82F6"
        legend_bg = "#222634"
        legend_edge = "#3A3D4A"
    else:  # light (clean publication style)
        bg_fig = "#FFFFFF"
        bg_ax = "#F8FAFC"
        text_color = "#0F172A"
        subtext_color = "#475569"
        grid_color = "#E2E8F0"
        spine_color = "#CBD5E1"
        color_mem = "#1E40AF"   # Deep Navy Blue
        color_gen = "#DC2626"   # Crimson Red
        fill_color = "#93C5FD"
        legend_bg = "#FFFFFF"
        legend_edge = "#E2E8F0"

    fig = plt.figure(figsize=(16, 5.2))
    fig.patch.set_facecolor(bg_fig)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.28)

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

        # Decoupling shade (area between A_mem and A_gen)
        ax.fill_between(ep, g, m, alpha=0.15 if theme == "light" else 0.12, color=fill_color, label="_nolegend_")

        # Main curve plots
        ax.plot(
            ep, m, color=color_mem, linewidth=2.8,
            label=r"Memorization ($A_{mem}$)",
            marker="o", markersize=4.5, markevery=max(1, len(ep) // 10)
        )
        ax.plot(
            ep, g, color=color_gen, linewidth=2.8,
            label=r"Generalization ($A_{gen}$)",
            marker="s", markersize=4.5, markevery=max(1, len(ep) // 10)
        )

        # Paper Target Reference Lines
        pt = PAPER_TARGETS[task]
        ax.axhline(
            pt["mem"], color=color_mem, linestyle="--", linewidth=1.2,
            alpha=0.6, label=f"Paper $A_{{mem}}$ Target ({pt['mem']:.0%})"
        )
        ax.axhline(
            pt["gen"], color=color_gen, linestyle="--", linewidth=1.2,
            alpha=0.6, label=f"Paper $A_{{gen}}$ Target ({pt['gen']:.0%})"
        )

        # Final Epoch Values Annotation
        if len(ep) > 0:
            final_ep = ep[-1]
            last_m = m[-1]
            last_g = g[-1]

            # Offset logic for clear text placement
            m_offset_y = 0.04 if last_m >= last_g else -0.07
            g_offset_y = -0.08 if last_m >= last_g else 0.04
            if abs(last_m - last_g) < 0.05:
                m_offset_y = 0.05
                g_offset_y = -0.08

            ax.annotate(
                f"{last_m:.1%}",
                xy=(final_ep, last_m),
                xytext=(final_ep - 4, min(1.02, max(0.0, last_m + m_offset_y))),
                color=color_mem, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color_mem, lw=1.0)
            )
            ax.annotate(
                f"{last_g:.1%}",
                xy=(final_ep, last_g),
                xytext=(final_ep - 4, min(1.02, max(0.0, last_g + g_offset_y))),
                color=color_gen, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color_gen, lw=1.0)
            )

        # Styling
        ax.set_xlim(0, max(ep) + 1)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xlabel("Epoch", color=text_color, fontsize=12, fontweight="bold", labelpad=6)
        if col == 0:
            ax.set_ylabel("Accuracy", color=text_color, fontsize=12, fontweight="bold", labelpad=6)

        title = pt["label"]
        ax.set_title(f"Task {chr(65+col)}: {title}", color=text_color, fontsize=13, fontweight="bold", pad=10)

        # Grid and spines
        ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=text_color, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(1.0)

        # Legend
        ax.legend(
            fontsize=8.5, loc="lower right" if col != 1 else "center right",
            framealpha=0.9, facecolor=legend_bg, edgecolor=legend_edge,
            labelcolor=text_color
        )

    # Super title
    fig.suptitle(
        f"Figure 7 Replication — Standard SFT Training Dynamics (Qwen2.5-1.5B on STaRK)\n"
        f"Memorization ($A_{{mem}}$) vs. Generalization ($A_{{gen}}$) across 50 Epochs{title_suffix}",
        color=text_color, fontsize=14, fontweight="bold", y=1.03
    )

    # Save PNG & PDF
    png_path = str(out_path)
    pdf_path = str(out_path.with_suffix(".pdf"))

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"  Saved PNG (300 DPI) -> {png_path}")
    print(f"  Saved Vector PDF    -> {pdf_path}")


def print_summary_table(results: dict):
    """Print clean ASCII table of the replication metrics."""
    print("\n" + "=" * 78)
    print("  FIGURE 7 REPLICATION SUMMARY (Qwen2.5-1.5B Baseline SFT)")
    print("=" * 78)
    print(f"  {'Task':<16} | {'Final A_mem':<12} | {'Final A_gen':<12} | {'KU Gap':<10} | {'Paper Target':<18}")
    print("-" * 78)

    targets = {
        "chaining": "A_mem>95%, A_gen<5%",
        "intersection": "A_mem~92%, A_gen~90%",
        "fact_checking": "A_mem~100%, A_gen~50%",
    }

    for task in TASK_ORDER:
        ep = results[task]["epoch"]
        if ep:
            last_m = results[task]["mem"][-1]
            last_g = results[task]["gen"][-1]
            gap = last_m - last_g
            label = PAPER_TARGETS[task]["label"]
            print(f"  {label:<16} | {last_m:>10.1%}   | {last_g:>10.1%}   | {gap:>+8.1%}   | {targets[task]:<18}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Recreate Figure 7 replication plot using matplotlib.")
    parser.add_argument("--log", type=str, default=None, help="Path to fast_eval log. Defaults to latest.")
    parser.add_argument("--out", type=str, default="figures/figure7_sft_replication.png", help="Output PNG path.")
    parser.add_argument("--theme", type=str, default="all", choices=["light", "dark", "all"],
                        help="Plot style theme: light (paper white), dark (sleek dark mode), or all.")
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    print(f"Loading evaluation log: {log_path}")
    results = parse_eval_log(log_path)
    total_epochs = max(len(v["epoch"]) for v in results.values())
    print(f"Found {total_epochs} evaluated epochs across {len(results)} tasks.")

    print_summary_table(results)

    out_base = Path(args.out)

    if args.theme in ["light", "all"]:
        light_path = out_base.parent / f"{out_base.stem}.png" if args.theme == "light" else out_base.parent / "figure7_sft_replication.png"
        print(f"Generating Publication (Light) Figure 7...")
        plot_figure7(results, light_path, theme="light")

        # Also create standard figure7.png/pdf in figures/
        plot_figure7(results, out_base.parent / "figure7.png", theme="light")

    if args.theme in ["dark", "all"]:
        dark_path = out_base.parent / f"{out_base.stem}_dark.png" if args.theme == "dark" else out_base.parent / "figure7_sft_v5.png"
        print(f"Generating Dark-Theme Figure 7...")
        plot_figure7(results, dark_path, theme="dark")

    print("All Figure 7 replication figures successfully generated in ./figures/!")


if __name__ == "__main__":
    main()
