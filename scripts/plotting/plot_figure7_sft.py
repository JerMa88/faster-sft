"""
Plot Figure 7 — SFT Training Dynamics (Memorization vs. Generalization)
=======================================================================
Parses the fast_eval log file and generates a 3-panel plot matching
Figure 7 of Dai et al. (2025):
  - Panel 1: Chaining    (A_mem high, A_gen <5%)
  - Panel 2: Intersection (A_mem high, A_gen >90% in paper — shows our deviation)
  - Panel 3: Fact-checking (A_mem high, A_gen ~50% in paper)

Usage:
  python scripts/plotting/plot_figure7_sft.py \
      --log outputs/logs/fast_eval_474756.out \
      --out outputs/figures/figure7_sft_baseline.png
"""

import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


PAPER_TARGETS = {
    # Paper Figure 7 expected asymptotic values (approximate, from paper text)
    "chaining":     {"mem": 0.95, "gen": 0.03},
    "intersection": {"mem": 0.95, "gen": 0.90},
    "fact_checking":{"mem": 0.95, "gen": 0.50},
}

TASK_LABELS = {
    "chaining":     "Chaining",
    "intersection": "Intersection",
    "fact_checking":"Fact Checking",
}

COLORS = {
    "mem": "#2563EB",   # blue
    "gen": "#DC2626",   # red
}


def parse_eval_log(log_path: str) -> dict:
    """Parse fast_eval log into per-task per-epoch accuracy arrays."""
    results = {
        task: {"epoch": [], "mem": [], "gen": []}
        for task in ["chaining", "intersection", "fact_checking"]
    }

    current_epoch = None
    epoch_data = {}

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Evaluating Epoch (\d+)\.\.\.", line)
            if m:
                if current_epoch is not None and epoch_data:
                    # Commit previous epoch
                    for task in results:
                        mem_key = f"eval/acc_mem_{task}"
                        gen_key = f"eval/acc_gen_{task}"
                        if mem_key in epoch_data:
                            results[task]["epoch"].append(current_epoch)
                            results[task]["mem"].append(epoch_data[mem_key])
                            results[task]["gen"].append(epoch_data[gen_key])
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
            if mem_key in epoch_data:
                results[task]["epoch"].append(current_epoch)
                results[task]["mem"].append(epoch_data[mem_key])
                results[task]["gen"].append(epoch_data[gen_key])

    return results


def plot_figure7(results: dict, out_path: str, method_name: str = "SFT Baseline (Ours)"):
    """Generate 3-panel Figure 7 matching the paper's style."""

    tasks = ["chaining", "intersection", "fact_checking"]
    n_completed = max(len(results[t]["epoch"]) for t in tasks)
    print(f"  Plotting {n_completed} epochs across 3 tasks...")

    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor('#0F1117')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    for col, task in enumerate(tasks):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('#1A1D27')

        epochs = results[task]["epoch"]
        mem    = results[task]["mem"]
        gen    = results[task]["gen"]

        if not epochs:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    color='white', transform=ax.transAxes)
            continue

        ep = np.array(epochs)
        m  = np.array(mem)
        g  = np.array(gen)

        # Shade decoupling region (A_mem > A_gen)
        ax.fill_between(ep, g, m, alpha=0.08, color='#2563EB',
                         label='_nolegend_')

        # Plot lines
        ax.plot(ep, m, color=COLORS["mem"], linewidth=2.5, label='$A_{mem}$',
                marker='o', markersize=3, markevery=max(1, len(ep)//10))
        ax.plot(ep, g, color=COLORS["gen"], linewidth=2.5, label='$A_{gen}$',
                marker='s', markersize=3, markevery=max(1, len(ep)//10))

        # Paper target dashed lines
        pt = PAPER_TARGETS[task]
        ax.axhline(pt["mem"], color=COLORS["mem"], linestyle='--', linewidth=1,
                   alpha=0.5, label=f'Paper $A_{{mem}}$ target ({pt["mem"]:.0%})')
        ax.axhline(pt["gen"], color=COLORS["gen"], linestyle='--', linewidth=1,
                   alpha=0.5, label=f'Paper $A_{{gen}}$ target ({pt["gen"]:.0%})')

        # Final values annotation
        if len(ep) > 0:
            final_ep = ep[-1]
            ax.annotate(f'{m[-1]:.1%}', xy=(final_ep, m[-1]),
                        xytext=(final_ep - 3, m[-1] + 0.03),
                        color=COLORS["mem"], fontsize=8, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=COLORS["mem"], lw=0.8))
            ax.annotate(f'{g[-1]:.1%}', xy=(final_ep, g[-1]),
                        xytext=(final_ep - 3, max(0, g[-1] - 0.08)),
                        color=COLORS["gen"], fontsize=8, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=COLORS["gen"], lw=0.8))

        # Threshold line at 90% for A_mem target
        ax.axhline(0.90, color='white', linestyle=':', linewidth=0.7, alpha=0.3)

        ax.set_xlim(0, max(ep) + 1)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Epoch', color='white', fontsize=11)
        if col == 0:
            ax.set_ylabel('Accuracy', color='white', fontsize=11)
        ax.set_title(f'{TASK_LABELS[task]}', color='white', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.3,
                  labelcolor='white', facecolor='#2A2D3A', edgecolor='#444')

        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#3A3D4A')

        # Print summary to terminal
        last_mem = m[-1] if len(m) else 0
        last_gen = g[-1] if len(g) else 0
        ep10_mem = m[min(9, len(m)-1)] if len(m) > 0 else 0
        print(f"  {TASK_LABELS[task]:15s}: ep10 A_mem={ep10_mem:.1%}  "
              f"final A_mem={last_mem:.1%}  final A_gen={last_gen:.1%}")

    # Super title
    fig.suptitle(
        f'Figure 7 Replication — {method_name}\n'
        f'Memorization ($A_{{mem}}$) vs. Generalization ($A_{{gen}}$) by Task',
        color='white', fontsize=14, fontweight='bold', y=1.02
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved -> {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 7 SFT Training Dynamics")
    parser.add_argument("--log", type=str, required=True, help="Path to fast_eval .out log")
    parser.add_argument("--out", type=str,
                        default="outputs/figures/figure7_sft_baseline.png")
    parser.add_argument("--method", type=str, default="SFT Baseline (Ours)")
    args = parser.parse_args()

    print(f"Parsing eval log: {args.log}")
    results = parse_eval_log(args.log)

    total_epochs = max(len(v["epoch"]) for v in results.values())
    print(f"Found data for {total_epochs} epochs")

    if total_epochs == 0:
        print("ERROR: No epoch data found in log file.")
        return

    plot_figure7(results, args.out, method_name=args.method)
    print("Done.")


if __name__ == "__main__":
    main()
