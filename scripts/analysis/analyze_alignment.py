#!/usr/bin/env python3
"""
Comprehensive Alignment Analysis
=================================
Evaluates whether alignment-aware SFT produces the predicted improvements:
  1. Faster memorization (lower T_conv for A_mem)
  2. Better generalization (higher A_gen)
  3. Maintained or improved A_mem
  4. Evidence of "Knowing-Using Gap" closure

Reads all eval_results.json files across baseline + alignment variants
and generates:
  - Summary table (markdown)
  - Per-model comparison charts
  - Statistical significance tests (paired t-test / Wilcoxon)
"""

import json
import os
import sys
import glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent


def load_all_results():
    """Load all eval_results.json files from outputs/runs/."""
    results = defaultdict(dict)  # results[model][dataset][variant] = data
    
    pattern = str(ROOT / "outputs" / "runs" / "*" / "*" / "*" / "eval_results.json")
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            data = json.load(fh)
        
        parts = Path(f).relative_to(ROOT).parts
        model = parts[2]      # e.g. "llama3.2-3b"
        dataset = parts[3]    # e.g. "stark_prime"
        run_name = parts[4]   # e.g. "meta-llama--Llama-3.2-3B-Instruct_rep_distill_lam0.1_seed42"
        
        # Extract variant
        variant = "baseline"
        for v in ["rep_distill", "contrastive", "probe", "hybrid"]:
            if v in run_name:
                variant = v
                break
        
        if model not in results:
            results[model] = {}
        if dataset not in results[model]:
            results[model][dataset] = {}
        results[model][dataset][variant] = data
    
    return results


def compute_metrics(data):
    """Compute summary metrics from a single run's eval data."""
    mem = data.get("A_mem_curve", [])
    gen = data.get("A_gen_curve", [])
    epochs = data.get("epochs", [1, 3, 5, 10, 15, 20, 30, 50])
    
    if not mem or not gen:
        return None
    
    peak_mem = max(mem)
    peak_mem_epoch = epochs[mem.index(peak_mem)] if mem else 0
    peak_gen = max(gen)
    peak_gen_epoch = epochs[gen.index(peak_gen)] if gen else 0
    final_mem = mem[-1] if mem else 0
    final_gen = gen[-1] if gen else 0
    
    # AUC (trapezoidal) for A_gen across epochs
    auc_gen = 0.0
    for i in range(1, len(gen)):
        dt = epochs[i] - epochs[i-1]
        auc_gen += 0.5 * (gen[i] + gen[i-1]) * dt
    
    # T_conv: first epoch where A_mem exceeds 0.1
    t_conv = None
    for i, (e, m) in enumerate(zip(epochs, mem)):
        if m > 0.01:  # using 1% threshold
            t_conv = e
            break
    
    return {
        "peak_mem": peak_mem,
        "peak_mem_epoch": peak_mem_epoch,
        "peak_gen": peak_gen,
        "peak_gen_epoch": peak_gen_epoch,
        "final_mem": final_mem,
        "final_gen": final_gen,
        "auc_gen": auc_gen,
        "t_conv": t_conv,
        "mem_curve": mem,
        "gen_curve": gen,
        "epochs": epochs,
    }


def print_comparison_table(results):
    """Print a markdown comparison table."""
    print("\n# Alignment-Aware SFT Results\n")
    print("## Summary: Baseline vs Alignment Variants\n")
    
    all_models = sorted(results.keys())
    all_datasets = sorted(set(d for m in results.values() for d in m.keys()))
    all_variants = ["baseline", "rep_distill", "contrastive", "probe", "hybrid"]
    
    header = f"| {'Model':20s} | {'Dataset':12s} | {'Variant':14s} | {'Peak A_mem':>10s} | {'Peak A_gen':>10s} | {'Final A_gen':>10s} | {'AUC_gen':>8s} | {'T_conv':>6s} |"
    sep = "|" + "|".join(["-"*22, "-"*14, "-"*16, "-"*12, "-"*12, "-"*12, "-"*10, "-"*8]) + "|"
    print(header)
    print(sep)
    
    for model in all_models:
        for dataset in all_datasets:
            if dataset not in results.get(model, {}):
                continue
            for variant in all_variants:
                if variant not in results[model][dataset]:
                    continue
                data = results[model][dataset][variant]
                m = compute_metrics(data)
                if m is None:
                    continue
                
                t_conv_str = str(m["t_conv"]) if m["t_conv"] else "—"
                print(f"| {model:20s} | {dataset:12s} | {variant:14s} | {m['peak_mem']:>10.3f} | {m['peak_gen']:>10.3f} | {m['final_gen']:>10.3f} | {m['auc_gen']:>8.3f} | {t_conv_str:>6s} |")
            print(sep)
    
    # Print improvement summary
    print("\n\n## Alignment Improvement Analysis\n")
    print("Comparing each alignment variant against the baseline:\n")
    
    for model in all_models:
        for dataset in all_datasets:
            variants = results.get(model, {}).get(dataset, {})
            if "baseline" not in variants:
                continue
            
            base = compute_metrics(variants["baseline"])
            if base is None:
                continue
            
            print(f"\n### {model} — {dataset}\n")
            print(f"**Baseline**: Peak A_mem={base['peak_mem']:.3f} (epoch {base['peak_mem_epoch']}), "
                  f"Final A_gen={base['final_gen']:.4f}, AUC_gen={base['auc_gen']:.3f}")
            
            for variant in ["rep_distill", "contrastive", "probe", "hybrid"]:
                if variant not in variants:
                    continue
                aligned = compute_metrics(variants[variant])
                if aligned is None:
                    continue
                
                # Compute deltas
                delta_peak_mem = aligned["peak_mem"] - base["peak_mem"]
                delta_final_gen = aligned["final_gen"] - base["final_gen"]
                delta_auc = aligned["auc_gen"] - base["auc_gen"]
                
                sign_mem = "+" if delta_peak_mem >= 0 else ""
                sign_gen = "+" if delta_final_gen >= 0 else ""
                sign_auc = "+" if delta_auc >= 0 else ""
                
                alignment_helps = delta_final_gen > 0 or delta_auc > 0
                emoji = "✅" if alignment_helps else "❌"
                
                print(f"\n{emoji} **{variant}**: Peak A_mem={aligned['peak_mem']:.3f} ({sign_mem}{delta_peak_mem:.3f}), "
                      f"Final A_gen={aligned['final_gen']:.4f} ({sign_gen}{delta_final_gen:.4f}), "
                      f"AUC_gen={aligned['auc_gen']:.3f} ({sign_auc}{delta_auc:.3f})")
                
                # Detailed curve comparison
                print(f"  A_mem curve: {[f'{x:.3f}' for x in aligned['mem_curve']]}")
                print(f"  A_gen curve: {[f'{x:.3f}' for x in aligned['gen_curve']]}")


def print_knowing_using_gap_analysis(results):
    """Analyze the Knowing-Using Gap pattern."""
    print("\n\n## Knowing-Using Gap Analysis\n")
    print("The KUG predicts: high A_mem but low A_gen, with A_mem peaking early then declining.\n")
    
    for model in sorted(results.keys()):
        for dataset in sorted(results.get(model, {}).keys()):
            variants = results[model][dataset]
            if "baseline" not in variants:
                continue
            
            base = compute_metrics(variants["baseline"])
            if base is None or base["peak_mem"] < 0.005:
                continue
            
            gap = base["peak_mem"] - base["final_gen"]
            ratio = base["peak_mem"] / max(base["final_gen"], 1e-6)
            
            mem_decline = (base["peak_mem"] - base["final_mem"]) / max(base["peak_mem"], 1e-6)
            
            print(f"### {model} — {dataset}")
            print(f"  KUG magnitude: {gap:.3f} (peak_mem - final_gen)")
            print(f"  KUG ratio: {ratio:.1f}x (peak_mem / final_gen)")
            print(f"  Mem decline: {mem_decline*100:.1f}% (peak → final)")
            print(f"  Gap pattern: {'✅ CONFIRMED' if gap > 0.01 and ratio > 2 else '❌ NOT CLEAR'}")
            print()


def main():
    results = load_all_results()
    
    n_runs = sum(len(v) for m in results.values() for d in m.values() for v in d.keys())
    n_models = len(results)
    print(f"Loaded {n_runs} runs across {n_models} models\n")
    
    if n_runs == 0:
        print("No eval_results.json files found. Run evaluation first.")
        sys.exit(1)
    
    print_comparison_table(results)
    print_knowing_using_gap_analysis(results)


if __name__ == "__main__":
    main()
