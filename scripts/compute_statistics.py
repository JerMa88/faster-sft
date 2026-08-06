"""
scripts/compute_statistics.py
==============================
Post-processing statistical analysis script for ACL submission.

Walks run directories containing `eval_results.json`, computes statistical
rigor metrics, performs McNemar significance tests, and generates LaTeX tables.

Calculations performed:
1. 95% Wilson Score Confidence Intervals for A_mem and A_gen (strict & lenient).
2. McNemar's Test 1 (KUG Gap Significance):
   H0: P(mem=1, gen=0) == P(mem=0, gen=1) per run. Tests if memorization
   significantly exceeds generalization.
3. McNemar's Test 2 (Method Comparison):
   H0: P(gen_baseline=1, gen_aligned=0) == P(gen_baseline=0, gen_aligned=1).
   Paired comparison testing if alignment method significantly improves
   generalization over baseline.
4. Per-task-type breakdown (chaining vs. intersection).
5. LaTeX table generator for paper inclusion.

Usage:
    python scripts/compute_statistics.py \
        --runs_dir outputs/runs \
        --out_dir outputs/stats \
        --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    accuracy_with_wilson_ci,
    mcnemar_test,
    wilson_ci,
)


def find_eval_results(runs_dir: Path) -> List[Tuple[Path, dict]]:
    """Find all eval_results.json files in runs_dir."""
    results = []
    for path in runs_dir.rglob("eval_results.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append((path.parent, data))
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    return results


def parse_run_identifier(run_dir_name: str) -> Dict[str, str]:
    """
    Extract model, dataset, and loss_variant from run directory name.
    Example names:
      - qwen2.5-1.5b_stark_prime_baseline-lora
      - llama-3.2-1b_stark_mag_hybrid-v2
    """
    name = run_dir_name.lower()
    # Identify dataset
    if "stark_prime" in name or "prime" in name:
        dataset = "STaRK-Prime"
    elif "stark_mag" in name or "mag" in name:
        dataset = "STaRK-MAG"
    else:
        dataset = "Unknown-Dataset"

    # Identify model
    if "qwen" in name:
        model = "Qwen2.5-1.5B"
    elif "llama" in name:
        model = "Llama-3.2-1B"
    elif "nanbeige" in name:
        model = "Nanbeige2-8B"
    else:
        model = run_dir_name.split("_")[0]

    # Identify loss variant
    if "baseline" in name:
        variant = "Baseline-LoRA"
    elif "hybrid-v2" in name or "hybrid_v2" in name:
        variant = "Hybrid-v2"
    elif "hybrid" in name:
        variant = "Hybrid-v1"
    elif "repdist" in name:
        variant = "RepDist-LoRA"
    elif "contraroute" in name or "contra" in name:
        variant = "ContraRoute-LoRA"
    elif "probeloss" in name or "probe" in name:
        variant = "ProbeLoss-LoRA"
    elif "bridge" in name:
        variant = "BridgeAlign"
    else:
        variant = name.split("_")[-1]

    return {
        "model": model,
        "dataset": dataset,
        "variant": variant,
        "raw_name": run_dir_name,
    }


def compute_statistics_for_runs(
    runs_dir: Path, alpha: float = 0.05
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process all runs, compute CIs, McNemar tests, and per-task-type stats.
    """
    found_runs = find_eval_results(runs_dir)
    print(f"Found {len(found_runs)} evaluated runs in {runs_dir}")

    structured_results: Dict[str, Any] = {}
    flat_rows: List[Dict[str, Any]] = []

    # Map for pairing baseline and alignment runs: (model, dataset) -> baseline_run_data
    baselines: Dict[Tuple[str, str], dict] = {}

    # First pass: collect metadata and Wilson CIs
    for run_path, data in found_runs:
        meta = parse_run_identifier(run_path.name)
        model = meta["model"]
        dataset = meta["dataset"]
        variant = meta["variant"]

        # Ensure correctness vectors exist
        mem_correct = data.get("mem_correct_final", [])
        gen_correct = data.get("gen_correct_final", [])

        # Fallback if strict fields are not explicitly separated in older format
        a_mem_strict = data.get("A_mem_strict_final", data.get("A_mem_final", 0.0))
        a_gen_strict = data.get("A_gen_strict_final", data.get("A_gen_final", 0.0))

        a_mem_lenient = data.get("A_mem_final", a_mem_strict)
        a_gen_lenient = data.get("A_gen_final", a_gen_strict)

        # Wilson CIs
        mem_ci = (
            data.get("A_mem_strict_ci")
            or accuracy_with_wilson_ci(mem_correct)["ci_lo"],
            accuracy_with_wilson_ci(mem_correct)["ci_hi"],
        )
        if isinstance(data.get("A_mem_strict_ci"), list):
            mem_ci = tuple(data["A_mem_strict_ci"])

        gen_ci = (
            data.get("A_gen_strict_ci")
            or accuracy_with_wilson_ci(gen_correct)["ci_lo"],
            accuracy_with_wilson_ci(gen_correct)["ci_hi"],
        )
        if isinstance(data.get("A_gen_strict_ci"), list):
            gen_ci = tuple(data["A_gen_strict_ci"])

        # McNemar Test 1: KUG Gap Significance (A_mem vs A_gen)
        gap_mcnemar = {}
        if mem_correct and gen_correct:
            gap_mcnemar = mcnemar_test(mem_correct, gen_correct)

        row_entry = {
            "run_name": run_path.name,
            "model": model,
            "dataset": dataset,
            "variant": variant,
            "A_mem_strict": a_mem_strict,
            "A_mem_strict_ci": list(mem_ci),
            "A_gen_strict": a_gen_strict,
            "A_gen_strict_ci": list(gen_ci),
            "A_mem_lenient": a_mem_lenient,
            "A_gen_lenient": a_gen_lenient,
            "delta_A_strict": round(a_mem_strict - a_gen_strict, 4),
            "T_conv_strict": data.get("T_conv_strict", data.get("T_conv")),
            "AUC_strict": data.get("AUC_strict", data.get("AUC")),
            "mcnemar_kug_gap": gap_mcnemar,
            "mcnemar_vs_baseline": None,  # Computed in second pass
            "task_types": data.get("task_types", {}),
            "mem_correct": mem_correct,
            "gen_correct": gen_correct,
        }

        group_key = f"{model} / {dataset}"
        if group_key not in structured_results:
            structured_results[group_key] = {}
        structured_results[group_key][variant] = row_entry
        flat_rows.append(row_entry)

        if "baseline" in variant.lower():
            baselines[(model, dataset)] = row_entry

    # Second pass: McNemar Test 2 (Alignment vs Baseline generalization)
    for row in flat_rows:
        model = row["model"]
        dataset = row["dataset"]
        variant = row["variant"]

        if "baseline" not in variant.lower() and (model, dataset) in baselines:
            base_row = baselines[(model, dataset)]
            base_gen = base_row.get("gen_correct", [])
            align_gen = row.get("gen_correct", [])

            if base_gen and align_gen and len(base_gen) == len(align_gen):
                vs_base = mcnemar_test(base_gen, align_gen)
                row["mcnemar_vs_baseline"] = vs_base
                structured_results[f"{model} / {dataset}"][variant][
                    "mcnemar_vs_baseline"
                ] = vs_base

    return structured_results, flat_rows


def format_latex_table(structured_results: Dict[str, Any]) -> str:
    """Generate a clean LaTeX tabular block for the ACL submission paper."""
    lines = []
    lines.append("% Auto-generated ACL LaTeX table from compute_statistics.py")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llcccccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Model} & \\textbf{Method} & \\textbf{$A_{\\text{mem}}$ (95\\% CI)} & \\textbf{$A_{\\text{gen}}$ (95\\% CI)} & \\textbf{$\\Delta A$} & \\textbf{$T_{\\text{conv}}$} & \\textbf{AUC} & \\textbf{$p_{\\text{vs base}}$} \\\\"
    )
    lines.append("\\midrule")

    for group, variants in sorted(structured_results.items()):
        lines.append(f"% --- {group} ---")
        first = True
        for var_name, data in sorted(variants.items()):
            model_str = f"\\multirow{{{len(variants)}}}{{*}}{{{group}}}" if first else ""
            first = False

            a_mem = data["A_mem_strict"]
            mem_ci = data["A_mem_strict_ci"]
            mem_str = f"{a_mem:.3f}$_{{[ {mem_ci[0]:.3f}, {mem_ci[1]:.3f} ]}}$"

            a_gen = data["A_gen_strict"]
            gen_ci = data["A_gen_strict_ci"]

            mcn = data.get("mcnemar_vs_baseline")
            sig_marker = ""
            p_str = "--"
            if mcn and "p_value" in mcn:
                p_val = mcn["p_value"]
                p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
                if mcn.get("significant"):
                    sig_marker = "^\\dagger"

            gen_str = f"{a_gen:.3f}{sig_marker}$_{{[ {gen_ci[0]:.3f}, {gen_ci[1]:.3f} ]}}$"

            delta_a = data["delta_A_strict"]
            t_conv = data["T_conv_strict"] if data["T_conv_strict"] is not None else "--"
            auc = f"{data['AUC_strict']:.3f}" if data["AUC_strict"] is not None else "--"

            lines.append(
                f"{model_str} & {var_name} & {mem_str} & {gen_str} & {delta_a:.3f} & {t_conv} & {auc} & {p_str} \\\\"
            )
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Knowing-Using Gap ($A_{\\text{mem}}$ vs. $A_{\\text{gen}}$) across fine-tuning methods. Values report strict exact match with 95\\% Wilson score confidence intervals. $^\\dagger$ denotes $p < 0.05$ via mid-p McNemar's test vs. Baseline-LoRA.}"
    )
    lines.append("\\label{tab:kug_results}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistical rigor metrics for ACL submission."
    )
    parser.add_argument(
        "--runs_dir",
        default=str(ROOT / "outputs" / "runs"),
        help="Directory containing run subdirectories with eval_results.json",
    )
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "outputs" / "stats"),
        help="Output directory for JSON summary and LaTeX tables",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance threshold"
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"[ERROR] runs_dir does not exist: {runs_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  compute_statistics.py — ACL Statistical Analysis")
    print(f"  runs_dir = {runs_dir}")
    print(f"  out_dir  = {out_dir}")
    print("=" * 60)

    structured_results, flat_rows = compute_statistics_for_runs(
        runs_dir, alpha=args.alpha
    )

    # Remove non-serializable raw lists before saving JSON summary
    clean_structured = {}
    for grp, vars_dict in structured_results.items():
        clean_structured[grp] = {}
        for vname, vdata in vars_dict.items():
            clean_item = {k: v for k, v in vdata.items() if k not in ("mem_correct", "gen_correct")}
            if "task_types" in clean_item and isinstance(clean_item["task_types"], dict):
                clean_tt = {}
                for tt_name, tt_data in clean_item["task_types"].items():
                    clean_tt[tt_name] = {k: v for k, v in tt_data.items() if k not in ("mem_correct", "gen_correct")}
                clean_item["task_types"] = clean_tt
            clean_structured[grp][vname] = clean_item

    # Save JSON summary
    json_path = out_dir / "statistics_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_structured, f, indent=2)
    print(f"\n[OK] Statistical summary saved to {json_path}")

    # Save LaTeX table
    latex_table = format_latex_table(structured_results)
    tex_path = out_dir / "paper_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"[OK] LaTeX paper table saved to {tex_path}")

    print("\nSample LaTeX Table Output:")
    print("-" * 60)
    print(latex_table)
    print("-" * 60)


if __name__ == "__main__":
    main()
