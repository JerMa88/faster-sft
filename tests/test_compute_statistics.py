"""
tests/test_compute_statistics.py
=================================
Unit tests for scripts/compute_statistics.py

Tests cover:
  - parse_run_identifier(): correctly identifies model, dataset, and loss variant.
  - compute_statistics_for_runs(): processes mock eval_results.json files,
    computes Wilson CIs, McNemar gap test, and McNemar vs baseline test.
  - format_latex_table(): outputs valid LaTeX code containing headers, metrics,
    and CIs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_statistics import (
    parse_run_identifier,
    compute_statistics_for_runs,
    format_latex_table,
)


class TestParseRunIdentifier:
    def test_qwen_stark_prime_baseline(self):
        meta = parse_run_identifier("qwen2.5-1.5b_stark_prime_baseline-lora")
        assert meta["model"] == "Qwen2.5-1.5B"
        assert meta["dataset"] == "STaRK-Prime"
        assert meta["variant"] == "Baseline-LoRA"

    def test_llama_stark_mag_hybrid_v2(self):
        meta = parse_run_identifier("llama-3.2-1b_stark_mag_hybrid-v2")
        assert meta["model"] == "Llama-3.2-1B"
        assert meta["dataset"] == "STaRK-MAG"
        assert meta["variant"] == "Hybrid-v2"

    def test_nanbeige_stark_prime_contraroute(self):
        meta = parse_run_identifier("nanbeige2-8b_stark_prime_contraroute-lora")
        assert meta["model"] == "Nanbeige2-8B"
        assert meta["dataset"] == "STaRK-Prime"
        assert meta["variant"] == "ContraRoute-LoRA"


class TestComputeStatisticsForRuns:
    def _create_mock_eval_file(self, run_dir: Path, filename: str, content: dict):
        d = run_dir / filename
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "eval_results.json", "w") as f:
            json.dump(content, f)

    def test_end_to_end_stats_computation(self, tmp_path):
        runs_dir = tmp_path / "runs"

        # Mock baseline run
        baseline_content = {
            "A_mem_strict_final": 0.99,
            "A_gen_strict_final": 0.10,
            "mem_correct_final": [1] * 99 + [0],
            "gen_correct_final": [1] * 10 + [0] * 90,
            "T_conv_strict": 30,
            "AUC_strict": 0.15,
        }
        self._create_mock_eval_file(
            runs_dir, "qwen2.5-1.5b_stark_prime_baseline-lora", baseline_content
        )

        # Mock alignment run
        hybrid_content = {
            "A_mem_strict_final": 0.99,
            "A_gen_strict_final": 0.40,
            "mem_correct_final": [1] * 99 + [0],
            "gen_correct_final": [1] * 40 + [0] * 60,
            "T_conv_strict": 15,
            "AUC_strict": 0.35,
        }
        self._create_mock_eval_file(
            runs_dir, "qwen2.5-1.5b_stark_prime_hybrid-v2", hybrid_content
        )

        structured, flat = compute_statistics_for_runs(runs_dir)

        assert len(flat) == 2
        grp_key = "Qwen2.5-1.5B / STaRK-Prime"
        assert grp_key in structured
        assert "Baseline-LoRA" in structured[grp_key]
        assert "Hybrid-v2" in structured[grp_key]

        hybrid_entry = structured[grp_key]["Hybrid-v2"]
        assert hybrid_entry["A_gen_strict"] == 0.40
        assert hybrid_entry["mcnemar_vs_baseline"] is not None
        assert hybrid_entry["mcnemar_vs_baseline"]["significant"] is True
        assert hybrid_entry["mcnemar_vs_baseline"]["p_value"] < 0.05

    def test_latex_formatter(self):
        mock_structured = {
            "Qwen2.5-1.5B / STaRK-Prime": {
                "Baseline-LoRA": {
                    "A_mem_strict": 0.99,
                    "A_mem_strict_ci": [0.97, 1.0],
                    "A_gen_strict": 0.10,
                    "A_gen_strict_ci": [0.05, 0.15],
                    "delta_A_strict": 0.89,
                    "T_conv_strict": 30,
                    "AUC_strict": 0.15,
                    "mcnemar_vs_baseline": None,
                },
                "Hybrid-v2": {
                    "A_mem_strict": 0.99,
                    "A_mem_strict_ci": [0.97, 1.0],
                    "A_gen_strict": 0.40,
                    "A_gen_strict_ci": [0.30, 0.50],
                    "delta_A_strict": 0.59,
                    "T_conv_strict": 15,
                    "AUC_strict": 0.35,
                    "mcnemar_vs_baseline": {"p_value": 0.001, "significant": True},
                },
            }
        }

        latex = format_latex_table(mock_structured)
        assert "\\begin{table*}" in latex
        assert "Qwen2.5-1.5B / STaRK-Prime" in latex
        assert "Baseline-LoRA" in latex
        assert "Hybrid-v2" in latex
        assert "^\\dagger" in latex  # Significant marker
        assert "\\end{table*}" in latex
