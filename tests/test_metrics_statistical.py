"""
tests/test_metrics_statistical.py
==================================
Unit tests for Component 2 additions in src/evaluation/metrics.py:
  - strict_exact_match / strict_accuracy / strict_accuracy_with_indicators
  - wilson_ci / accuracy_with_wilson_ci
  - mcnemar_test
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from src.evaluation.metrics import (
    normalize_answer_strict,
    strict_exact_match,
    strict_accuracy,
    strict_accuracy_with_indicators,
    wilson_ci,
    accuracy_with_wilson_ci,
    mcnemar_test,
)


# ─────────────────────────────────────────────────────────────────────────────
# Strict Exact Match
# ─────────────────────────────────────────────────────────────────────────────

class TestStrictExactMatch:
    def test_exact_match_identical(self):
        assert strict_exact_match("BRCA1", "BRCA1") is True

    def test_case_insensitive(self):
        assert strict_exact_match("brca1", "BRCA1") is True
        assert strict_exact_match("  Brca1  ", "brca1") is True

    def test_strips_punctuation(self):
        assert strict_exact_match("BRCA1.", "BRCA1") is True
        assert strict_exact_match("BRCA1!", "brca1") is True
        assert strict_exact_match("TP53, gene", "TP53 gene") is True

    def test_rejects_substring_fallback(self):
        """Unlike string_exact_match, strict EM must NOT accept substrings."""
        assert strict_exact_match("The answer is BRCA1", "BRCA1") is False
        assert strict_exact_match("BRCA1 (breast cancer 1)", "BRCA1") is False
        assert strict_exact_match("1406577: BRCA1", "BRCA1") is False

    def test_strict_accuracy(self):
        preds = ["BRCA1", "TP53.", "EGFR (gene)", "MYC"]
        tgts  = ["brca1", "TP53",  "EGFR",        "MYC"]
        # Match: BRCA1 (yes), TP53. (yes), EGFR (gene) (no), MYC (yes) -> 3/4 = 0.75
        acc = strict_accuracy(preds, tgts)
        assert acc == 0.75

    def test_strict_accuracy_with_indicators(self):
        preds = ["BRCA1", "TP53.", "EGFR (gene)", "MYC"]
        tgts  = ["brca1", "TP53",  "EGFR",        "MYC"]
        acc, indicators = strict_accuracy_with_indicators(preds, tgts)
        assert acc == 0.75
        assert indicators == [1, 1, 0, 1]

    def test_empty_sequences(self):
        assert strict_accuracy([], []) == 0.0
        acc, ind = strict_accuracy_with_indicators([], [])
        assert acc == 0.0 and ind == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            strict_accuracy(["A"], ["A", "B"])


# ─────────────────────────────────────────────────────────────────────────────
# Wilson Score Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────

class TestWilsonCI:
    def test_known_values_50_percent(self):
        """k=500, n=1000 at 95% CI (z=1.96) -> ~ [0.469, 0.531]"""
        lo, hi = wilson_ci(500, 1000, z=1.96)
        assert pytest.approx(lo, abs=0.005) == 0.469
        assert pytest.approx(hi, abs=0.005) == 0.531

    def test_near_boundary_high(self):
        """k=998, n=1000 -> Wilson interval handles boundary gracefully without exceeding 1.0"""
        lo, hi = wilson_ci(998, 1000, z=1.96)
        assert lo < 0.998
        assert hi <= 1.0
        assert hi > 0.998

    def test_near_boundary_low(self):
        """k=2, n=1000 -> Wilson interval handles lower boundary without dropping below 0.0"""
        lo, hi = wilson_ci(2, 1000, z=1.96)
        assert lo >= 0.0
        assert lo < 0.002
        assert hi > 0.002

    def test_perfect_score(self):
        """k=100, n=100 -> upper bound clamped to 1.0"""
        lo, hi = wilson_ci(100, 100)
        assert hi == 1.0
        assert lo > 0.95

    def test_zero_score(self):
        """k=0, n=100 -> lower bound clamped to 0.0"""
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.05

    def test_zero_sample_size(self):
        lo, hi = wilson_ci(0, 0)
        assert (lo, hi) == (0.0, 1.0)

    def test_accuracy_with_wilson_ci(self):
        indicators = [1] * 80 + [0] * 20  # 80/100 = 0.80
        res = accuracy_with_wilson_ci(indicators)
        assert res["acc"] == 0.80
        assert res["k"] == 80
        assert res["n"] == 100
        assert res["ci_lo"] < 0.80
        assert res["ci_hi"] > 0.80


# ─────────────────────────────────────────────────────────────────────────────
# McNemar Test
# ─────────────────────────────────────────────────────────────────────────────

class TestMcNemarTest:
    def test_identical_outcomes(self):
        a = [1, 1, 0, 0]
        b = [1, 1, 0, 0]
        res = mcnemar_test(a, b)
        assert res["p_value"] == 1.0
        assert res["significant"] is False
        assert res["effect_size"] == 0.0
        assert res["n_01"] == 0 and res["n_10"] == 0

    def test_strong_disagreement(self):
        """A is almost always correct, B is almost always wrong (large gap)."""
        a = [1] * 100 + [0] * 5
        b = [0] * 90 + [1] * 15
        res = mcnemar_test(a, b)
        # indices 0..89: a=1, b=0 -> n_10 = 90 (A right, B wrong)
        # indices 90..99: a=1, b=1 -> n_11 = 10
        # indices 100..104: a=0, b=1 -> n_01 = 5
        assert res["n_10"] == 90
        assert res["n_01"] == 5
        assert res["p_value"] < 1e-10
        assert res["significant"] is True

    def test_symmetric_discordance(self):
        """Equal discordant pairs -> no systematic difference."""
        a = [1, 0, 1, 0] * 25
        b = [0, 1, 1, 0] * 25
        # n_10 = 25, n_01 = 25
        res = mcnemar_test(a, b)
        assert res["n_10"] == 25
        assert res["n_01"] == 25
        assert res["p_value"] > 0.8
        assert res["significant"] is False


    def test_empty_inputs(self):
        res = mcnemar_test([], [])
        assert res["p_value"] == 1.0
        assert res["significant"] is False

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            mcnemar_test([1, 0], [1])
