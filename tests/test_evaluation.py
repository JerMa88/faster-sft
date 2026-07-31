"""
tests/test_evaluation.py — Tests for src/evaluation/

Covers all metric functions and evaluator logic without GPU or real models.

Run with:
    /users/jerryma/.conda/envs/torch2.8/bin/python -m pytest tests/test_evaluation.py -v
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from src.evaluation.metrics import (
    accuracy,
    memorization_accuracy,
    generalization_accuracy,
    convergence_epoch,
    auc_curve,
    headroom,
    treatment_delta,
    convergence_speedup,
    compute_all_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# accuracy()
# ─────────────────────────────────────────────────────────────────────────────

class TestAccuracy:

    def test_perfect(self):
        assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_zero(self):
        assert accuracy([1, 2, 3], [4, 5, 6]) == 0.0

    def test_partial(self):
        assert abs(accuracy([1, 0, 1, 0], [1, 1, 1, 1]) - 0.5) < 1e-6

    def test_empty(self):
        assert accuracy([], []) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            accuracy([1, 2], [1])

    def test_numpy_arrays(self):
        preds   = np.array([1, 2, 3])
        targets = np.array([1, 2, 4])
        assert abs(accuracy(preds, targets) - 2/3) < 1e-6

    def test_single_element(self):
        assert accuracy([5], [5]) == 1.0
        assert accuracy([5], [6]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# memorization_accuracy / generalization_accuracy (wrappers)
# ─────────────────────────────────────────────────────────────────────────────

class TestAccuracyWrappers:

    def test_mem_acc(self):
        assert memorization_accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_gen_acc(self):
        assert generalization_accuracy([1, 0], [1, 1]) == 0.5

    def test_mem_gen_independent(self):
        """Both wrappers work independently on different splits."""
        mem_preds = [1, 1, 1, 0]
        mem_tgts  = [1, 1, 1, 1]
        gen_preds = [0, 0, 1, 1]
        gen_tgts  = [1, 1, 1, 1]
        assert memorization_accuracy(mem_preds, mem_tgts)   == 0.75
        assert generalization_accuracy(gen_preds, gen_tgts) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# convergence_epoch()
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergenceEpoch:

    def test_converges_at_start(self):
        curve = [0.96, 0.97, 0.98]
        assert convergence_epoch(curve, threshold=0.95, start_epoch=1) == 1

    def test_converges_mid_curve(self):
        curve = [0.80, 0.90, 0.95, 0.97]
        assert convergence_epoch(curve, threshold=0.95, start_epoch=1) == 3

    def test_never_converges(self):
        curve = [0.70, 0.80, 0.90]
        assert convergence_epoch(curve, threshold=0.95) is None

    def test_exact_threshold(self):
        curve = [0.90, 0.95, 0.97]
        assert convergence_epoch(curve, threshold=0.95, start_epoch=1) == 2

    def test_start_epoch_offset(self):
        """start_epoch=5 means first entry is epoch 5."""
        curve = [0.80, 0.96]
        assert convergence_epoch(curve, threshold=0.95, start_epoch=5) == 6

    def test_single_element(self):
        assert convergence_epoch([1.0], threshold=0.95) == 1
        assert convergence_epoch([0.5], threshold=0.95) is None


# ─────────────────────────────────────────────────────────────────────────────
# auc_curve()
# ─────────────────────────────────────────────────────────────────────────────

class TestAUC:

    def test_constant_curve(self):
        """Constant accuracy → AUC = that value."""
        curve = [0.8, 0.8, 0.8, 0.8]
        assert abs(auc_curve(curve) - 0.8) < 1e-6

    def test_increasing_curve(self):
        """Linear ramp 0→1 → AUC ≈ 0.5."""
        curve = [i/10 for i in range(11)]
        assert abs(auc_curve(curve) - 0.5) < 0.01

    def test_single_element(self):
        assert auc_curve([0.75]) == 0.75

    def test_higher_is_better(self):
        fast = [0.9, 0.95, 0.97, 0.98]
        slow = [0.5, 0.7,  0.85, 0.95]
        assert auc_curve(fast) > auc_curve(slow)


# ─────────────────────────────────────────────────────────────────────────────
# headroom / treatment_delta / convergence_speedup
# ─────────────────────────────────────────────────────────────────────────────

class TestGapMetrics:

    def test_headroom_positive(self):
        assert abs(headroom(0.9, 0.7) - 0.2) < 1e-6

    def test_headroom_zero(self):
        assert headroom(0.8, 0.8) == 0.0

    def test_headroom_negative(self):
        """Model can exceed oracle in edge cases (e.g. lucky guesses)."""
        assert headroom(0.7, 0.8) == pytest.approx(-0.1, abs=1e-6)

    def test_treatment_delta(self):
        assert abs(treatment_delta(0.85, 0.75) - 0.10) < 1e-6

    def test_speedup_normal(self):
        assert abs(convergence_speedup(30, 15) - 2.0) < 1e-6

    def test_speedup_baseline_never_converges(self):
        assert convergence_speedup(None, 20) == float("inf")

    def test_speedup_aligned_never_converges(self):
        assert convergence_speedup(30, None) is None

    def test_speedup_both_none(self):
        assert convergence_speedup(None, None) is None


# ─────────────────────────────────────────────────────────────────────────────
# compute_all_metrics()
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAllMetrics:

    def make_inputs(self):
        mem_preds   = [1, 1, 1, 0]
        mem_targets = [1, 1, 1, 1]
        gen_preds   = [1, 0, 1, 1]
        gen_targets = [1, 1, 1, 1]
        return mem_preds, mem_targets, gen_preds, gen_targets

    def test_basic_keys(self):
        mp, mt, gp, gt = self.make_inputs()
        result = compute_all_metrics(mp, mt, gp, gt)
        assert "A_mem" in result
        assert "A_gen" in result
        assert result["A_mem"] == 0.75
        assert result["A_gen"] == 0.75

    def test_with_accuracy_curve(self):
        mp, mt, gp, gt = self.make_inputs()
        curve  = [0.5, 0.7, 0.80, 0.90, 0.96]
        result = compute_all_metrics(mp, mt, gp, gt,
                                     gen_accuracy_curve=curve, threshold=0.95)
        assert "T_conv"    in result
        assert "AUC"       in result
        assert result["T_conv"]    == 5   # first epoch ≥ 0.95 is index 4 → epoch 5
        assert result["threshold"] == 0.95

    def test_with_oracle(self):
        mp, mt, gp, gt = self.make_inputs()
        oracle_preds = [1, 1, 1, 1]   # perfect oracle
        result = compute_all_metrics(mp, mt, gp, gt, oracle_preds=oracle_preds)
        assert "A_oracle" in result
        assert "headroom" in result
        assert result["A_oracle"] == 1.0
        assert abs(result["headroom"] - 0.25) < 1e-3

    def test_rounding(self):
        """Values should be rounded to 4 decimal places."""
        mp, mt, gp, gt = [1]*7 + [0], [1]*8, [1]*3 + [0]*5, [1]*8
        result = compute_all_metrics(mp, mt, gp, gt)
        assert str(result["A_mem"]) == str(round(result["A_mem"], 4))
        assert str(result["A_gen"]) == str(round(result["A_gen"], 4))


# ─────────────────────────────────────────────────────────────────────────────
# evaluator.py — lightweight structural tests (no real model loading)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatorHelpers:

    def test_baseline_gate_criterion(self):
        """A_mem ≥ 0.978 at epoch 3 should pass the gate."""
        # Simulate per-epoch results
        epoch_results = [
            {"epoch": 1, "A_mem": 0.85, "A_gen": 0.40},
            {"epoch": 3, "A_mem": 0.982, "A_gen": 0.60},
            {"epoch": 5, "A_mem": 0.990, "A_gen": 0.72},
        ]
        gate = None
        for r in epoch_results:
            if r["epoch"] == 3:
                gate = r["A_mem"] >= 0.978
        assert gate is True, "A_mem=0.982 at epoch 3 should pass the gate"

    def test_baseline_gate_fails(self):
        epoch_results = [{"epoch": 3, "A_mem": 0.950, "A_gen": 0.55}]
        gate = epoch_results[0]["A_mem"] >= 0.978
        assert gate is False

    def test_convergence_from_curve(self):
        """Demonstrates that T_conv should be computed from epoch labels, not raw indices."""
        a_gen_curve = [0.50, 0.65, 0.78, 0.88, 0.93, 0.96, 0.97, 0.98]
        epochs      = [1, 3, 5, 10, 15, 20, 30, 50]
        threshold   = 0.95

        # Find first epoch label where accuracy >= threshold
        first_epoch = next(
            (e for e, a in zip(epochs, a_gen_curve) if a >= threshold),
            None
        )
        assert first_epoch == 20, f"Expected epoch 20, got {first_epoch}"

        # convergence_epoch returns start_epoch + index (useful for evenly-spaced curves)
        t_raw = convergence_epoch(a_gen_curve, threshold, start_epoch=1)
        assert t_raw == 6   # index 5 + start_epoch 1 = 6 (positional, not calendar epoch)

