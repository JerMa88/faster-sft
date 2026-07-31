"""
src/evaluation/__init__.py — public API for evaluation modules.
"""

from .metrics import (
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

__all__ = [
    "accuracy",
    "memorization_accuracy",
    "generalization_accuracy",
    "convergence_epoch",
    "auc_curve",
    "headroom",
    "treatment_delta",
    "convergence_speedup",
    "compute_all_metrics",
]
