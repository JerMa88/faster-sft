"""
src/evaluation/metrics.py
==========================
Pure, stateless metric functions used by the evaluator and analysis scripts.

All metrics defined in implementation_plan.md §Evaluation:
  - A_mem   : memorization accuracy  (P_mem → first answer token)
  - A_gen   : generalization accuracy (P_gen → first answer token)
  - T_conv  : convergence epoch (first epoch ≥ threshold)
  - headroom: A_oracle − A_gen (gap alignment is trying to close)
  - delta   : A_gen_aligned − A_gen_baseline (the treatment effect)
  - speedup : T_conv_baseline / T_conv_aligned (convergence acceleration)

All functions accept plain Python scalars, lists, or numpy arrays.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# String-level exact match (fixes BPE token ID mismatch)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Normalize an answer string: strip, lowercase, collapse whitespace."""
    import re
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    # Remove trailing punctuation that models sometimes add
    s = re.sub(r'[.\n]+$', '', s)
    # Strip markdown bold markers
    s = s.replace('**', '')
    # Strip common conversational prefixes
    for prefix in [
        'this paper is about ',
        'the answer is ',
        'answer: ',
    ]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip()


def string_exact_match(predicted: str, target: str) -> bool:
    """
    Case-insensitive, whitespace-normalized exact match.
    Handles the BPE token mismatch where tokenizer("small") != tokenizer(" small").
    Also checks if the target appears as a prefix of the prediction (the model
    may generate extra tokens after the entity name), or if the target is
    contained as a substring (for models that prepend IDs or conversational text).
    """
    pred_norm = normalize_answer(predicted)
    tgt_norm  = normalize_answer(target)
    # Direct match or prediction starts with target
    if pred_norm == tgt_norm or pred_norm.startswith(tgt_norm):
        return True
    # Target contained in prediction (for predictions like "1406577: entity name...")
    if tgt_norm in pred_norm:
        return True
    return False


def string_accuracy(predictions: list[str], targets: list[str]) -> float:
    """
    Fraction of predictions matching targets via string_exact_match.
    Returns 0.0 if sequences are empty.
    """
    if len(predictions) == 0:
        return 0.0
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have same length, "
            f"got {len(predictions)} vs {len(targets)}"
        )
    matches = sum(1 for p, t in zip(predictions, targets)
                  if string_exact_match(p, t))
    return float(matches / len(predictions))


# ─────────────────────────────────────────────────────────────────────────────
# Core accuracy metrics (token-level — kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """
    Fraction of predictions matching targets (exact token match).
    Returns 0.0 if sequences are empty.
    """
    if len(predictions) == 0:
        return 0.0
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have same length, "
            f"got {len(predictions)} vs {len(targets)}"
        )
    preds  = np.asarray(predictions, dtype=np.int64)
    tgts   = np.asarray(targets,     dtype=np.int64)
    return float((preds == tgts).mean())


def memorization_accuracy(
    predictions: Sequence[int],
    targets:     Sequence[int],
) -> float:
    """
    A_mem: accuracy on P_mem prompts.
    The model should reconstruct the entity token when given the memorization
    prompt (the fact itself). Ideally A_mem ≥ 0.978 (baseline gate criterion).
    """
    return accuracy(predictions, targets)


def generalization_accuracy(
    predictions: Sequence[int],
    targets:     Sequence[int],
) -> float:
    """
    A_gen: accuracy on P_gen prompts.
    The model must answer a multi-hop question that requires combining the
    memorized fact with other knowledge. This is the primary outcome metric.
    """
    return accuracy(predictions, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence
# ─────────────────────────────────────────────────────────────────────────────

def convergence_epoch(
    accuracy_curve: Sequence[float],
    threshold:      float = 0.95,
    start_epoch:    int   = 1,
) -> int | None:
    """
    T_conv: first epoch (1-indexed) where accuracy ≥ threshold.

    Args:
        accuracy_curve : sequence of per-epoch accuracy values
        threshold      : convergence threshold (default 0.95)
        start_epoch    : epoch number of the first entry (default 1)

    Returns:
        First epoch index where acc ≥ threshold, or None if never reached.
    """
    for i, acc in enumerate(accuracy_curve):
        if acc >= threshold:
            return start_epoch + i
    return None


def auc_curve(accuracy_curve: Sequence[float]) -> float:
    """
    Area under the accuracy-vs-epoch curve (trapezoidal rule), normalised by
    the number of epochs. Higher is better (faster convergence).
    """
    if len(accuracy_curve) < 2:
        return float(accuracy_curve[0]) if accuracy_curve else 0.0
    return float(np.trapz(accuracy_curve) / (len(accuracy_curve) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Gap / effect size metrics
# ─────────────────────────────────────────────────────────────────────────────

def headroom(a_oracle: float, a_gen: float) -> float:
    """
    Δ_headroom = A_oracle − A_gen.
    The gap between what the oracle patch achieves and what the trained model
    achieves without patching. A positive headroom means the alignment loss
    has room to close the gap.
    """
    return float(a_oracle - a_gen)


def treatment_delta(a_aligned: float, a_baseline: float) -> float:
    """
    Δ = A_gen_aligned − A_gen_baseline.
    The lift from adding the alignment loss. Should be ≥ 0 for the method
    to be beneficial.
    """
    return float(a_aligned - a_baseline)


def convergence_speedup(
    t_conv_baseline: int | None,
    t_conv_aligned:  int | None,
) -> float | None:
    """
    Speedup = T_conv_baseline / T_conv_aligned.
    Returns None if either value is None (convergence not reached).
    Returns float('inf') if aligned converges but baseline never does.
    """
    if t_conv_aligned is None:
        return None
    if t_conv_baseline is None:
        return float("inf")
    return float(t_conv_baseline) / float(t_conv_aligned)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summary
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    mem_preds:   Sequence[int],
    mem_targets: Sequence[int],
    gen_preds:   Sequence[int],
    gen_targets: Sequence[int],
    gen_accuracy_curve: Sequence[float] | None = None,
    oracle_preds: Sequence[int] | None = None,
    threshold:   float = 0.95,
) -> dict:
    """
    Compute the full metric bundle for a single run.

    Returns a dict with keys:
      A_mem, A_gen, T_conv (or None), AUC, headroom (if oracle_preds given)
    """
    a_mem = memorization_accuracy(mem_preds, mem_targets)
    a_gen = generalization_accuracy(gen_preds, gen_targets)

    result = {
        "A_mem": round(a_mem, 4),
        "A_gen": round(a_gen, 4),
    }

    if gen_accuracy_curve is not None:
        t = convergence_epoch(gen_accuracy_curve, threshold)
        result["T_conv"]    = t
        result["AUC"]       = round(auc_curve(gen_accuracy_curve), 4)
        result["threshold"] = threshold

    if oracle_preds is not None:
        a_oracle = generalization_accuracy(oracle_preds, gen_targets)
        result["A_oracle"]  = round(a_oracle, 4)
        result["headroom"]  = round(headroom(a_oracle, a_gen), 4)

    return result
