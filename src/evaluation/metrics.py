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


def normalize_answer_strict(s: str) -> str:
    """
    ACL-grade strict normalization:
    - Lowercase
    - Remove ALL punctuation
    - Collapse extra whitespace
    No prefix stripping or substring logic.
    """
    import re
    import string
    s = s.strip().lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def strict_exact_match(predicted: str, target: str) -> bool:
    """
    ACL-grade strict Exact Match (EM): Returns True only if the normalized
    prediction is identical to the normalized target string.
    No substring or prefix matching fallback.
    """
    return normalize_answer_strict(predicted) == normalize_answer_strict(target)


def strict_accuracy(predictions: list[str], targets: list[str]) -> float:
    """
    Fraction of predictions matching targets via strict_exact_match.
    Returns 0.0 if sequences are empty.
    """
    if len(predictions) == 0:
        return 0.0
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have same length, "
            f"got {len(predictions)} vs {len(targets)}"
        )
    matches = sum(1 for p, t in zip(predictions, targets) if strict_exact_match(p, t))
    return float(matches / len(predictions))


def strict_accuracy_with_indicators(
    predictions: list[str], targets: list[str]
) -> tuple[float, list[int]]:
    """
    Returns (accuracy, per_instance_binary_list) using strict_exact_match.
    Binary list is 1 for match, 0 for mismatch (needed for McNemar test).
    """
    if len(predictions) == 0:
        return 0.0, []
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have same length, "
            f"got {len(predictions)} vs {len(targets)}"
        )
    indicators = [int(strict_exact_match(p, t)) for p, t in zip(predictions, targets)]
    acc = float(sum(indicators) / len(indicators)) if indicators else 0.0
    return acc, indicators


def relaxed_exact_match(predicted: str, target: str) -> bool:
    """
    Primary metric for multi-word entity recall (following Mem2Gen / KUG paper).

    Credits the model if the normalized target appears ANYWHERE in the normalized
    prediction. This handles the common case where the model generates the correct
    paper title followed by trailing generation artifacts (e.g. newlines, extra
    sentences), which strict EM would unfairly penalize.

    UNIDIRECTIONAL (gold ⊆ pred only) — NOT bidirectional.
    The old broken string_exact_match did (gold in pred OR pred in gold), which
    inflated scores when a short prediction was a truncation of a long gold title.
    Here, only gold ⊆ pred is checked: a prediction shorter than the target is
    always a miss.
    """
    pred_norm = normalize_answer_strict(predicted)
    tgt_norm  = normalize_answer_strict(target)
    if not tgt_norm:
        return False
    return pred_norm == tgt_norm or tgt_norm in pred_norm


def relaxed_accuracy_with_indicators(
    predictions: list[str], targets: list[str]
) -> tuple[float, list[int]]:
    """
    Returns (accuracy, per_instance_binary_list) using relaxed_exact_match.
    Binary list is 1 for match, 0 for mismatch (needed for McNemar test).
    PRIMARY reporting metric for STARK-MAG/Prime datasets.
    """
    if len(predictions) == 0:
        return 0.0, []
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have same length, "
            f"got {len(predictions)} vs {len(targets)}"
        )
    indicators = [int(relaxed_exact_match(p, t)) for p, t in zip(predictions, targets)]
    acc = float(sum(indicators) / len(indicators)) if indicators else 0.0
    return acc, indicators



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
# Statistical Rigor Metrics (ACL Submission Standards)
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(
    k: int,
    n: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """
    Computes 95% Wilson score confidence interval for a proportion k / n.
    Preferred over normal approximation when proportion is near 0 or 1.
    Formula: (Wilson 1927; Agresti & Coull 1998).

    Args:
        k : number of successes (correct predictions)
        n : total number of samples
        z : z-score for confidence level (1.96 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple clamped to [0.0, 1.0].
    """
    import math
    if n <= 0:
        return (0.0, 1.0)
    k = max(0, min(k, n))
    if k == 0:
        p_hat = 0.0
        denominator = 1 + z**2 / n
        margin = (z / denominator) * math.sqrt(z**2 / (4 * n**2))
        return (0.0, min(1.0, (z**2 / (2 * n)) / denominator + margin))
    if k == n:
        p_hat = 1.0
        denominator = 1 + z**2 / n
        margin = (z / denominator) * math.sqrt(z**2 / (4 * n**2))
        centre = (1.0 + z**2 / (2 * n)) / denominator
        return (max(0.0, centre - margin), 1.0)

    p_hat = k / n
    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(
        max(0.0, p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    )
    return (max(0.0, min(1.0, centre - margin)), min(1.0, max(0.0, centre + margin)))



def accuracy_with_wilson_ci(
    correct_indicators: list[int] | Sequence[int],
    z: float = 1.96,
) -> dict[str, float | int]:
    """
    Given a sequence of 0/1 correctness indicators, returns accuracy and 95% Wilson CI.

    Returns:
        {
            "acc": float,
            "ci_lo": float,
            "ci_hi": float,
            "k": int,
            "n": int,
        }
    """
    n = len(correct_indicators)
    k = int(sum(correct_indicators)) if n > 0 else 0
    acc = float(k / n) if n > 0 else 0.0
    lo, hi = wilson_ci(k, n, z=z)
    return {
        "acc": acc,
        "ci_lo": lo,
        "ci_hi": hi,
        "k": k,
        "n": n,
    }


def mcnemar_test(
    binary_a: list[int] | Sequence[int],
    binary_b: list[int] | Sequence[int],
) -> dict[str, float | int | bool]:
    """
    Mid-p McNemar's test for paired binary outcomes (Dror et al., ACL 2018).

    Null hypothesis H0: P(A=1, B=0) == P(A=0, B=1)

    Use cases:
      - KUG Gap: A = mem_correct, B = gen_correct → test if mem != gen
      - Method comparison: A = gen_baseline, B = gen_aligned → test if alignment improves gen

    Returns:
        {
            "n_00": int,        # both wrong
            "n_01": int,        # A wrong, B right (B wins)
            "n_10": int,        # A right, B wrong (A wins)
            "n_11": int,        # both right
            "n_total": int,
            "statistic": float, # chi2 statistic with continuity correction
            "p_value": float,   # mid-p value from binomial test on discordant pairs
            "significant": bool,# p < 0.05
            "effect_size": float, # (n_01 - n_10) / n_total
        }
    """
    import math
    from scipy import stats

    if len(binary_a) != len(binary_b):
        raise ValueError(
            f"binary_a and binary_b must have same length, "
            f"got {len(binary_a)} vs {len(binary_b)}"
        )

    n_total = len(binary_a)
    if n_total == 0:
        return {
            "n_00": 0, "n_01": 0, "n_10": 0, "n_11": 0, "n_total": 0,
            "statistic": 0.0, "p_value": 1.0, "significant": False, "effect_size": 0.0
        }

    n_00 = sum(1 for a, b in zip(binary_a, binary_b) if a == 0 and b == 0)
    n_01 = sum(1 for a, b in zip(binary_a, binary_b) if a == 0 and b == 1)
    n_10 = sum(1 for a, b in zip(binary_a, binary_b) if a == 1 and b == 0)
    n_11 = sum(1 for a, b in zip(binary_a, binary_b) if a == 1 and b == 1)

    n_disc = n_01 + n_10

    if n_disc == 0:
        # Outcomes are identical across all instances
        return {
            "n_00": n_00, "n_01": n_01, "n_10": n_10, "n_11": n_11, "n_total": n_total,
            "statistic": 0.0, "p_value": 1.0, "significant": False, "effect_size": 0.0
        }

    # Chi-squared statistic with Edwards continuity correction
    b, c = n_01, n_10
    chi2_stat = ((abs(b - c) - 0.5) ** 2) / n_disc if n_disc > 0 else 0.0

    # Mid-p binomial test calculation
    # Mid-p = Exact binomial p-value minus 0.5 * P(X = min(b, c))
    k_min = min(b, c)
    exact_p = stats.binom.cdf(k_min, n_disc, 0.5) * 2
    exact_p = min(1.0, exact_p)
    prob_exact_k = stats.binom.pmf(k_min, n_disc, 0.5)
    mid_p = exact_p - prob_exact_k  # Subtract half the point probability for mid-p

    mid_p = max(0.0, min(1.0, float(mid_p)))
    effect_size = float((n_01 - n_10) / n_total)

    return {
        "n_00": n_00,
        "n_01": n_01,
        "n_10": n_10,
        "n_11": n_11,
        "n_total": n_total,
        "statistic": float(chi2_stat),
        "p_value": mid_p,
        "significant": mid_p < 0.05,
        "effect_size": effect_size,
    }



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
