"""
src/losses/__init__.py — public API for all alignment loss variants.

Usage in train_sft.py:
    from src.losses import RepDistLoss, ContraRouteLoss, ProbeLoss, HybridLoss
    from src.losses import build_loss

The functional equivalents (used in src/training/losses.py) are also re-exported
for backward compatibility:
    from src.losses import rep_distill_loss, contrastive_loss
"""

from .rep_distill import RepDistLoss, rep_distill_loss
from .contrastive import ContraRouteLoss, contrastive_loss
from .probe_loss  import ProbeLoss
from .hybrid      import HybridLoss
from .base        import AlignmentLoss, AlignmentLossConfig

import torch
import torch.nn as nn


def build_loss(
    variant: str,
    probe: nn.Linear | None = None,
    alpha: float = 0.5,
    temperature: float = 0.07,
) -> AlignmentLoss:
    """
    Factory function — construct the correct AlignmentLoss for a given variant string.

    Args:
        variant     : one of "rep_distill", "contrastive", "probe", "hybrid"
        probe       : frozen nn.Linear for probe/hybrid (None = fallback to rep/contra)
        alpha       : Hybrid weighting (default 0.5)
        temperature : ContraRoute/Hybrid temperature τ (default 0.07)

    Returns:
        AlignmentLoss instance
    """
    variant = variant.lower()
    if variant == "rep_distill":
        return RepDistLoss()
    elif variant == "contrastive":
        return ContraRouteLoss(temperature=temperature)
    elif variant == "probe":
        if probe is None:
            import warnings
            warnings.warn(
                "build_loss('probe') called without a probe; "
                "falling back to RepDistLoss. Run pretrain_probe.py first.",
                UserWarning,
            )
            return RepDistLoss()
        probe_loss = ProbeLoss(probe)
        return probe_loss
    elif variant == "hybrid":
        probe_loss = ProbeLoss(probe) if probe is not None else None
        return HybridLoss(alpha=alpha, probe=probe_loss, temperature=temperature)
    elif variant == "baseline":
        raise ValueError("'baseline' uses no alignment loss; do not call build_loss.")
    else:
        raise ValueError(
            f"Unknown loss variant '{variant}'. "
            "Choose from: rep_distill, contrastive, probe, hybrid"
        )


__all__ = [
    "AlignmentLoss",
    "AlignmentLossConfig",
    "RepDistLoss",
    "ContraRouteLoss",
    "ProbeLoss",
    "HybridLoss",
    "build_loss",
    "rep_distill_loss",
    "contrastive_loss",
]
