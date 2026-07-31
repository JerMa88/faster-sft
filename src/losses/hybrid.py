"""
Loss Variant 4: Hybrid (RepDist + ProbeLoss)
=============================================
Weighted combination that enforces both:
  - **Directional alignment** (RepDist): push h_gen toward the memorized direction
  - **Linear decodability** (ProbeLoss): push h_gen into the probe-decodable subspace

These are complementary: a representation can be directionally aligned with the
source but still not linearly decodable if the probe weight matrix doesn't
match the aligned direction, and vice versa.

Formula (implementation_plan.md §Loss Variant 4):
    L_Hybrid = α · L_RepDist + (1 - α) · L_Probe

where α = 0.5 by default (ablated in Phase 7 over {0.3, 0.5, 0.7}).

ProbeLoss fallback: if φ* is not available, falls back to ContraRoute
(same as train_sft.py's runtime behavior).
"""

import torch

from .base import AlignmentLoss
from .rep_distill import RepDistLoss
from .probe_loss import ProbeLoss
from .contrastive import ContraRouteLoss


class HybridLoss(AlignmentLoss):
    """
    α·RepDist + (1-α)·ProbeLoss, averaged over both source layers.

    Args:
        alpha : float — weight for RepDist term (0.5 by default)
        probe : ProbeLoss or None — if None, substitutes ContraRouteLoss
    """

    def __init__(
        self,
        alpha: float = 0.5,
        probe: ProbeLoss | None = None,
        temperature: float = 0.07,
    ):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha       = alpha
        self._rep_dist   = RepDistLoss()
        self._probe      = probe
        self._contra     = ContraRouteLoss(temperature=temperature)

    def forward(
        self,
        h_mem_early: torch.Tensor,
        h_mem_late:  torch.Tensor,
        h_gen:       torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        rd_loss = self._rep_dist(h_mem_early, h_mem_late, h_gen)

        if self._probe is not None:
            pl_loss = self._probe(h_mem_early, h_mem_late, h_gen, **kwargs)
        else:
            # Fallback: ContraRoute (no φ* needed)
            pl_loss = self._contra(h_mem_early, h_mem_late, h_gen)

        return self.alpha * rd_loss + (1.0 - self.alpha) * pl_loss
