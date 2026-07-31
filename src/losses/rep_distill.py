"""
Loss Variant 1: Representation Distillation (RepDist)
=======================================================
Directly operationalizes self-patching as a differentiable cosine loss.

Formula (implementation_plan.md §Loss Variant 1):
    L_RepDist = (1/2) Σ_{l_s ∈ {l_s_early, l_s_late}} (
        1 - cos(h_E^{l_t}(P_gen), sg[h_E^{l_s}(P_mem)])
    )

where:
  - cos(u, v) = u·v / (‖u‖ ‖v‖)   (cosine similarity)
  - sg[·] = stop-gradient (source representations are detached)
  - Gradient flows only through h_gen (the P_gen mid-layer representation)

Rationale (PoLM Part 3.1): knowledge is stored as a *direction* in
entity embedding space; cosine distance captures directional alignment
without sensitivity to norm differences across layers.
"""

import torch
import torch.nn.functional as F

from .base import AlignmentLoss


def rep_distill_loss(
    h_source: torch.Tensor,
    h_target: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine-distance loss between h_target and a stop-gradient copy of h_source.

    Args:
        h_source: (B, D) — source representation (will be detached)
        h_target: (B, D) — target representation (must have grad)

    Returns:
        Scalar loss in [0, 1]. Loss = 0 iff vectors are identical in direction.
    """
    h_src = h_source.detach().float()
    h_tgt = h_target.float()
    cos_sim = F.cosine_similarity(h_tgt, h_src, dim=-1)   # (B,)
    return (1.0 - cos_sim).mean()


class RepDistLoss(AlignmentLoss):
    """
    Averaged over both source layers: l_s_early and l_s_late.
    Each contributes weight 0.5, matching the formula above.
    """

    def forward(
        self,
        h_mem_early: torch.Tensor,
        h_mem_late:  torch.Tensor,
        h_gen:       torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        loss  = rep_distill_loss(h_mem_early, h_gen) * 0.5
        loss += rep_distill_loss(h_mem_late,  h_gen) * 0.5
        return loss
