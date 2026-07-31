"""
Loss Variant 3: Contrastive Routing (ContraRoute)
===================================================
InfoNCE-style loss that pulls h_E^{l_t}(P_gen) toward the source-layer
representation of the *same* fact, while pushing it away from other facts
in the batch.

Formula (implementation_plan.md §Loss Variant 3):
    L_Contra = -(1/2) Σ_{l_s ∈ {l_s_early, l_s_late}} log(
        exp(sim(q, k+_{l_s}) / τ) /
        [ exp(sim(q, k+_{l_s}) / τ) + Σ_{j≠i} exp(sim(q, k-_{j,l_s}) / τ) ]
    )

where for fact i in the batch:
  q          = h_E^{l_t}(P_gen^{(i)})          — query (has grad)
  k+_{l_s}   = sg[h_E^{l_s}(P_mem^{(i)})]      — positive key (same fact, detached)
  k-_{j,l_s} = sg[h_E^{l_s}(P_mem^{(j)})]      — negative keys (other facts, detached)
  sim(u, v)  = u·v / (‖u‖ ‖v‖)               — cosine similarity
  τ = 0.07   — temperature

Key properties:
  - In-batch negatives are free (no extra forward passes)
  - All keys are stop-gradient; gradient only flows through q (P_gen path)
  - Batch size ≥ 8 recommended for sufficient negatives
"""

import torch
import torch.nn.functional as F

from .base import AlignmentLoss


def contrastive_loss(
    h_source: torch.Tensor,
    h_target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss with in-batch negatives.

    Args:
        h_source : (B, D) — source representations (detached internally)
        h_target : (B, D) — query representations (must have grad)
        temperature : τ, default 0.07

    Returns:
        Scalar loss ≥ 0. Lower is better (same-fact alignment).
        Minimum ~0 when each query collapses onto its positive key.
    """
    B = h_target.size(0)
    if B < 2:
        # Cannot form negatives; fall back to cosine distillation
        cos_sim = F.cosine_similarity(h_target.float(), h_source.detach().float(), dim=-1)
        return (1.0 - cos_sim).mean()

    # Normalise to unit sphere (cosine similarity = dot product after L2 norm)
    q = F.normalize(h_target.float(),         dim=-1)   # (B, D)  — has grad
    k = F.normalize(h_source.detach().float(), dim=-1)   # (B, D)  — stop-grad

    # Similarity matrix: logits[i, j] = sim(q_i, k_j)
    logits = torch.mm(q, k.T) / temperature              # (B, B)

    # Diagonal entries are the positives (same-fact pairs)
    labels = torch.arange(B, device=h_target.device)
    return F.cross_entropy(logits, labels)


class ContraRouteLoss(AlignmentLoss):
    """
    InfoNCE averaged over l_s_early and l_s_late with τ=0.07.
    """

    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def forward(
        self,
        h_mem_early: torch.Tensor,
        h_mem_late:  torch.Tensor,
        h_gen:       torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        loss  = contrastive_loss(h_mem_early, h_gen, self.temperature) * 0.5
        loss += contrastive_loss(h_mem_late,  h_gen, self.temperature) * 0.5
        return loss
