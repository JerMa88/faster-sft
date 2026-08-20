"""
Knowledge-Circuit Routing (KCR) Loss Module
Implements the causal layer-pair entity routing derived by Dai et al. (2025/2026).
Aligns mid-layer reasoning activations on P_gen with early & late storage activations on P_mem
specifically at the Head-Entity token span.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeCircuitRoutingLoss(nn.Module):
    """
    Computes cosine distance alignment between policy reasoning layer (ltgt ~ 0.5L)
    and reference storage layers (lsrc ~ 0.1L and 0.8L) at the head-entity position.
    """

    def __init__(
        self,
        target_layer_ratio: float = 0.50,
        source_layer_ratios: Tuple[float, ...] = (0.10, 0.80),
    ):
        super().__init__()
        self.target_layer_ratio = target_layer_ratio
        self.source_layer_ratios = source_layer_ratios

    def forward(
        self,
        policy_hidden_states: Tuple[torch.Tensor, ...],
        ref_hidden_states: Tuple[torch.Tensor, ...],
        gen_entity_masks: torch.Tensor,  # (B, seq_len_gen) normalized mask
        mem_entity_masks: torch.Tensor,  # (B, seq_len_mem) normalized mask
    ) -> torch.Tensor:
        """
        Args:
            policy_hidden_states: Tuple of (B, seq_len_gen, D) for each layer 0..L
            ref_hidden_states: Tuple of (B, seq_len_mem, D) for each layer 0..L
            gen_entity_masks: (B, seq_len_gen) weights for head entity in P_gen
            mem_entity_masks: (B, seq_len_mem) weights for head entity in P_mem
        """
        num_layers = len(policy_hidden_states) - 1  # L (excluding embedding layer 0 if present)
        if num_layers <= 0:
            return torch.tensor(0.0, device=policy_hidden_states[0].device)

        # 1. Determine target and source layer indices
        tgt_idx = max(1, min(num_layers, int(round(self.target_layer_ratio * num_layers))))
        src_indices = [
            max(1, min(num_layers, int(round(ratio * num_layers))))
            for ratio in self.source_layer_ratios
        ]

        # 2. Extract policy target representation at head-entity position: (B, D)
        h_gen_tgt = policy_hidden_states[tgt_idx]  # (B, seq_len_gen, D)
        # Weighted mean pool over entity tokens
        # gen_entity_masks: (B, seq_len_gen, 1)
        h_gen_entity = torch.sum(h_gen_tgt * gen_entity_masks.unsqueeze(-1), dim=1)  # (B, D)

        # 3. For each source layer in ref model, extract entity representation with stop-gradient
        total_kcr_loss = torch.tensor(0.0, device=h_gen_entity.device, dtype=h_gen_entity.dtype)

        for src_idx in src_indices:
            h_mem_src = ref_hidden_states[src_idx].detach()  # (B, seq_len_mem, D)
            h_mem_entity = torch.sum(h_mem_src * mem_entity_masks.unsqueeze(-1), dim=1).detach()  # (B, D)

            # Cosine distance: 1 - cosine_similarity(h_gen, sg[h_mem])
            cos_sim = F.cosine_similarity(h_gen_entity, h_mem_entity, dim=-1)  # (B,)
            kcr_step = torch.mean(1.0 - cos_sim)
            total_kcr_loss = total_kcr_loss + kcr_step

        return total_kcr_loss / max(1, len(src_indices))
