"""
Abstract base class for all alignment loss variants.
All losses share the same interface so train_sft.py can call them uniformly.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class AlignmentLossConfig:
    """Shared configuration for all alignment losses."""
    lambda_align:  float = 0.1    # overall weight λ applied in train loop
    warmup_epochs: int   = 3      # K: epochs before loss activates


class AlignmentLoss(ABC):
    """
    Abstract interface for all alignment loss variants.

    Each variant receives:
      h_mem_early : (B, D)  — entity-span rep from l_s_early on P_mem (stop-grad)
      h_mem_late  : (B, D)  — entity-span rep from l_s_late  on P_mem (stop-grad)
      h_gen       : (B, D)  — entity-span rep from l_t       on P_gen (has grad)
      **kwargs              — variant-specific extras (e.g. tgt_ids for ProbeLoss)

    Returns:
      loss : scalar Tensor with grad_fn on h_gen
    """

    @abstractmethod
    def forward(
        self,
        h_mem_early: torch.Tensor,
        h_mem_late:  torch.Tensor,
        h_gen:       torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)
