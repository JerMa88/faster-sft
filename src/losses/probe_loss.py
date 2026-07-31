"""
Loss Variant 2: Probing Loss (ProbeLoss)
==========================================
Uses a pretrained, frozen linear probe φ* to push h_E^{l_t}(P_gen) into the
subspace where the fact is *linearly decodable* — the precise condition PoLM
Part 3.1 shows is necessary for knowledge extractability.

Formula (implementation_plan.md §Loss Variant 2):
    L_Probe = CE(φ*(h_E^{l_t}(P_gen)), y*)

where:
  φ* : R^d → R^|V|  — frozen linear probe (pretrained by pretrain_probe.py)
  y*                 — first answer token id
  Gradient flows through h_E^{l_t}(P_gen) only; φ* weights are frozen.

The probe is loaded from data/processed/probe_phi_<model_key>.pt by the
training script. This module only wraps the frozen probe into the AlignmentLoss
interface and handles the CE computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AlignmentLoss


class ProbeLoss(AlignmentLoss):
    """
    CE loss through a frozen linear probe φ*.

    Args:
        probe : nn.Linear(hidden_size, vocab_size) — must be frozen and eval
    """

    def __init__(self, probe: nn.Linear):
        self.probe   = probe
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Verify probe is frozen
        for p in self.probe.parameters():
            if p.requires_grad:
                raise ValueError(
                    "ProbeLoss: probe must be frozen. "
                    "Call probe.requires_grad_(False) before wrapping."
                )

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> "ProbeLoss":
        """Load φ* from a .pt checkpoint produced by pretrain_probe.py."""
        ckpt = torch.load(path, map_location=device)
        probe = nn.Linear(ckpt["hidden_size"], ckpt["vocab_size"], bias=True)
        probe.load_state_dict(ckpt["state_dict"])
        probe.requires_grad_(False)
        probe.eval()
        probe = probe.to(device)
        val_acc = ckpt.get("val_acc", "N/A")
        print(f"  ProbeLoss: loaded φ* from {path} (val_acc={val_acc:.3f})")
        return cls(probe)

    def forward(
        self,
        h_mem_early: torch.Tensor,   # unused — probe operates on h_gen at l_t
        h_mem_late:  torch.Tensor,   # unused
        h_gen:       torch.Tensor,   # (B, D) — P_gen mid-layer rep, has grad
        tgt_ids:     torch.Tensor | None = None,  # (B, max_entity_len) token ids
        **kwargs,
    ) -> torch.Tensor:
        if tgt_ids is None:
            raise ValueError("ProbeLoss.forward() requires 'tgt_ids' kwarg.")

        # L2-normalise h_gen before probing (matches training convention)
        h_norm = F.normalize(h_gen.float(), dim=-1)      # (B, D)  — has grad
        logits = self.probe(h_norm)                        # (B, V)

        # Use the first valid (non -100) answer token as label
        device = h_gen.device
        labels = tgt_ids[:, 0].to(device)                 # (B,)
        # Safety: mask any remaining -100 labels (shouldn't happen but just in case)
        labels = labels.masked_fill(labels == tokenizer_pad_id(tgt_ids), -100)

        return self.loss_fn(logits, labels)


def tokenizer_pad_id(tgt_ids: torch.Tensor) -> int:
    """Return the pad id used in tgt_ids (-100 by construction in paired_dataloader)."""
    return -100
