"""
src/training/losses.py — backward-compatible re-exports.

The canonical implementations now live in src/losses/.
This file re-exports the functional forms so existing imports in
train_sft.py and tests continue to work without modification.
"""

from src.losses.rep_distill import rep_distill_loss
from src.losses.contrastive import contrastive_loss

__all__ = ["rep_distill_loss", "contrastive_loss"]
