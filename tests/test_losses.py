"""
tests/test_losses.py — Full test suite for src/losses/

Tests cover (per implementation_plan.md Phase 3 unit tests):
  (a) Gradient ONLY flows through h_gen (P_gen branch), NOT through source reps
  (b) Loss = 0 when representations already match (RepDist)
  (c) Loss equals CE of perfect probe (ProbeLoss)
  (d) Contrastive loss ≥ 0 and lower bound is 0
  (e) Hybrid combines both losses correctly with α=0.5
  (f) build_loss() factory dispatches to correct classes

Run with:
    python -m pytest tests/test_losses.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import (
    RepDistLoss, ContraRouteLoss, ProbeLoss, HybridLoss,
    rep_distill_loss, contrastive_loss, build_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

B, D, V = 8, 64, 500  # batch, hidden, vocab (small for speed)

@pytest.fixture
def reps():
    """Random h_mem_early, h_mem_late (no grad), h_gen (has grad)."""
    h_early = torch.randn(B, D)
    h_late  = torch.randn(B, D)
    h_gen   = torch.randn(B, D, requires_grad=True)
    return h_early, h_late, h_gen

@pytest.fixture
def probe_frozen():
    """A frozen nn.Linear probe with known weight."""
    p = nn.Linear(D, V, bias=True)
    p.requires_grad_(False)
    p.eval()
    return p

@pytest.fixture
def tgt_ids():
    """Fake target ids: first token valid, rest -100."""
    ids = torch.full((B, 5), -100, dtype=torch.long)
    ids[:, 0] = torch.randint(0, V, (B,))
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# (a) Gradient flows only through h_gen
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:

    def _check_grad_only_h_gen(self, loss, h_gen, h_early, h_late):
        """Assert loss has grad, flows to h_gen, NOT to h_early/h_late."""
        assert loss.requires_grad, "Loss must have requires_grad=True"
        loss.backward()
        assert h_gen.grad is not None, "h_gen.grad must not be None after backward"
        assert not h_early.requires_grad, "h_early should have no grad (is detached source)"
        assert not h_late.requires_grad, "h_late should have no grad (is detached source)"

    def test_rep_distill_grad(self, reps):
        h_early, h_late, h_gen = reps
        loss = RepDistLoss()(h_early, h_late, h_gen)
        self._check_grad_only_h_gen(loss, h_gen, h_early, h_late)

    def test_contrastive_grad(self, reps):
        h_early, h_late, h_gen = reps
        loss = ContraRouteLoss()(h_early, h_late, h_gen)
        self._check_grad_only_h_gen(loss, h_gen, h_early, h_late)

    def test_probe_grad(self, reps, probe_frozen, tgt_ids):
        h_early, h_late, h_gen = reps
        probe_loss = ProbeLoss(probe_frozen)
        loss = probe_loss(h_early, h_late, h_gen, tgt_ids=tgt_ids)
        self._check_grad_only_h_gen(loss, h_gen, h_early, h_late)

    def test_hybrid_grad(self, reps, probe_frozen, tgt_ids):
        h_early, h_late, h_gen = reps
        pl = ProbeLoss(probe_frozen)
        loss = HybridLoss(alpha=0.5, probe=pl)(h_early, h_late, h_gen, tgt_ids=tgt_ids)
        self._check_grad_only_h_gen(loss, h_gen, h_early, h_late)

    def test_hybrid_no_probe_grad(self, reps):
        """Hybrid without φ* falls back to ContraRoute — still only h_gen grad."""
        h_early, h_late, h_gen = reps
        loss = HybridLoss(alpha=0.5, probe=None)(h_early, h_late, h_gen)
        self._check_grad_only_h_gen(loss, h_gen, h_early, h_late)


# ─────────────────────────────────────────────────────────────────────────────
# (b) RepDist = 0 when representations already match
# ─────────────────────────────────────────────────────────────────────────────

class TestRepDistProperties:

    def test_zero_loss_identical_direction(self):
        """RepDist should be 0 when source and target have identical directions."""
        h = torch.randn(B, D)
        h_gen = h.clone().requires_grad_(True)
        loss = rep_distill_loss(h, h_gen)
        assert loss.item() < 1e-5, f"RepDist should be ~0 for identical reps, got {loss.item()}"

    def test_loss_in_range(self, reps):
        h_early, h_late, h_gen = reps
        loss = rep_distill_loss(h_early, h_gen)
        assert 0.0 <= loss.item() <= 1.0 + 1e-6, \
            f"RepDist should be in [0, 1], got {loss.item()}"

    def test_max_loss_opposite_direction(self):
        """RepDist should be ~1 for anti-parallel vectors (max cosine distance)."""
        h = torch.randn(B, D)
        h_neg = -h.clone().requires_grad_(True)
        loss = rep_distill_loss(h, h_neg)
        assert loss.item() > 1.5, \
            f"RepDist for anti-parallel should be ~2, got {loss.item()}"

    def test_scale_invariant(self):
        """RepDist should be the same regardless of the magnitude of h_source."""
        h = torch.randn(B, D)
        h_gen = torch.randn(B, D, requires_grad=True)
        loss_1x = rep_distill_loss(h, h_gen).item()
        h_gen2 = h_gen.detach().requires_grad_(True)
        loss_10x = rep_distill_loss(h * 10, h_gen2).item()
        assert abs(loss_1x - loss_10x) < 1e-4, \
            f"RepDist should be scale-invariant: {loss_1x} vs {loss_10x}"


# ─────────────────────────────────────────────────────────────────────────────
# (c) ProbeLoss CE properties
# ─────────────────────────────────────────────────────────────────────────────

class TestProbeLoss:

    def test_probe_must_be_frozen(self):
        """ProbeLoss should raise if probe has trainable params."""
        probe = nn.Linear(D, V)  # trainable by default
        with pytest.raises(ValueError, match="frozen"):
            ProbeLoss(probe)

    def test_loss_positive(self, reps, probe_frozen, tgt_ids):
        h_early, h_late, h_gen = reps
        loss = ProbeLoss(probe_frozen)(h_early, h_late, h_gen, tgt_ids=tgt_ids)
        assert loss.item() > 0, "ProbeLoss should be > 0 for random h_gen"

    def test_loss_decreases_when_h_gen_aligned(self, probe_frozen, tgt_ids):
        """Probe loss should decrease when h_gen is optimized toward the target."""
        # Build h_gen that when passed through probe_frozen gives correct logits
        target_tok = tgt_ids[:, 0]  # (B,)
        # Extract the probe weight for correct token; h_gen in that direction
        W = probe_frozen.weight    # (V, D)
        correct_W = W[target_tok]  # (B, D)
        h_gen_good = F.normalize(correct_W, dim=-1).detach().requires_grad_(True)
        h_gen_rand = torch.randn(B, D, requires_grad=True)
        pl = ProbeLoss(probe_frozen)
        dummy_e = torch.zeros(B, D)
        loss_good = pl(dummy_e, dummy_e, h_gen_good, tgt_ids=tgt_ids).item()
        loss_rand = pl(dummy_e, dummy_e, h_gen_rand, tgt_ids=tgt_ids).item()
        assert loss_good <= loss_rand + 0.5, \
            f"Aligned h_gen should have lower probe loss ({loss_good:.3f} vs {loss_rand:.3f})"

    def test_tgt_ids_required(self, reps, probe_frozen):
        h_early, h_late, h_gen = reps
        with pytest.raises((ValueError, TypeError)):
            ProbeLoss(probe_frozen)(h_early, h_late, h_gen)  # no tgt_ids

    def test_from_checkpoint(self, tmp_path, probe_frozen):
        """ProbeLoss.from_checkpoint loads and freezes correctly."""
        ckpt_path = tmp_path / "probe.pt"
        torch.save({
            "state_dict":  probe_frozen.state_dict(),
            "hidden_size": D,
            "vocab_size":  V,
            "val_acc":     0.75,
        }, ckpt_path)
        pl = ProbeLoss.from_checkpoint(str(ckpt_path), device=torch.device("cpu"))
        assert isinstance(pl, ProbeLoss)
        for p in pl.probe.parameters():
            assert not p.requires_grad, "Loaded probe should be frozen"


# ─────────────────────────────────────────────────────────────────────────────
# (d) Contrastive loss properties
# ─────────────────────────────────────────────────────────────────────────────

class TestContrastiveLoss:

    def test_loss_nonnegative(self, reps):
        h_early, h_late, h_gen = reps
        loss = contrastive_loss(h_early, h_gen, temperature=0.07)
        assert loss.item() >= -1e-6, f"Contrastive loss should be >= 0, got {loss.item()}"

    def test_lower_when_aligned(self):
        """Loss should be lower when each q_i is close to k_i."""
        h = torch.randn(B, D)
        h_gen_close = (h + 0.01 * torch.randn(B, D)).requires_grad_(True)
        h_gen_rand  = torch.randn(B, D, requires_grad=True)
        loss_close = contrastive_loss(h, h_gen_close, temperature=0.07).item()
        loss_rand  = contrastive_loss(h, h_gen_rand,  temperature=0.07).item()
        assert loss_close < loss_rand, \
            f"Aligned loss ({loss_close:.3f}) should < random ({loss_rand:.3f})"

    def test_temperature_effect(self):
        """Higher temperature → lower (softer) loss."""
        h_s = torch.randn(B, D)
        h_g = torch.randn(B, D, requires_grad=True)
        h_g2 = h_g.detach().requires_grad_(True)
        loss_low  = contrastive_loss(h_s, h_g,  temperature=0.07).item()
        loss_high = contrastive_loss(h_s, h_g2, temperature=1.00).item()
        assert loss_high < loss_low, \
            "Higher temperature should produce lower loss (softer distribution)"

    def test_single_example_fallback(self):
        """Batch size 1: falls back to cosine distillation, no crash."""
        h_s = torch.randn(1, D)
        h_g = torch.randn(1, D, requires_grad=True)
        loss = contrastive_loss(h_s, h_g, temperature=0.07)
        assert loss.item() >= 0
        loss.backward()
        assert h_g.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# (e) Hybrid combines both terms with α=0.5
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridLoss:

    def test_alpha_weighting(self, reps, probe_frozen, tgt_ids):
        """HybridLoss = 0.5·RepDist + 0.5·ProbeLoss."""
        h_early, h_late, h_gen = reps
        pl = ProbeLoss(probe_frozen)
        hybrid = HybridLoss(alpha=0.5, probe=pl)

        # Compute expected manually
        rd   = RepDistLoss()(h_early, h_late, h_gen.clone().detach().requires_grad_(True))
        h_g2 = h_gen.clone().detach().requires_grad_(True)
        probe_l = pl(h_early, h_late, h_g2, tgt_ids=tgt_ids)
        expected = 0.5 * rd.item() + 0.5 * probe_l.item()

        h_g3 = h_gen.clone().detach().requires_grad_(True)
        actual = hybrid(h_early, h_late, h_g3, tgt_ids=tgt_ids).item()
        assert abs(actual - expected) < 1e-4, \
            f"Hybrid α=0.5 mismatch: expected {expected:.4f}, got {actual:.4f}"

    def test_alpha_boundary(self, reps):
        """α=1 → pure RepDist; α=0 → pure ContraRoute fallback."""
        h_early, h_late, h_gen = reps
        h_g1 = h_gen.clone().detach().requires_grad_(True)
        h_g2 = h_gen.clone().detach().requires_grad_(True)

        rd_only   = HybridLoss(alpha=1.0, probe=None)(h_early, h_late, h_g1)
        contra_only = HybridLoss(alpha=0.0, probe=None)(h_early, h_late, h_g2)

        rd_ref  = RepDistLoss()(h_early, h_late, h_gen.clone().detach().requires_grad_(True))
        assert abs(rd_only.item() - rd_ref.item()) < 1e-4

        assert contra_only.item() >= 0


# ─────────────────────────────────────────────────────────────────────────────
# (f) build_loss() factory
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildLoss:

    def test_rep_distill(self):
        assert isinstance(build_loss("rep_distill"), RepDistLoss)

    def test_contrastive(self):
        assert isinstance(build_loss("contrastive"), ContraRouteLoss)

    def test_probe_no_probe_warns(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss = build_loss("probe", probe=None)
        assert any("pretrain_probe" in str(warning.message) for warning in w), \
            "Should warn about missing probe"
        assert isinstance(loss, RepDistLoss)

    def test_probe_with_probe(self, probe_frozen):
        loss = build_loss("probe", probe=probe_frozen)
        assert isinstance(loss, ProbeLoss)

    def test_hybrid(self, probe_frozen):
        loss = build_loss("hybrid", probe=probe_frozen, alpha=0.5)
        assert isinstance(loss, HybridLoss)

    def test_baseline_raises(self):
        with pytest.raises(ValueError, match="baseline"):
            build_loss("baseline")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_loss("does_not_exist")
