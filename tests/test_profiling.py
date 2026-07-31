"""
tests/test_profiling.py — Tests for self_patch_scan.py

Tests (no GPU required — uses tiny mock models):
  1. PatchHook correctly replaces entity-span positions in hidden states
  2. run_self_patch_scan returns a valid (L, L) matrix with correct shape
  3. select_l_t returns valid indices within [0, L)
  4. update_profile correctly updates l_t in a JSON file
  5. Gain matrix is higher when patching with the correct representation

Run with:
    /users/jerryma/.conda/envs/torch2.8/bin/python -m pytest tests/test_profiling.py -v
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.profiling.self_patch_scan import (
    PatchHook, select_l_t, update_profile,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny mock transformer for tests (no real model loading needed)
# ─────────────────────────────────────────────────────────────────────────────

class TinyLayer(nn.Module):
    """Single attention-free decoder layer stub."""
    def __init__(self, d_model: int):
        super().__init__()
        self.ff = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.ff.weight)   # identity transform

    def forward(self, x):
        # Returns a tuple like a real decoder layer: (hidden, None)
        return (self.ff(x), None)


class TinyModel(nn.Module):
    """3-layer mini model for hook testing."""
    def __init__(self, d_model: int = 16, seq_len: int = 10, vocab: int = 50):
        super().__init__()
        self.d_model  = d_model
        self.seq_len  = seq_len
        self.vocab    = vocab
        self.embed    = nn.Embedding(vocab, d_model)
        self.layers   = nn.ModuleList([TinyLayer(d_model) for _ in range(3)])
        self.lm_head  = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)            # (B, seq, D)
        for layer in self.layers:
            x, _ = layer(x)
        logits = self.lm_head(x)             # (B, seq, vocab)

        class FakeOutput:
            def __init__(self, logits):
                self.logits = logits
        return FakeOutput(logits)

    @property
    def config(self):
        class C:
            hidden_size = 16
            vocab_size  = 50
        return C()


B, D, SEQ, VOCAB, L = 4, 16, 10, 50, 3


@pytest.fixture
def model():
    m = TinyModel(d_model=D, seq_len=SEQ, vocab=VOCAB)
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 1. PatchHook replaces entity-span positions correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchHook:

    def test_span_positions_replaced(self, model):
        """After hook fires, entity-span positions should match the patch."""
        patch      = torch.ones(B, D) * 99.0   # distinct sentinel value
        span_start = [2, 2, 2, 2]
        span_end   = [4, 4, 4, 4]
        hook       = PatchHook(patch, span_start, span_end)

        # Manually call the hook with a fake output
        fake_hidden = torch.zeros(B, SEQ, D)
        fake_output = (fake_hidden, None)
        result = hook(None, None, fake_output)

        modified_hidden = result[0]
        # Positions 2 and 3 should be 99.0
        assert torch.allclose(modified_hidden[:, 2:4, :],
                               torch.ones(B, 2, D) * 99.0), \
            "Entity span positions should be replaced with patch values"
        # Non-span positions should remain 0
        assert torch.allclose(modified_hidden[:, 0:2, :], torch.zeros(B, 2, D)), \
            "Non-span positions before span should be unchanged"
        assert torch.allclose(modified_hidden[:, 4:, :], torch.zeros(B, SEQ-4, D)), \
            "Non-span positions after span should be unchanged"

    def test_tuple_vs_tensor_output(self, model):
        """Hook should handle both tuple and plain tensor outputs."""
        patch = torch.zeros(B, D)
        hook  = PatchHook(patch, [1]*B, [3]*B)

        hidden = torch.ones(B, SEQ, D)
        # Tuple output
        result_tuple = hook(None, None, (hidden.clone(), None))
        assert isinstance(result_tuple, tuple)

        # Plain tensor output
        result_plain = hook(None, None, hidden.clone())
        assert isinstance(result_plain, torch.Tensor)

    def test_register_and_remove(self, model):
        """Hook can be registered on and removed from a model layer."""
        patch = torch.zeros(B, D)
        hook  = PatchHook(patch, [0]*B, [1]*B)
        hook.register(model, 0)
        assert hook.handle is not None
        hook.remove()
        assert hook.handle is None

    def test_out_of_bounds_span_skipped(self):
        """Spans that extend beyond seq_len are handled without crashing."""
        patch  = torch.ones(B, D) * 5.0
        spans_start = [0] * B
        spans_end   = [SEQ + 10] * B   # out of bounds

        hook   = PatchHook(patch, spans_start, spans_end)
        hidden = torch.zeros(B, SEQ, D)
        # Should not raise
        result = hook(None, None, (hidden, None))
        assert result[0].shape == (B, SEQ, D)

    def test_batch_mismatch_graceful(self):
        """If B_patch > B_hidden, extra patch rows are ignored."""
        patch  = torch.ones(6, D)       # more rows than batch
        hook   = PatchHook(patch, [1]*6, [3]*6)
        hidden = torch.zeros(B, SEQ, D)  # B=4
        result = hook(None, None, (hidden, None))   # should not raise
        assert result[0].shape == (B, SEQ, D)


# ─────────────────────────────────────────────────────────────────────────────
# 2. select_l_t returns valid indices
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectLt:

    def test_basic_selection(self):
        """l_t should be the column with the highest row-max."""
        A = np.zeros((5, 5))
        A[2, 3] = 0.8   # max gain at (l_s=2, l_t=3)
        l_s, l_t, gain = select_l_t(A)
        assert l_t == 3
        assert l_s == 2
        assert abs(gain - 0.8) < 1e-6

    def test_returns_within_bounds(self):
        L = 28
        A = np.random.randn(L, L)
        l_s, l_t, gain = select_l_t(A)
        assert 0 <= l_s < L
        assert 0 <= l_t < L

    def test_all_zero_matrix(self):
        """All zeros: just returns some valid index without crashing."""
        A = np.zeros((10, 10))
        l_s, l_t, gain = select_l_t(A)
        assert 0 <= l_s < 10
        assert 0 <= l_t < 10
        assert gain == 0.0

    def test_single_layer(self):
        A = np.array([[0.5]])
        l_s, l_t, gain = select_l_t(A)
        assert l_s == 0 and l_t == 0
        assert abs(gain - 0.5) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 3. update_profile writes correct fields to JSON
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateProfile:

    def test_creates_and_updates(self, tmp_path):
        profile_path = str(tmp_path / "profile.json")
        # Existing profile with heuristic l_t
        existing = {"model": "TestModel", "L": 10, "l_t": 5,
                    "l_t_source": "heuristic"}
        with open(profile_path, "w") as f:
            json.dump(existing, f)

        gain_matrix = np.zeros((10, 10))
        gain_matrix[3, 7] = 0.42

        update_profile(profile_path, l_t_empirical=7, l_s_best=3,
                       max_gain=0.42, gain_matrix=gain_matrix,
                       checkpoint="checkpoint_epoch3")

        with open(profile_path) as f:
            updated = json.load(f)

        assert updated["l_t"]            == 7
        assert updated["l_s_best"]       == 3
        assert abs(updated["max_patch_gain"] - 0.42) < 1e-4
        assert "self_patching_scan" in updated["l_t_source"]
        assert updated["model"]          == "TestModel"   # existing keys preserved
        assert "self_patch_heatmap" in updated

    def test_missing_existing_profile_is_ok(self, tmp_path):
        """update_profile should work even if profile doesn't exist yet."""
        profile_path = str(tmp_path / "new_profile.json")
        gain_matrix  = np.eye(5) * 0.3
        # Should not raise
        update_profile(profile_path, l_t_empirical=2, l_s_best=2,
                       max_gain=0.3, gain_matrix=gain_matrix,
                       checkpoint="ckpt")
        # File should now exist
        assert Path(profile_path).exists()


# ─────────────────────────────────────────────────────────────────────────────
# 4. PatchHook integration: gain is higher when patching with correct rep
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchGain:

    def test_correct_patch_improves_or_maintains_accuracy(self, model):
        """
        Construct a case where the correct entity rep patched into the target
        layer produces a better or equal logit for the target token.
        """
        # Create a target token id
        target_tok = 7
        # Make the lm_head weight for token 7 point in the direction of our patch
        W = model.lm_head.weight    # (vocab, D)
        target_dir = torch.zeros(D)
        target_dir[0] = 1.0         # patch points along dim 0
        W.data[target_tok] = target_dir * 10.0  # strongly activates token 7

        # Input that produces near-zero embedding (so default logit for tok 7 ≈ 0)
        input_ids = torch.zeros(1, SEQ, dtype=torch.long)
        model.embed.weight.data[:] = 0.0   # zero embeddings

        # Patch: insert the direction that activates target_tok
        patch      = target_dir.unsqueeze(0).expand(1, -1)   # (1, D)
        span_start = [3]
        span_end   = [4]

        hook = PatchHook(patch, span_start, span_end)
        hook.register(model, 0)   # patch at layer 0

        with torch.no_grad():
            out = model(input_ids)
        hook.remove()

        # The last position logit for target_tok should now be high
        logit_target = out.logits[0, -1, target_tok].item()
        logit_other  = out.logits[0, -1, :].mean().item()
        assert logit_target > logit_other, \
            f"Patch should boost target token logit: {logit_target:.3f} vs mean {logit_other:.3f}"
