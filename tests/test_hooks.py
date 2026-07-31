"""
tests/test_hooks.py
====================
Tests for src/models/hooks.py — hook registration and architecture dispatch.

Coverage (per implementation_plan.md Step E):
  1. RepresentationCache — store, clear, contains
  2. get_layer_hook — full-sequence caching, span mean-pooling, output passthrough
  3. _get_layer dispatch — all 6 mock architectures + PEFT wrapper + legacy HRM-Text
  4. register_hooks — multi-layer registration, handle removal, cache population
  5. Architecture compatibility — plain-tensor output, 2D output, OOB spans
  6. Gradient flow — representations retain grad when requires_grad=True

No GPU required. All tests use tiny mock nn.Module objects.

Run with:
    /users/jerryma/.conda/envs/torch2.8/bin/python -m pytest tests/test_hooks.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn

from src.models.hooks import (
    RepresentationCache,
    get_layer_hook,
    register_hooks,
    _get_layer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mock layer / model helpers
# ─────────────────────────────────────────────────────────────────────────────

class MockLayer(nn.Module):
    """Identity layer that returns (hidden_states, None) like a real decoder layer."""
    def forward(self, x):
        return (x, None)


class MockLayerPlainTensor(nn.Module):
    """Layer that returns a plain tensor (some SSM/hybrid architectures)."""
    def forward(self, x):
        return x


class MockInnerModel(nn.Module):
    def __init__(self, n_layers: int = 4, output_tuple: bool = True):
        super().__init__()
        LayerClass = MockLayer if output_tuple else MockLayerPlainTensor
        self.layers = nn.ModuleList([LayerClass() for _ in range(n_layers)])


class MockModel(nn.Module):
    """Standard wrapper: model.model.layers (Llama, Qwen, Gemma, Antares, Nanbeige, LFM2.5)."""
    def __init__(self, n_layers: int = 4, output_tuple: bool = True):
        super().__init__()
        self.model = MockInnerModel(n_layers, output_tuple)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        return x


class MockBareModel(nn.Module):
    """Bare model: model.layers (fallback path)."""
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x


class MockPEFTWrapper(nn.Module):
    """Simulates PEFT/LoRA: model.base_model.model.model.layers."""
    class _BaseModel(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

    def __init__(self, n_layers: int = 4):
        super().__init__()
        inner = MockModel(n_layers)
        self.base_model = self._BaseModel(inner)

    def forward(self, x):
        return self.base_model.model(x)


class MockHRMModel(nn.Module):
    """Legacy HRM-Text: model.model.L_module.layers + H_module.layers."""
    class _Trunk(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer() for _ in range(n)])

    class _InnerHRM(nn.Module):
        def __init__(self, n_l, n_h):
            super().__init__()
            self.L_module = MockHRMModel._Trunk(n_l)
            self.H_module = MockHRMModel._Trunk(n_h)

    def __init__(self, n_l: int = 4, n_h: int = 4):
        super().__init__()
        self.model = self._InnerHRM(n_l, n_h)


# Six mock models named after their real HF counterparts
def make_llama():    return MockModel(28)                      # Llama-3.2-3B: 28 layers
def make_qwen():     return MockModel(28)                      # Qwen3.5-1.5B: 28 layers
def make_gemma():    return MockModel(26)                      # Gemma4-E4B: 26 layers
def make_antares():  return MockModel(24)                      # Antares-1B: ~24 layers
def make_nanbeige(): return MockModel(32)                      # Nanbeige4.2-3B: 32 layers
def make_lfm():      return MockModel(28, output_tuple=False)  # LFM2.5: plain-tensor layers

B, D, SEQ = 4, 16, 10


# ─────────────────────────────────────────────────────────────────────────────
# 1. RepresentationCache
# ─────────────────────────────────────────────────────────────────────────────

class TestRepresentationCache:

    def test_empty_on_init(self):
        cache = RepresentationCache()
        assert len(cache.cache) == 0

    def test_store_and_retrieve(self):
        cache = RepresentationCache()
        t = torch.randn(B, D)
        cache.cache[3] = t
        assert 3 in cache
        assert torch.equal(cache.cache[3], t)

    def test_clear(self):
        cache = RepresentationCache()
        cache.cache[1] = torch.zeros(B, D)
        cache.cache[5] = torch.zeros(B, D)
        cache.clear()
        assert len(cache.cache) == 0
        assert 1 not in cache

    def test_contains(self):
        cache = RepresentationCache()
        assert 0 not in cache
        cache.cache[0] = torch.zeros(2, D)
        assert 0 in cache

    def test_repr_contains_keys(self):
        cache = RepresentationCache()
        cache.cache[7] = torch.zeros(B, D)
        r = repr(cache)
        assert "7" in r

    def test_multi_layer(self):
        cache = RepresentationCache()
        for l in [0, 5, 10, 20]:
            cache.cache[l] = torch.randn(B, D)
        assert set(cache.cache.keys()) == {0, 5, 10, 20}


# ─────────────────────────────────────────────────────────────────────────────
# 2. get_layer_hook — caching behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLayerHook:

    def _out(self, req_grad=False, as_tuple=True):
        h = torch.randn(B, SEQ, D, requires_grad=req_grad)
        return (h, None) if as_tuple else h

    def test_full_sequence_cached_tuple(self):
        cache = RepresentationCache()
        hook  = get_layer_hook(3, cache, entity_spans=None)
        hook(None, None, self._out())
        assert 3 in cache
        assert cache.cache[3].shape == (B, SEQ, D)

    def test_full_sequence_cached_plain_tensor(self):
        cache = RepresentationCache()
        hook  = get_layer_hook(7, cache, entity_spans=None)
        hook(None, None, self._out(as_tuple=False))
        assert 7 in cache
        assert cache.cache[7].shape == (B, SEQ, D)

    def test_span_mean_pooling(self):
        cache = RepresentationCache()
        spans = [(2, 5)] * B
        hook  = get_layer_hook(1, cache, entity_spans=spans)
        hook(None, None, self._out())
        assert 1 in cache
        assert cache.cache[1].shape == (B, D)

    def test_span_mean_equals_manual(self):
        cache  = RepresentationCache()
        spans  = [(2, 5)] * B
        hook   = get_layer_hook(0, cache, entity_spans=spans)
        hidden = torch.randn(B, SEQ, D)
        hook(None, None, (hidden, None))
        expected = hidden[:, 2:5, :].mean(dim=1)
        assert torch.allclose(cache.cache[0], expected, atol=1e-5)

    def test_output_passthrough_tuple(self):
        cache = RepresentationCache()
        hook  = get_layer_hook(0, cache)
        out   = self._out()
        assert hook(None, None, out) is out

    def test_output_passthrough_plain(self):
        cache = RepresentationCache()
        hook  = get_layer_hook(0, cache)
        t     = torch.randn(B, SEQ, D)
        assert hook(None, None, t) is t

    def test_grad_retained_when_required(self):
        cache = RepresentationCache()
        hook  = get_layer_hook(0, cache, entity_spans=None)
        h = torch.randn(B, SEQ, D, requires_grad=True)
        hook(None, None, (h, None))
        assert cache.cache[0].requires_grad

    def test_detached_when_no_grad(self):
        cache = RepresentationCache()
        hook  = get_layer_hook(0, cache, entity_spans=None)
        h = torch.randn(B, SEQ, D, requires_grad=False)
        hook(None, None, (h, None))
        assert not cache.cache[0].requires_grad

    def test_oob_span_clamped(self):
        cache = RepresentationCache()
        spans = [(1, SEQ + 100)] * B
        hook  = get_layer_hook(0, cache, entity_spans=spans)
        hook(None, None, self._out())
        assert 0 in cache
        assert cache.cache[0].shape == (B, D)

    def test_batch_partial_spans(self):
        """Fewer spans than batch: only fills up to len(spans) rows."""
        cache = RepresentationCache()
        spans = [(1, 3)] * (B - 1)
        hook  = get_layer_hook(0, cache, entity_spans=spans)
        hook(None, None, self._out())
        assert cache.cache[0].shape == (B - 1, D)


# ─────────────────────────────────────────────────────────────────────────────
# 3. _get_layer architecture dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLayer:

    @pytest.mark.parametrize("model_fn,idx", [
        (make_llama,     0),
        (make_llama,    14),
        (make_llama,    27),
        (make_qwen,      7),
        (make_gemma,    13),
        (make_antares,  12),
        (make_nanbeige, 16),
        (make_lfm,      20),
    ])
    def test_standard_dispatch(self, model_fn, idx):
        layer = _get_layer(model_fn(), idx)
        assert isinstance(layer, nn.Module), \
            f"Expected nn.Module at layer {idx} for {model_fn.__name__}"

    def test_peft_wrapper_unwrapped(self):
        model = MockPEFTWrapper(n_layers=4)
        layer = _get_layer(model, 2)
        assert isinstance(layer, nn.Module)

    def test_bare_model_fallback(self):
        model = MockBareModel(n_layers=3)
        layer = _get_layer(model, 1)
        assert isinstance(layer, nn.Module)

    def test_hrm_legacy_l_module(self):
        model = MockHRMModel(n_l=4, n_h=4)
        layer = _get_layer(model, 2)
        assert isinstance(layer, nn.Module)

    def test_hrm_legacy_h_module(self):
        model = MockHRMModel(n_l=4, n_h=4)
        layer = _get_layer(model, 16)
        assert isinstance(layer, nn.Module)

    def test_out_of_range_raises(self):
        with pytest.raises((IndexError, ValueError)):
            _get_layer(make_llama(), 999)

    def test_unsupported_raises(self):
        class NoLayers(nn.Module):
            pass
        with pytest.raises(ValueError, match="Unsupported"):
            _get_layer(NoLayers(), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. register_hooks — multi-layer, removal, cache population
# ─────────────────────────────────────────────────────────────────────────────

class TestRegisterHooks:

    def _run(self, model, layer_indices, entity_spans=None):
        cache   = RepresentationCache()
        handles = register_hooks(model, layer_indices, cache, entity_spans)
        model(torch.randn(B, SEQ, D))
        for h in handles:
            h.remove()
        return cache, handles

    @pytest.mark.parametrize("model_fn,layers", [
        (make_llama,    [0, 14, 27]),
        (make_qwen,     [0, 10, 20]),
        (make_gemma,    [5, 13, 25]),
        (make_antares,  [3, 12, 23]),
        (make_nanbeige, [0, 15, 31]),
        (make_lfm,      [0, 10, 27]),
    ])
    def test_all_six_architectures(self, model_fn, layers):
        cache, _ = self._run(model_fn(), layers)
        for l in layers:
            assert l in cache, f"Layer {l} missing for {model_fn.__name__}"

    def test_cached_shape_full_sequence(self):
        cache, _ = self._run(make_qwen(), [5])
        assert cache.cache[5].shape == (B, SEQ, D)

    def test_cached_shape_span_pooled(self):
        spans = [(2, 5)] * B
        cache, _ = self._run(make_llama(), [3], entity_spans=spans)
        assert cache.cache[3].shape == (B, D)

    def test_handles_removable(self):
        model   = make_qwen()
        cache   = RepresentationCache()
        handles = register_hooks(model, [0, 5], cache)
        for h in handles:
            h.remove()
        cache.clear()
        model(torch.randn(B, SEQ, D))
        assert len(cache.cache) == 0, "Removed hooks must not fire"

    def test_returns_correct_handle_count(self):
        model   = make_llama()
        cache   = RepresentationCache()
        handles = register_hooks(model, [0, 5, 10], cache)
        for h in handles:
            h.remove()
        assert len(handles) == 3

    def test_empty_layer_list(self):
        cache, handles = self._run(make_llama(), [])
        assert len(handles) == 0 and len(cache.cache) == 0

    def test_peft_hooks_work(self):
        model = MockPEFTWrapper(n_layers=4)
        inner = model.base_model.model
        cache = RepresentationCache()
        handles = register_hooks(inner, [1, 2], cache)
        inner(torch.randn(B, SEQ, D))
        for h in handles:
            h.remove()
        assert 1 in cache and 2 in cache


# ─────────────────────────────────────────────────────────────────────────────
# 5. Architecture compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestArchitectureCompatibility:

    def test_plain_tensor_output_lfm(self):
        model = make_lfm()
        cache = RepresentationCache()
        handles = register_hooks(model, [0, 5], cache)
        model(torch.randn(B, SEQ, D))
        for h in handles:
            h.remove()
        assert 0 in cache and 5 in cache
        assert cache.cache[0].shape == (B, SEQ, D)

    def test_2d_hidden_state_reshaped(self):
        class TwoDLayer(nn.Module):
            def forward(self, x):
                B, S, D = x.shape
                return (x.view(B * S, D), None)

        class TwoDModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([TwoDLayer()])
            def forward(self, x):
                h, _ = self.model.layers[0](x)
                return h

        model = TwoDModel()
        cache = RepresentationCache()
        spans = [(2, 4)] * B
        handles = register_hooks(model, [0], cache, spans)
        model(torch.randn(B, SEQ, D))
        for h in handles:
            h.remove()
        assert cache.cache[0].shape == (B, D)

    def test_nanbeige_looped_32_layers(self):
        """Nanbeige4.2-3B has 32 layers — verify all extremes accessible."""
        model = make_nanbeige()
        for idx in [0, 16, 31]:
            layer = _get_layer(model, idx)
            assert isinstance(layer, nn.Module)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Gradient flow
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_hooks_dont_break_backward(self):
        model   = make_llama()
        cache   = RepresentationCache()
        spans   = [(2, 5)] * B
        handles = register_hooks(model, [3], cache, spans)

        x    = torch.randn(B, SEQ, D, requires_grad=True)
        out  = model(x)
        out.sum().backward()
        for h in handles:
            h.remove()

        assert x.grad is not None, "Gradient must flow through hooked model"

    def test_no_grad_context_detached(self):
        model   = make_llama()
        cache   = RepresentationCache()
        handles = register_hooks(model, [5], cache, entity_spans=None)
        with torch.no_grad():
            model(torch.randn(B, SEQ, D))
        for h in handles:
            h.remove()
        assert not cache.cache[5].requires_grad
