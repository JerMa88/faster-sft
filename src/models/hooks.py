"""
src/models/hooks.py
====================
Forward hook utilities for extracting entity-span representations at
specific decoder layers during a single forward pass.

Supported architectures (auto-detected):
  - meta-llama/Llama-3.2-3B-Instruct   → model.model.layers[i]
  - Qwen/Qwen3.5-2B                   → model.model.layers[i]
  - google/gemma-2-2b-it                → model.model.layers[i]
  - fdtn-ai/antares-1b                  → model.model.layers[i]
  - Nanbeige/Nanbeige4.2-3B (Looped)    → model.model.layers[i]
  - LiquidAI/LFM2.5-1.2B-Base          → model.model.layers[i] (hybrid layers)

All six use the standard `model.model.layers` pattern (verified by HF config).
The deprecated HRM-Text dispatch path is kept for backward compatibility but
is no longer in the active model list.

Layer output formats handled:
  - Tuple (hidden_states, ...)  — standard decoder (Llama, Qwen, Gemma)
  - Plain tensor                — some custom architectures
  - 2-D hidden states           — rare, handled via batch reshape
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# Representation Cache
# ─────────────────────────────────────────────────────────────────────────────

class RepresentationCache:
    """
    Stores entity-span representations keyed by layer index.
    call .clear() before each forward pass to reset.
    """

    def __init__(self):
        self.cache: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.cache.clear()

    def __contains__(self, key):
        return key in self.cache

    def __repr__(self):
        keys = list(self.cache.keys())
        shapes = {k: tuple(v.shape) for k, v in self.cache.items()}
        return f"RepresentationCache(layers={keys}, shapes={shapes})"


# ─────────────────────────────────────────────────────────────────────────────
# Hook factory
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_hook(
    layer_idx:    int,
    cache:        RepresentationCache,
    entity_spans: Optional[List[Tuple[int, int]]] = None,
):
    """
    Returns a forward hook that extracts hidden states at the given layer.

    Args:
        layer_idx    : which layer index to cache under
        cache        : RepresentationCache to write into
        entity_spans : list of (start, end) tuples per batch item.
                       If None, caches the full sequence (B, seq_len, D).
                       If provided, caches mean-pooled entity rep (B, D).
    """
    def hook(module: nn.Module, inputs: tuple, outputs):
        # Normalise output: extract hidden states tensor regardless of format
        if isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs   # plain tensor

        # Handle rare 2-D hidden states (some custom layers)
        if hidden_states.dim() == 2 and entity_spans is not None:
            batch_size = len(entity_spans)
            seq_len    = hidden_states.size(0) // batch_size
            hidden_states = hidden_states.view(batch_size, seq_len, -1)

        if entity_spans is None:
            # Cache full sequence: retain grad if needed, else detach
            stored = hidden_states if hidden_states.requires_grad \
                     else hidden_states.detach()
            cache.cache[layer_idx] = stored
        else:
            batch_size  = hidden_states.size(0)
            entity_reps = []
            for i in range(min(batch_size, len(entity_spans))):
                span = entity_spans[i]
                if isinstance(span, (tuple, list)) and len(span) == 2:
                    start, end = int(span[0]), int(span[1])
                    # Clamp to valid range
                    start = max(0, min(start, hidden_states.size(1) - 1))
                    end   = max(start + 1, min(end, hidden_states.size(1)))
                    span_states = hidden_states[i, start:end, :]
                else:
                    # Legacy: list of explicit token indices
                    indices     = torch.tensor(span, dtype=torch.long,
                                               device=hidden_states.device)
                    indices     = indices.clamp(0, hidden_states.size(1) - 1)
                    span_states = hidden_states[i, indices, :]

                mean_pooled = span_states.mean(dim=0)  # (D,)
                entity_reps.append(mean_pooled)

            stacked = torch.stack(entity_reps)   # (B, D)
            cache.cache[layer_idx] = stacked

        return outputs   # never modify outputs — read-only hook

    return hook


# ─────────────────────────────────────────────────────────────────────────────
# Layer accessor — architecture dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _get_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """
    Return the layer_idx-th decoder layer from any supported architecture.

    Dispatch order:
      1. PEFT wrapper  → unwrap base_model first
      2. HRM-Text      → model.L_module / model.H_module (legacy)
      3. Standard      → model.model.layers[i]  (Llama, Qwen, Gemma, Antares,
                          Nanbeige, LFM2.5)
      4. Bare layers   → model.layers[i]
    """
    base = model
    # Unwrap PEFT / LoRA wrappers
    if hasattr(base, "peft_config") and hasattr(base, "base_model"):
        base = base.base_model.model

    # Legacy HRM-Text (L_module + H_module, 16 layers each)
    if hasattr(base, "model") and hasattr(base.model, "L_module"):
        if layer_idx < 16:
            return base.model.L_module.layers[layer_idx]
        else:
            return base.model.H_module.layers[layer_idx - 16]

    # Standard: model.model.layers  (covers all 6 active models)
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        layers = base.model.layers
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(
                f"Layer index {layer_idx} out of range for model with "
                f"{len(layers)} layers."
            )
        return layers[layer_idx]

    # Bare: model.layers (some custom wrappers)
    if hasattr(base, "layers"):
        layers = base.layers
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(
                f"Layer index {layer_idx} out of range (bare layers, "
                f"n_layers={len(layers)})."
            )
        return layers[layer_idx]

    raise ValueError(
        f"Unsupported model architecture. Cannot locate layer {layer_idx}. "
        f"Available attributes: {[a for a in dir(base) if not a.startswith('_')]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def register_hooks(
    model:        nn.Module,
    layer_indices: List[int],
    cache:        RepresentationCache,
    entity_spans: Optional[List[Tuple[int, int]]] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward hooks on the specified decoder layers.

    Args:
        model         : the language model (PEFT-wrapped or bare)
        layer_indices : list of layer indices to hook
        cache         : RepresentationCache to write representations into
        entity_spans  : optional list of (start, end) per batch item for mean-pooling

    Returns:
        List of removable hook handles. Call handle.remove() after the forward
        pass to prevent memory leaks.
    """
    handles = []
    for idx in layer_indices:
        layer  = _get_layer(model, idx)
        handle = layer.register_forward_hook(
            get_layer_hook(idx, cache, entity_spans)
        )
        handles.append(handle)
    return handles
