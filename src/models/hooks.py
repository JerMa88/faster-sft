import torch
import torch.nn as nn
from typing import Dict, List, Optional

class RepresentationCache:
    def __init__(self):
        self.cache: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.cache.clear()

def get_layer_hook(layer_idx: int, cache: RepresentationCache, entity_spans: Optional[List[List[int]]] = None):
    """
    Returns a forward hook that extracts the hidden state at the specified layer.
    If entity_spans is provided, it extracts the mean-pooled representation over the entity tokens.
    Otherwise, it caches the full sequence hidden state.
    
    entity_spans: list of token index lists, one for each sequence in the batch.
                  e.g. [[2, 3], [4, 5, 6]]
    """
    def hook(module: nn.Module, inputs: tuple, outputs: tuple):
        # outputs is a tuple for Llama/Qwen layers, first element is hidden_states
        # shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs[0]
        
        if entity_spans is None:
            # Cache the full hidden states
            cache.cache[layer_idx] = hidden_states.detach() if not hidden_states.requires_grad else hidden_states
        else:
            if len(hidden_states.shape) == 2:
                # Reshape to (batch_size, seq_len, hidden_size)
                # We know the first dimension of the original input is batch_size
                batch_size = len(entity_spans)
                seq_len = hidden_states.size(0) // batch_size
                hidden_states = hidden_states.view(batch_size, seq_len, -1)
                
            batch_size = hidden_states.size(0)
            entity_reps = []
            for i in range(batch_size):
                span = entity_spans[i]
                if isinstance(span, (tuple, list)) and len(span) == 2:
                    start, end = span
                    span_states = hidden_states[i, start:end, :]
                else:
                    span_states = hidden_states[i, span, :]
                mean_pooled = span_states.mean(dim=0)
                entity_reps.append(mean_pooled)
            
            # shape: (batch_size, hidden_size)
            stacked_reps = torch.stack(entity_reps)
            cache.cache[layer_idx] = stacked_reps
            
        return outputs
    return hook

def register_hooks(model: nn.Module, layer_indices: List[int], cache: RepresentationCache, entity_spans: Optional[List[List[int]]] = None) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Registers forward hooks on the specified layers of the model.
    """
    def _get_layer(layer_idx: int):
        base = model
        if hasattr(base, "base_model"):
            base = base.base_model.model
            
        # Handle HRM-Text architecture (L_module and H_module)
        if hasattr(base, "model") and hasattr(base.model, "L_module"):
            if layer_idx < 16:
                return base.model.L_module.layers[layer_idx]
            else:
                return base.model.H_module.layers[layer_idx - 16]
        # Handle standard Llama/Qwen architecture
        elif hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers[layer_idx]
        elif hasattr(base, "layers"):
            return base.layers[layer_idx]
        else:
            raise ValueError(f"Unsupported model architecture for hook registration. Base structure: {dir(base)}")

    handles = []
    for idx in layer_indices:
        layer = _get_layer(idx)
        handle = layer.register_forward_hook(get_layer_hook(idx, cache, entity_spans))
        handles.append(handle)
        
    return handles
