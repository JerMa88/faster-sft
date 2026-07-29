import socket
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.hooks import RepresentationCache, register_hooks

@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_id = "sapientinc/HRM-Text-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu" # For testing, cpu is fine
    )
    return model, tokenizer

def test_hooks_full_sequence(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    prompt = "Test prompt for hooks"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    cache = RepresentationCache()
    layers_to_hook = [3, 14, 22]
    handles = register_hooks(model, layers_to_hook, cache)
    
    with torch.no_grad():
        model(**inputs)
        
    for handle in handles:
        handle.remove()
        
    assert len(cache.cache) == 3
    for layer in layers_to_hook:
        assert layer in cache.cache
        assert cache.cache[layer].dim() == 3 # (batch, seq_len, hidden)
        assert cache.cache[layer].size(0) == 1
        assert cache.cache[layer].size(1) == inputs['input_ids'].size(1)
        assert cache.cache[layer].size(2) == model.config.hidden_size

def test_hooks_entity_spans(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    prompt = "Test prompt for hooks"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Let's say entity is tokens at index 1 and 2
    entity_spans = [[1, 2]]
    
    cache = RepresentationCache()
    layers_to_hook = [3, 14, 22]
    handles = register_hooks(model, layers_to_hook, cache, entity_spans=entity_spans)
    
    with torch.no_grad():
        model(**inputs)
        
    for handle in handles:
        handle.remove()
        
    assert len(cache.cache) == 3
    for layer in layers_to_hook:
        assert layer in cache.cache
        assert cache.cache[layer].dim() == 2 # (batch, hidden)
        assert cache.cache[layer].size(0) == 1
        assert cache.cache[layer].size(1) == model.config.hidden_size
