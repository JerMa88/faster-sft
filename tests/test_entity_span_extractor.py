"""
Unit tests for Entity Token Span Extractor (Pure Python, Zero PyTorch Model Loading)
"""

import pytest
from src.models.entity_span_extractor import (
    find_token_span_for_substring,
    extract_entity_token_mask,
)


def test_find_token_span_exact():
    text = "What disease has relation with fibrosarcomatous osteosarcoma?"
    entity = "fibrosarcomatous osteosarcoma"
    
    # Mock token offsets: [(0, 4), (5, 12), (13, 16), (17, 25), (26, 30), (31, 47), (48, 60), (60, 61)]
    # "What", "disease", "has", "relation", "with", "fibrosarcomatous", "osteosarcoma", "?"
    offsets = [
        (0, 4), (5, 12), (13, 16), (17, 25), (26, 30),
        (31, 47), (48, 60), (60, 61)
    ]
    
    span = find_token_span_for_substring(text, entity, offsets)
    assert span == (5, 7)  # tokens 5 and 6


def test_find_token_span_case_insensitive():
    text = "What paper does I.S. Kachur write?"
    entity = "i.s. kachur"
    offsets = [(0, 4), (5, 10), (11, 15), (16, 20), (21, 27), (28, 33), (33, 34)]
    
    span = find_token_span_for_substring(text, entity, offsets)
    assert span == (3, 5)


def test_find_token_span_not_found():
    text = "What paper does John Doe write?"
    entity = "Jane Doe"
    offsets = [(0, 4), (5, 10), (11, 15), (16, 20), (21, 24), (25, 30), (30, 31)]
    
    span = find_token_span_for_substring(text, entity, offsets)
    assert span is None


def test_extract_entity_token_mask():
    span = (2, 4)
    mask = extract_entity_token_mask(5, span)
    assert len(mask) == 5
    assert mask[0] == 0.0
    assert mask[1] == 0.0
    assert mask[2] == 0.5
    assert mask[3] == 0.5
    assert mask[4] == 0.0


def test_extract_entity_token_mask_fallback():
    mask = extract_entity_token_mask(4, None)
    assert len(mask) == 4
    assert mask == [0.25, 0.25, 0.25, 0.25]
