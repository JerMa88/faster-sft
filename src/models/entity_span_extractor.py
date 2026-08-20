"""
Entity Token Span Extractor for Knowledge-Circuit Routing
Finds token span [start_idx, end_idx) corresponding to an entity string mention
within tokenized input sequences, with full support for left-padded or right-padded batches.
"""

from typing import List, Optional, Tuple
import torch


def find_token_span_for_substring(
    full_text: str,
    substring: str,
    token_offsets: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """
    Given full_text, a target substring (e.g. head_entity), and list of
    (char_start, char_end) offsets for each token, find the token index span [tok_start, tok_end).
    """
    if not substring or not full_text:
        return None

    # Find character position in full text (case-insensitive fallback)
    char_start = full_text.find(substring)
    if char_start == -1:
        char_start = full_text.lower().find(substring.lower())
    if char_start == -1:
        return None

    char_end = char_start + len(substring)

    tok_start = None
    tok_end = None

    for idx, (t_start, t_end) in enumerate(token_offsets):
        # Check overlap between token char span [t_start, t_end) and substring [char_start, char_end)
        if t_end > char_start and t_start < char_end:
            if tok_start is None:
                tok_start = idx
            tok_end = idx + 1

    if tok_start is not None and tok_end is not None:
        return (tok_start, tok_end)
    return None


def extract_entity_token_mask(
    input_ids_len: int,
    span: Optional[Tuple[int, int]],
    pad_offset: int = 0,
) -> List[float]:
    """
    Creates a normalized mask vector of length input_ids_len where tokens in span are active.
    pad_offset shifts the span index (e.g. for left-padded batch inputs).
    If span is None, returns uniform mask across non-padded tokens.
    """
    mask = [0.0] * input_ids_len
    if span is None or span[1] <= span[0]:
        # Fallback to uniform over non-padded tokens
        valid_start = max(0, pad_offset)
        valid_len = max(1, input_ids_len - valid_start)
        for i in range(valid_start, input_ids_len):
            mask[i] = 1.0 / valid_len
        return mask

    s = max(0, span[0] + pad_offset)
    e = min(input_ids_len, span[1] + pad_offset)
    span_len = max(1, e - s)

    for i in range(s, e):
        mask[i] = 1.0 / span_len
    return mask
