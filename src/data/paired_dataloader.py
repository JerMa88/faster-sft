"""
Paired STaRK Dataset with Completion-Only Loss Masking
=======================================================
For each item provides paired (P_mem, P_gen) tensors where:
  - input_ids: full prompt+answer tokenized with truncation
  - labels:    -100 for ALL prompt tokens (context + query prefix),
               active loss ONLY on answer completion tokens ("Answer: ...")

Key implementation detail:
  We use CHARACTER-LEVEL splitting to find the exact prompt/answer boundary
  rather than token-ID matching (which is unreliable due to BPE context merging).
  The text is split at the LAST occurrence of '\nAnswer:' into:
    prompt_text = text[:sep_start + len('\nAnswer:')]   (masked → -100)
    answer_text = text[sep_start + len('\nAnswer:'):]   (active → trains on)
  We tokenize each part separately to get exact split position in token space.

This replicates the Mem2Gen-71FF DataCollatorForCompletionOnlyLM approach,
enabling A_mem >= 95% by epoch 10-15 as reported in Figure 7.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


# ─── Completion separator — everything AFTER this is trained on ───────────────
ANSWER_SEP = "\nAnswer:"


def _make_completion_only_labels_char_split(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> tuple:
    """
    Tokenize `text` and create completion-only labels using character-level splitting.

    Strategy (robust to BPE context merging):
    1. Find the LAST occurrence of ANSWER_SEP in `text`.
    2. Split into `prompt_part` (everything including the sep) and `answer_part`.
    3. Tokenize each part separately to get their individual lengths.
    4. Tokenize the full text with padding/truncation.
    5. Mask (set to -100) all tokens corresponding to `prompt_part` tokens.

    Returns:
        (input_ids, attention_mask, labels) — all as 1-D torch.Tensor of length max_length.
    """
    sep = ANSWER_SEP

    # Find the last occurrence of the separator in raw text
    sep_char_pos = text.rfind(sep)

    if sep_char_pos == -1:
        # No separator found — mask everything (train on nothing)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc.input_ids[0]
        mask = enc.attention_mask[0]
        labels = torch.full_like(ids, -100)
        return ids, mask, labels

    # Split at the separator boundary (include sep in prompt portion so we
    # don't train on the separator tokens themselves, only the answer)
    prompt_part = text[: sep_char_pos + len(sep)]  # "...Query: ...\nAnswer:"
    answer_part = text[sep_char_pos + len(sep):]    # " invasive breast carcinoma"

    # Tokenize prompt part alone (no padding, no truncation) to measure prefix length
    prompt_ids = tokenizer.encode(prompt_part, add_special_tokens=False)
    n_prompt_tokens = len(prompt_ids)

    # Tokenize full text with padding and truncation
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc.input_ids[0]      # (max_length,)
    attention_mask = enc.attention_mask[0]  # (max_length,)

    # Build labels: -100 for prompt tokens, copy input_ids for answer tokens
    labels = input_ids.clone()
    # Mask prompt tokens (first n_prompt_tokens positions)
    # Cap at actual sequence length in case of truncation
    mask_until = min(n_prompt_tokens, input_ids.shape[0])
    labels[:mask_until] = -100
    # Also mask all padding positions
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels


def format_thinking_prompt_and_completion(item: dict, is_mem: bool = True) -> tuple[str, str, str]:
    """
    Format (query_prompt, thought_completion, full_text) with coherent factual thinking traces
    preserving the model's native reasoning capability without corrupting its internal thinking manifold.
    """
    target_entity = item.get("target_entity", "").strip()
    task_type = item.get("task_type", "chaining")
    raw_text = item.get("p_mem" if is_mem else "p_gen", "")

    sep = ANSWER_SEP
    sep_pos = raw_text.rfind(sep)
    query_prompt = raw_text[:sep_pos].strip() if sep_pos != -1 else raw_text.strip()

    if is_mem:
        thought_completion = (
            f"<think>\n"
            f"Retrieving factual knowledge for query: {query_prompt}\n"
            f"Factual recall: The target entity is '{target_entity}'.\n"
            f"</think>\n"
            f"Answer: {target_entity}"
        )
    else:
        if task_type == "chaining" and item.get("chain_hops"):
            hops_str = "\n".join([f"Step {i+1}: {hop}" for i, hop in enumerate(item["chain_hops"])])
            thought_completion = (
                f"<think>\n"
                f"Multi-hop reasoning path:\n{hops_str}\n"
                f"Therefore, the final target entity is '{target_entity}'.\n"
                f"</think>\n"
                f"Answer: {target_entity}"
            )
        elif task_type == "intersection" and item.get("all_heads"):
            heads_str = ", ".join(item["all_heads"])
            thought_completion = (
                f"<think>\n"
                f"Identifying common entity intersecting all heads: {heads_str}.\n"
                f"Intersection result: '{target_entity}'.\n"
                f"</think>\n"
                f"Answer: {target_entity}"
            )
        else:
            thought_completion = (
                f"<think>\n"
                f"Verifying relational facts for query: {query_prompt}\n"
                f"Result: '{target_entity}'.\n"
                f"</think>\n"
                f"Answer: {target_entity}"
            )

    full_text = f"{query_prompt}\n{thought_completion}"
    return query_prompt, thought_completion, full_text


class PairedSTaRKDataset(Dataset):
    """
    Dataset for KUG experiments providing paired (P_mem, P_gen) samples
    with task disaggregation (chaining, intersection, fact_checking)
    and completion-only loss masking for fast A_mem convergence.

    Supports both standard direct format and native thinking trace format
    (for reasoning models like Qwen/Qwen3.5-2B).
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        use_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_thinking = use_thinking
        self.data = []
        n_filtered_no_sep = 0
        n_filtered_too_long = 0

        raw_records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append(json.loads(line))

        for item in raw_records:
            p_mem = item.get("p_mem", "")
            sep_pos = p_mem.rfind(ANSWER_SEP)
            if sep_pos == -1:
                n_filtered_no_sep += 1
                continue
            # Quick prefix-length check (tokenize prompt only, no padding)
            prefix_text = p_mem[: sep_pos + len(ANSWER_SEP)]
            prefix_len = len(tokenizer.encode(prefix_text, add_special_tokens=False))
            if prefix_len >= max_length:
                n_filtered_too_long += 1
                continue
            self.data.append(item)

        total_raw = len(raw_records)
        kept = len(self.data)
        print(
            f"[PairedSTaRKDataset] Loaded {kept}/{total_raw} samples "
            f"(filtered {n_filtered_too_long} too-long, {n_filtered_no_sep} missing sep, use_thinking={use_thinking})"
        )

    def __len__(self) -> int:
        return len(self.data)

    def _find_entity_span(self, token_ids: list, entity_ids: list) -> list:
        """Find start and end indices of entity_ids inside token_ids sequence."""
        if not entity_ids:
            return [0, 0]
        seq_len = len(entity_ids)
        for i in range(len(token_ids) - seq_len + 1):
            if token_ids[i : i + seq_len] == entity_ids:
                return [i, i + seq_len]
        # Fallback: first token of entity
        first_id = entity_ids[0]
        for i in range(len(token_ids)):
            if token_ids[i] == first_id:
                return [i, min(len(token_ids), i + seq_len)]
        return [0, max(1, seq_len)]

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        item_id = item.get("id", f"item_{idx}")
        domain = item.get("domain", "STaRK-PRIME")
        task_type = item.get("task_type", "chaining")

        target_entity = item.get("target_entity", "")
        head_entity = item.get("head_entity", target_entity)

        if self.use_thinking:
            mem_prompt, _, p_mem_text = format_thinking_prompt_and_completion(item, is_mem=True)
            gen_prompt, _, p_gen_text = format_thinking_prompt_and_completion(item, is_mem=False)
            p_mem_prompt = f"{mem_prompt}\n<think>\n"
            p_gen_prompt = f"{gen_prompt}\n<think>\n"
        else:
            p_mem_text = item.get(
                "p_mem",
                f"Context: {item.get('document', '')}\nQuery: What entity does this describe?\nAnswer: {target_entity}",
            )
            p_gen_text = item.get(
                "p_gen",
                f"Query: {item.get('query', '')}\nAnswer: {target_entity}",
            )
            sep = ANSWER_SEP
            mem_sep_pos = p_mem_text.rfind(sep)
            gen_sep_pos = p_gen_text.rfind(sep)
            p_mem_prompt = p_mem_text[: mem_sep_pos + len(sep)] if mem_sep_pos != -1 else p_mem_text
            p_gen_prompt = p_gen_text[: gen_sep_pos + len(sep)] if gen_sep_pos != -1 else p_gen_text

        # Completion-only labels via character-level splitting
        mem_ids, mem_mask, mem_labels = _make_completion_only_labels_char_split(
            p_mem_text, self.tokenizer, self.max_length
        )
        gen_ids, gen_mask, gen_labels = _make_completion_only_labels_char_split(
            p_gen_text, self.tokenizer, self.max_length
        )

        # Entity span (for Patchscope / KCR routing)
        head_enc = self.tokenizer(head_entity, add_special_tokens=False)
        head_token_ids = head_enc.input_ids if head_enc.input_ids else [0]

        target_enc = self.tokenizer(target_entity, add_special_tokens=False)
        target_token_ids = target_enc.input_ids if target_enc.input_ids else [0]

        mem_span = self._find_entity_span(mem_ids.tolist(), head_token_ids)
        gen_span = self._find_entity_span(gen_ids.tolist(), head_token_ids)

        max_entity_len = 32
        target_ids_padded = (
            target_token_ids[:max_entity_len]
            + [-100] * max(0, max_entity_len - len(target_token_ids))
        )

        return {
            "id": item_id,
            "domain": domain,
            "task_type": task_type,
            "mem_input_ids": mem_ids,
            "mem_attention_mask": mem_mask,
            "mem_labels": mem_labels,
            "gen_input_ids": gen_ids,
            "gen_attention_mask": gen_mask,
            "gen_labels": gen_labels,
            "mem_span": torch.tensor(mem_span, dtype=torch.long),
            "gen_span": torch.tensor(gen_span, dtype=torch.long),
            "target_ids": torch.tensor(target_ids_padded, dtype=torch.long),
            "target_entity": target_entity,
            "head_entity": head_entity,
            "p_mem_text": p_mem_text,          # Full text (prompt + answer) — for training labels
            "p_gen_text": p_gen_text,           # Full text (prompt + answer) — for training labels
            "p_mem_prompt": p_mem_prompt,       # Prompt ONLY (no answer) — for model.generate() eval
            "p_gen_prompt": p_gen_prompt,       # Prompt ONLY (no answer) — for model.generate() eval
        }


def collate_kug_batch(batch: list) -> dict:
    """Custom collate function for KUG dataloader."""
    str_keys = {"id", "domain", "task_type", "target_entity", "head_entity",
                "p_mem_text", "p_gen_text", "p_mem_prompt", "p_gen_prompt"}
    collated = {}
    for k in batch[0].keys():
        if k in str_keys:
            collated[k] = [b[k] for b in batch]
        else:
            collated[k] = torch.stack([b[k] for b in batch])
    return collated


def get_kug_dataloader(
    jsonl_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    shuffle: bool = True,
    use_thinking: bool = False,
) -> DataLoader:
    dataset = PairedSTaRKDataset(jsonl_path, tokenizer, max_length=max_length, use_thinking=use_thinking)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_kug_batch,
        num_workers=0,
        pin_memory=True,
    )
