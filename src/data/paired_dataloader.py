import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class PairedSTaRKDataset(Dataset):
    """
    Dataset for KUG experiments providing paired (P_mem, P_gen) samples 
    with task disaggregation (chaining, intersection, fact_checking) 
    and entity span indexing for head-entity E_head.
    """
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _find_entity_span(self, token_ids: list, entity_ids: list) -> list:
        """Find start and end indices of entity_ids inside token_ids sequence."""
        if not entity_ids:
            return [0, 0]
        seq_len = len(entity_ids)
        for i in range(len(token_ids) - seq_len + 1):
            if token_ids[i:i + seq_len] == entity_ids:
                return [i, i + seq_len]
        
        # Fallback: search for first token of entity
        first_id = entity_ids[0]
        for i in range(len(token_ids)):
            if token_ids[i] == first_id:
                return [i, min(len(token_ids), i + seq_len)]
        return [0, max(1, seq_len)]

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        item_id = item.get('id', f"item_{idx}")
        domain = item.get('domain', 'STaRK-PRIME')
        task_type = item.get('task_type', 'chaining')
        
        doc = item.get('document', '')
        query = item.get('query', '')
        target_entity = item.get('target_entity', '')
        head_entity = item.get('head_entity', target_entity)
        bridge_entity = item.get('bridge_entity', '')

        p_mem_text = item.get('p_mem', f"Context: {doc}\nQuery: What entity does this describe?\nAnswer: {target_entity}")
        p_gen_text = item.get('p_gen', f"Query: {query}\nAnswer: {target_entity}")

        mem_enc = self.tokenizer(
            p_mem_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        gen_enc = self.tokenizer(
            p_gen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        head_enc = self.tokenizer(head_entity, add_special_tokens=False)
        head_ids = head_enc.input_ids if head_enc.input_ids else [0]

        target_enc = self.tokenizer(target_entity, add_special_tokens=False)
        target_ids = target_enc.input_ids if target_enc.input_ids else [0]

        mem_ids = mem_enc.input_ids[0].tolist()
        gen_ids = gen_enc.input_ids[0].tolist()

        mem_span = self._find_entity_span(mem_ids, head_ids)
        gen_span = self._find_entity_span(gen_ids, head_ids)

        max_entity_len = 32
        target_ids_padded = target_ids[:max_entity_len] + [-100] * max(0, max_entity_len - len(target_ids))

        return {
            "id": item_id,
            "domain": domain,
            "task_type": task_type,
            "mem_input_ids": mem_enc.input_ids[0],
            "mem_attention_mask": mem_enc.attention_mask[0],
            "gen_input_ids": gen_enc.input_ids[0],
            "gen_attention_mask": gen_enc.attention_mask[0],
            "mem_span": torch.tensor(mem_span, dtype=torch.long),
            "gen_span": torch.tensor(gen_span, dtype=torch.long),
            "target_ids": torch.tensor(target_ids_padded, dtype=torch.long),
            "target_entity": target_entity,
            "head_entity": head_entity,
            "p_mem_text": p_mem_text,
            "p_gen_text": p_gen_text,
        }


def collate_kug_batch(batch: list) -> dict:
    """Custom collate function for KUG dataloader."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if k in ["id", "domain", "task_type", "target_entity", "head_entity", "p_mem_text", "p_gen_text"]:
            collated[k] = [b[k] for b in batch]
        else:
            collated[k] = torch.stack([b[k] for b in batch])
    return collated


def get_kug_dataloader(jsonl_path: str, tokenizer: PreTrainedTokenizer, batch_size: int = 4, max_length: int = 512, shuffle: bool = True) -> DataLoader:
    dataset = PairedSTaRKDataset(jsonl_path, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_kug_batch)
