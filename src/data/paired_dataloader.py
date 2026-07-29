import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class PairedSTaRKDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer: PreTrainedTokenizer, max_length=512):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
        
    def _find_entity_span(self, token_ids, entity_ids):
        # The entity is always at the very end of the text, before padding tokens.
        # Find the first padding token (or end of list)
        pad_id = self.tokenizer.pad_token_id
        end_idx = len(token_ids)
        for i in range(len(token_ids)):
            if token_ids[i] == pad_id:
                end_idx = i
                break
        
        # We assume the entity tokens occupy roughly the last len(entity_ids) tokens
        start_idx = max(0, end_idx - len(entity_ids))
        return start_idx, end_idx

    def __getitem__(self, idx):
        item = self.data[idx]
        doc = item['document']
        query = item['query']
        target_entity = item['target_entity']
        
        # P_mem: Document -> Target Entity
        p_mem_text = f"Context: {doc}\nQuery: What entity is this about?\nAnswer: {target_entity}"
        # P_gen: Query -> Target Entity
        p_gen_text = f"Query: {query}\nAnswer: {target_entity}"
        
        mem_enc = self.tokenizer(p_mem_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        gen_enc = self.tokenizer(p_gen_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        
        target_enc = self.tokenizer(target_entity, add_special_tokens=False)
        target_ids = target_enc.input_ids
        
        mem_ids = mem_enc.input_ids[0].tolist()
        gen_ids = gen_enc.input_ids[0].tolist()
        
        mem_span = self._find_entity_span(mem_ids, target_ids)
        gen_span = self._find_entity_span(gen_ids, target_ids)
        
        # Pad target_ids to fixed length for collation
        max_entity_len = 32
        target_ids_padded = target_ids[:max_entity_len] + [-100] * max(0, max_entity_len - len(target_ids))
        
        return {
            "mem_input_ids": mem_enc.input_ids[0],
            "mem_attention_mask": mem_enc.attention_mask[0],
            "gen_input_ids": gen_enc.input_ids[0],
            "gen_attention_mask": gen_enc.attention_mask[0],
            "mem_span": torch.tensor(mem_span),
            "gen_span": torch.tensor(gen_span),
            "target_ids": torch.tensor(target_ids_padded) # y*
        }

def get_dataloader(jsonl_path, tokenizer, batch_size=4, max_length=512, shuffle=True):
    dataset = PairedSTaRKDataset(jsonl_path, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
