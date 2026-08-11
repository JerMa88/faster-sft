import json
import torch
import unittest
from unittest.mock import MagicMock
from src.data.paired_dataloader import PairedSTaRKDataset, collate_kug_batch

class TestPairedDataloader(unittest.TestCase):
    def test_dataloader_structure(self):
        mock_tokenizer = MagicMock()
        mock_enc = MagicMock()
        mock_enc.input_ids = torch.ones((1, 32), dtype=torch.long)
        mock_enc.attention_mask = torch.ones((1, 32), dtype=torch.long)
        mock_tokenizer.side_effect = lambda text, **kwargs: mock_enc
        
        mock_head_enc = MagicMock()
        mock_head_enc.input_ids = [10, 20]
        mock_tokenizer.side_effect = lambda text, **kwargs: mock_head_enc if kwargs.get('add_special_tokens') == False else mock_enc

        dataset = PairedSTaRKDataset('data/processed/kug_dataset_prime.jsonl', mock_tokenizer, max_length=64)
        self.assertGreater(len(dataset), 0)
        
        sample = dataset[0]
        self.assertIn('mem_input_ids', sample)
        self.assertIn('gen_input_ids', sample)
        self.assertIn('task_type', sample)
        self.assertIn('mem_span', sample)

        batch = collate_kug_batch([sample, sample])
        self.assertEqual(batch['mem_input_ids'].shape[0], 2)
        self.assertEqual(len(batch['task_type']), 2)

if __name__ == '__main__':
    unittest.main()
