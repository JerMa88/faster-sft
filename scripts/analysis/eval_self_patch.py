import os
import sys
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.hooks import _get_layer
from src.data.paired_dataloader import PairedSTaRKDataset

def get_intervention_hook(layer_idx: int, inject_hidden_state: torch.Tensor, position: int):
    """
    Returns a forward hook that injects a specific hidden state at a given sequence position.
    inject_hidden_state: (D,) tensor
    """
    def hook(module, inputs, outputs):
        if isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        # hidden_states: (B, seq_len, D)
        # Inject at the specified position for all items in batch
        # We only support batch_size = 1 for simple evaluation
        hidden_states[:, position, :] = inject_hidden_state.to(hidden_states.dtype)

        if isinstance(outputs, (tuple, list)):
            return (hidden_states,) + outputs[1:]
        else:
            return hidden_states
    return hook

def run_oracle_self_patch(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading tokenizer {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.eval()
    
    dataset = PairedSTaRKDataset(args.data_path, tokenizer, max_length=512)
    
    correct = 0
    total = min(args.limit, len(dataset)) if args.limit > 0 else len(dataset)
    
    # We will do Oracle Self-Patching:
    # 1. Run P_mem to get h_target at l_s_late
    # 2. Run P_gen with intervention at l_t (inject h_target at the last token position)
    # 3. Check if the generated next token matches the target entity
    
    with torch.no_grad():
        for i in tqdm(range(total)):
            item = dataset[i]
            
            mem_ids = item["mem_input_ids"].unsqueeze(0).to(device)
            gen_ids = item["gen_input_ids"].unsqueeze(0).to(device)
            
            # Find the actual unpadded length
            mem_len = (item["mem_attention_mask"] == 1).sum().item()
            gen_len = (item["gen_attention_mask"] == 1).sum().item()
            
            mem_ids = mem_ids[:, :mem_len]
            gen_ids = gen_ids[:, :gen_len]
            
            # target span in P_mem
            start_idx, end_idx = item["mem_span"]
            if start_idx == -1: continue # invalid
            
            # 1. Forward P_mem and extract h at l_s_late
            layer_s_late = _get_layer(model, args.l_s_late)
            h_extract = []
            def extract_hook(module, inputs, outputs):
                hs = outputs[0] if isinstance(outputs, tuple) else outputs
                # Mean pool over the target entity span
                h_extract.append(hs[0, start_idx:end_idx, :].mean(dim=0))
            
            handle1 = layer_s_late.register_forward_hook(extract_hook)
            model(mem_ids)
            handle1.remove()
            
            if not h_extract: continue
            h_target = h_extract[0]
            
            # 2. Forward P_gen with intervention at l_t
            # We inject at the last token of the query prompt
            inject_pos = gen_len - 1
            layer_t = _get_layer(model, args.l_t)
            handle2 = layer_t.register_forward_hook(get_intervention_hook(args.l_t, h_target, inject_pos))
            
            outputs = model(gen_ids)
            handle2.remove()
            
            logits = outputs.logits[0, -1, :]
            pred_token = logits.argmax().item()
            
            tgt_ids = item["target_ids"]
            valid_tgt_ids = tgt_ids[tgt_ids != -100]
            if len(valid_tgt_ids) == 0: continue
            
            if pred_token == valid_tgt_ids[0].item():
                correct += 1
                
    acc = correct / total if total > 0 else 0
    print(f"Oracle Self-Patching Accuracy: {acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--data_path", type=str, default="data/processed/stark_prime_qa_v2.jsonl")
    parser.add_argument("--l_s_late", type=int, default=24)
    parser.add_argument("--l_t", type=int, default=13)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    run_oracle_self_patch(args)
