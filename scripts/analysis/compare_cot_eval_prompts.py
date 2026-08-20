"""
Diagnostic script to compare Chaining Generalization Accuracy under:
1. Standard SFT Prompt:  `{Query}\nAnswer:`
2. CoT Scratchpad Prompt: `{Query}\nThought: `
Evaluated on checkpoint-epoch-41 of two_stage_cot_rlvr.
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.training.rlvr_verifier import split_cot_completion, verify_entity_target, normalize_text

def relaxed_match(predicted: str, target: str) -> bool:
    p = predicted.strip().lower()
    t = target.strip().lower()
    if not t:
        return False
    return t in p or p in t

def main():
    base_model_path = "Qwen/Qwen2.5-1.5B"
    ckpt_dir = "outputs/kug_overhaul_v2/two_stage_cot_rlvr_qwen2.5-1.5b/checkpoint-epoch-41"
    data_path = "data/processed/kug_dataset_all.jsonl"
    
    print(f"Loading Base Model: {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    
    print(f"Loading Adapter: {ckpt_dir}...")
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model.eval()
    
    # Load Chaining queries
    chaining_items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("task_type") == "chaining":
                    chaining_items.append(item)
                    
    print(f"Loaded {len(chaining_items)} chaining items.")
    
    # Sample 100 chaining items
    import random
    random.seed(42)
    sample_items = random.sample(chaining_items, min(100, len(chaining_items)))
    
    # Evaluate under Format 1: Direct \nAnswer:
    # Evaluate under Format 2: \nThought:
    correct_direct = 0
    correct_cot = 0
    bridge_in_thought_count = 0
    
    sep = "\nAnswer:"
    
    for idx, item in enumerate(sample_items):
        p_gen = item["p_gen"]
        sep_pos = p_gen.rfind(sep)
        query = p_gen[:sep_pos] if sep_pos != -1 else p_gen
        
        target = item.get("target_entity", "")
        bridge = item.get("bridge_entity", "")
        
        # 1. Direct prompt
        prompt_direct = f"{query}\nAnswer:"
        enc1 = tokenizer(prompt_direct, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out1 = model.generate(**enc1, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        pred1 = tokenizer.decode(out1[0][enc1.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 2. CoT prompt
        prompt_cot = f"{query}\nThought: "
        enc2 = tokenizer(prompt_cot, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out2 = model.generate(**enc2, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw_pred2 = tokenizer.decode(out2[0][enc2.input_ids.shape[1]:], skip_special_tokens=True).strip()
        thought2, pred2 = split_cot_completion(raw_pred2)
        
        # Scoring
        if relaxed_match(pred1, target):
            correct_direct += 1
            
        if relaxed_match(pred2, target) or relaxed_match(raw_pred2, target):
            correct_cot += 1
            
        if bridge and (normalize_text(bridge) in normalize_text(thought2) or normalize_text(bridge) in normalize_text(raw_pred2)):
            bridge_in_thought_count += 1
            
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx+1}/100 | Direct Acc: {correct_direct/(idx+1)*100:.1f}% | CoT Acc: {correct_cot/(idx+1)*100:.1f}% | Bridge in Thought: {bridge_in_thought_count/(idx+1)*100:.1f}%")
            
    print("\n=== FINAL COMPARISON RESULTS (N=100) ===")
    print(f"1. Direct Prompt  (\\nAnswer:):   {correct_direct}/100 ({correct_direct}%)")
    print(f"2. CoT Scratchpad (\\nThought: ): {correct_cot}/100 ({correct_cot}%)")
    print(f"3. Bridge Entity in Thought:      {bridge_in_thought_count}/100 ({bridge_in_thought_count}%)")

if __name__ == "__main__":
    main()
