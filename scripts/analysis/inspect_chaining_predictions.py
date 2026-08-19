"""
Diagnostic script to inspect Chaining predictions, error patterns, and failure modes.
Runs on GPU via SLURM.
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.training.rlvr_verifier import verify_entity_target, normalize_text

def main():
    base_model_path = "Qwen/Qwen2.5-1.5B"
    ckpt_dir = "outputs/kug_overhaul_v2/two_stage_breadcrumb_rlvr_qwen2.5-1.5b/checkpoint-epoch-50"
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
    
    # Sample 50 chaining items for inspection
    import random
    random.seed(42)
    sample_items = random.sample(chaining_items, min(50, len(chaining_items)))
    
    categories = {
        "exact_target_match": 0,
        "bridge_entity_match": 0,
        "hop_entity_match": 0,
        "head_entity_match": 0,
        "hallucination_or_other": 0,
    }
    
    inspection_logs = []
    
    sep = "\nAnswer:"
    for idx, item in enumerate(sample_items):
        p_gen = item["p_gen"]
        sep_pos = p_gen.rfind(sep)
        prompt = p_gen[: sep_pos + len(sep)] if sep_pos != -1 else p_gen
        
        target = item.get("target_entity", "")
        bridge = item.get("bridge_entity", "")
        head = item.get("head_entity", "")
        hops = item.get("chain_hops", [])
        
        enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(out_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Categorize prediction
        cat = "hallucination_or_other"
        if verify_entity_target(pred, target) > 0.0:
            cat = "exact_target_match"
        elif bridge and verify_entity_target(pred, bridge) > 0.0:
            cat = "bridge_entity_match"
        elif head and verify_entity_target(pred, head) > 0.0:
            cat = "head_entity_match"
        else:
            # Check intermediate hops
            for hop in hops:
                parts = hop.split("--[")
                for p in parts:
                    clean_ent = p.split("]-->")[-1].strip()
                    if clean_ent and verify_entity_target(pred, clean_ent) > 0.0:
                        cat = "hop_entity_match"
                        break
                        
        categories[cat] += 1
        
        inspection_logs.append({
            "id": item.get("id"),
            "prompt": prompt,
            "prediction": pred,
            "target": target,
            "bridge": bridge,
            "head": head,
            "category": cat
        })
        
    print("\n=== CHAINING PREDICTION ERROR ANALYSIS (N=50) ===")
    for k, v in categories.items():
        print(f"  {k:25s}: {v:2d} / 50 ({v/50*100:5.1f}%)")
        
    out_json = "outputs/metrics/chaining_error_analysis.json"
    with open(out_json, "w") as f:
        json.dump({"summary": categories, "samples": inspection_logs}, f, indent=2)
    print(f"\nDetailed sample logs written to {out_json}")

if __name__ == "__main__":
    main()
