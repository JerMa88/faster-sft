import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from src.data.paired_dataloader import get_dataloader
from src.models.hooks import RepresentationCache, register_hooks
from src.training.losses import contrastive_loss, rep_distill_loss
import json

def train(model_id="sapientinc/HRM-Text-1B", data_path="data/processed/synthetic_qa.jsonl", epochs=3, alpha=1.0):
    device = "cpu"  # Assuming cpu for this sandbox env
    print(f"Loading tokenizer and model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, device_map=device)
    
    # Add LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    
    # Layer indices (L=32 for HRM-Text-1B)
    L_t = 10  # Generation / Usage layer
    L_s = 24  # Memorization / Storage layer
    
    cache = RepresentationCache()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    loader = get_dataloader(data_path, tokenizer, batch_size=2)
    
    metrics = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_align_loss = 0
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            
            mem_ids = batch["mem_input_ids"].to(device)
            gen_ids = batch["gen_input_ids"].to(device)
            mem_span = batch["mem_span"]
            gen_span = batch["gen_span"]
            
            # 1. Forward P_mem to get h_mem at L_s
            cache.clear()
            mem_spans = [(int(s[0]), int(s[1])) for s in mem_span]
            handles_mem = register_hooks(model, [L_s], cache, mem_spans)
            with torch.no_grad():
                model(mem_ids)
            for h in handles_mem: h.remove()
            h_mem = cache.cache[L_s] # (B, D)
            
            # 2. Forward P_gen to get h_gen at L_t and CE loss
            cache.clear()
            gen_spans = [(int(s[0]), int(s[1])) for s in gen_span]
            handles_gen = register_hooks(model, [L_t], cache, gen_spans)
            
            # Construct labels for P_gen: -100 for everything except the target span
            labels = gen_ids.clone()
            for b_idx in range(len(gen_span)):
                s_start, s_end = int(gen_span[b_idx][0]), int(gen_span[b_idx][1])
                labels[b_idx, :s_start] = -100
                labels[b_idx, s_end:] = -100
            
            outputs = model(gen_ids, labels=labels)
            ce_loss = outputs.loss
            for h in handles_gen: h.remove()
            
            h_gen = cache.cache[L_t]
            
            # 3. Alignment Loss
            align_loss = contrastive_loss(h_mem, h_gen)
            
            total_loss = ce_loss + alpha * align_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_align_loss += align_loss.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} | Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}, Align: {align_loss.item():.4f})")
                
            if step >= 20: # Fast dev loop
                break
                
        metrics.append({
            "epoch": epoch,
            "total_loss": epoch_loss / (step + 1),
            "ce_loss": epoch_ce_loss / (step + 1),
            "align_loss": epoch_align_loss / (step + 1)
        })
        
    with open("data/processed/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Training finished.")

if __name__ == "__main__":
    train()
