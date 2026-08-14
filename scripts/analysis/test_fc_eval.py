import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "Qwen/Qwen2.5-1.5B"
ckpt_path = "outputs/kug_overhaul_v2/baseline_qwen2.5-1.5b/checkpoint-epoch-50"

print("Loading tokenizer and model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ckpt_path)
model.eval()

def relaxed_match(predicted: str, target: str) -> bool:
    p = predicted.strip().lower()
    t = target.strip().lower()
    if not t:
        return False
    return t in p or p in t

with open("data/processed/kug_dataset_all.jsonl") as f:
    fc_samples = [json.loads(l) for l in f if json.loads(l).get("task_type") == "fact_checking"]

print(f"Total fact checking samples in dataset: {len(fc_samples)}", flush=True)

sep = "\nAnswer:"
mem_prompts = []
gen_prompts_curr = []
gen_prompts_inst = []
gold_mems = []
gold_gens = []

for s in fc_samples:
    p_mem = s["p_mem"]
    p_gen = s["p_gen"]
    mem_prompt = p_mem[:p_mem.rfind(sep) + len(sep)]
    gen_prompt_curr = p_gen[:p_gen.rfind(sep) + len(sep)]
    gold_mem = p_mem[p_mem.rfind(sep) + len(sep):].strip()
    gold_gen = p_gen[p_gen.rfind(sep) + len(sep):].strip()
    
    # Instructed format (Mem2Gen paper template)
    stmt = gen_prompt_curr.split("Verify:")[1].split("\nAnswer:")[0].strip()
    instruct_prompt = f"Decide whether the following statement is true or false, answer with 'true' or 'false' ONLY.\nStatement: {stmt}\nAnswer:"

    mem_prompts.append(mem_prompt)
    gen_prompts_curr.append(gen_prompt_curr)
    gen_prompts_inst.append(instruct_prompt)
    gold_mems.append(gold_mem)
    gold_gens.append(gold_gen)

def batch_generate(prompts, max_tokens=32, batch_size=32):
    preds = []
    for i in range(0, len(prompts), batch_size):
        b_prompts = prompts[i:i + batch_size]
        enc = tokenizer(b_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out_ids = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        for j, (in_ids, out) in enumerate(zip(enc.input_ids, out_ids)):
            p_len = in_ids.shape[0]
            pred = tokenizer.decode(out[p_len:], skip_special_tokens=True).strip()
            preds.append(pred)
        print(f"  Generated {len(preds)}/{len(prompts)}...", flush=True)
    return preds

print("1. Running Memorization Generation...", flush=True)
pred_mems = batch_generate(mem_prompts, max_tokens=32, batch_size=32)

print("2. Running Current Prompt Generation (Bare 'Verify:...')...", flush=True)
pred_gens_curr = batch_generate(gen_prompts_curr, max_tokens=32, batch_size=32)

print("3. Running Mem2Gen Instructed Generation ('Decide whether...')...", flush=True)
pred_gens_inst = batch_generate(gen_prompts_inst, max_tokens=16, batch_size=32)

# Compute accuracies
correct_mem = sum(1 for p, g in zip(pred_mems, gold_mems) if relaxed_match(p, g))
correct_gen_curr = sum(1 for p, g in zip(pred_gens_curr, gold_gens) if relaxed_match(p, g))
correct_gen_inst = sum(1 for p, g in zip(pred_gens_inst, gold_gens) if relaxed_match(p, g))

print("\n" + "="*60, flush=True)
print("=== FACT CHECKING EVALUATION RESULTS (N=400) ===", flush=True)
print("="*60, flush=True)
print(f"A_mem (Memorization)                 : {correct_mem / len(fc_samples):.4f} ({correct_mem}/{len(fc_samples)})", flush=True)
print(f"A_gen (Current bare 'Verify:' prompt): {correct_gen_curr / len(fc_samples):.4f} ({correct_gen_curr}/{len(fc_samples)})", flush=True)
print(f"A_gen (Mem2Gen Instructed template)  : {correct_gen_inst / len(fc_samples):.4f} ({correct_gen_inst}/{len(fc_samples)})", flush=True)

# Sample comparisons
print("\n--- First 5 Sample Predictions ---", flush=True)
for k in range(5):
    print(f"\nSample {k+1}: id={fc_samples[k]['id']} (gold={gold_gens[k]})", flush=True)
    print(f"  Mem Gold: {gold_mems[k]} | Pred: {pred_mems[k]}", flush=True)
    print(f"  Curr Prompt Pred: {pred_gens_curr[k]!r}", flush=True)
    print(f"  Inst Prompt Pred: {pred_gens_inst[k]!r}", flush=True)
