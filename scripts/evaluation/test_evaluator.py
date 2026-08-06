"""Quick test: evaluate one checkpoint with the fixed evaluator (20 examples)."""
import os, sys, json
ROOT = "/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
sys.path.insert(0, ROOT)
os.environ["HF_HOME"] = f"{ROOT}/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

import torch
from src.evaluation.evaluator import _load_model, _run_generative_eval
from src.evaluation.metrics import string_accuracy, string_exact_match

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

ckpt_path = f"{ROOT}/outputs/runs/llama3.2-3b/stark_prime/meta-llama--Llama-3.2-3B-Instruct_baseline_lam0.0_seed42/checkpoint_epoch50"
base_id = "meta-llama/Llama-3.2-3B-Instruct"
data_path = f"{ROOT}/data/processed/stark_prime_qa.jsonl"
hf_cache = f"{ROOT}/hf_cache"

# Only test on first 20 examples for speed
data = []
with open(data_path) as f:
    for i, line in enumerate(f):
        if i >= 20:
            break
        data.append(json.loads(line))

# Write temp file with 20 examples
tmp_data = f"{ROOT}/data/processed/test_eval_20.jsonl"
with open(tmp_data, "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")

print(f"Loading model from checkpoint_epoch50...")
model, tokenizer = _load_model(ckpt_path, base_id, device, dtype, hf_cache)

print("Running mem eval (20 examples)...")
mem_preds, mem_targets = _run_generative_eval(model, tokenizer, tmp_data, device, dtype, kind="mem", batch_size=10)

print("Running gen eval (20 examples)...")
gen_preds, gen_targets = _run_generative_eval(model, tokenizer, tmp_data, device, dtype, kind="gen", batch_size=10)

a_mem = string_accuracy(mem_preds, mem_targets)
a_gen = string_accuracy(gen_preds, gen_targets)

print(f"\n{'='*60}")
print(f"A_mem = {a_mem:.3f}  A_gen = {a_gen:.3f}")
print(f"{'='*60}")
print(f"\nFirst 10 mem predictions:")
for i in range(min(10, len(mem_preds))):
    match = "✓" if string_exact_match(mem_preds[i], mem_targets[i]) else "✗"
    print(f"  [{match}] pred='{mem_preds[i][:60]}' | target='{mem_targets[i][:60]}'")

print(f"\nFirst 10 gen predictions:")
for i in range(min(10, len(gen_preds))):
    match = "✓" if string_exact_match(gen_preds[i], gen_targets[i]) else "✗"
    print(f"  [{match}] pred='{gen_preds[i][:60]}' | target='{gen_targets[i][:60]}'")

# Cleanup temp
os.remove(tmp_data)
