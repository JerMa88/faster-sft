import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "Qwen/Qwen2.5-1.5B"
ckpt_path = "outputs/kug_overhaul_v2/baseline_qwen2.5-1.5b/checkpoint-epoch-50"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ckpt_path)
model.eval()

# Load fact checking samples from dataset
with open("data/processed/kug_dataset_all.jsonl") as f:
    fc_samples = [json.loads(l) for l in f if json.loads(l).get("task_type") == "fact_checking"][:10]

print("=== INSPECTING RAW MODEL OUTPUTS ON CURRENT FACT CHECKING PROMPTS ===")
for s in fc_samples:
    p_mem = s["p_mem"]
    p_gen = s["p_gen"]
    sep = "\nAnswer:"
    mem_prompt = p_mem[:p_mem.rfind(sep) + len(sep)]
    gen_prompt = p_gen[:p_gen.rfind(sep) + len(sep)]
    gold_mem = p_mem[p_mem.rfind(sep) + len(sep):].strip()
    gold_gen = p_gen[p_gen.rfind(sep) + len(sep):].strip()

    # Generate for mem
    enc_mem = tokenizer(mem_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out_mem = model.generate(**enc_mem, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    pred_mem = tokenizer.decode(out_mem[0][enc_mem.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Generate for gen (Current prompt)
    enc_gen = tokenizer(gen_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out_gen = model.generate(**enc_gen, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    pred_gen = tokenizer.decode(out_gen[0][enc_gen.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Generate for gen with explicit instruction template (Mem2Gen paper template)
    stmt = gen_prompt.split("Verify:")[1].split("\nAnswer:")[0].strip()
    instruct_prompt = f"Decide whether the following statement is true or false, answer with 'true' or 'false' ONLY.\nStatement: {stmt}\nAnswer:"
    enc_inst = tokenizer(instruct_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out_inst = model.generate(**enc_inst, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    pred_inst = tokenizer.decode(out_inst[0][enc_inst.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"\nID: {s['id']} (fc_label={s.get('fc_label')})")
    print(f"  P_mem gold   : {gold_mem}")
    print(f"  P_mem pred   : {pred_mem}")
    print(f"  P_gen gold   : {gold_gen}")
    print(f"  P_gen (curr) : {pred_gen}")
    print(f"  P_gen (inst) : {pred_inst}")

