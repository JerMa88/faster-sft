import os
import socket
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_load_qwen_1_5b():
    model_id = "Qwen/Qwen2.5-1.5B"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Model loaded successfully.")
    
    # Test greedy decode
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Output: {output_text}")
    assert "Paris" in output_text, f"Expected 'Paris' in output, got: {output_text}"
    print("Inference successful.")

if __name__ == "__main__":
    test_load_qwen_1_5b()
