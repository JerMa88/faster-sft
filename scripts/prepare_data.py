import os
import socket
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

import urllib.request
import zipfile
import pickle
import json
from datasets import load_dataset

def generate_qa_pairs_prime(num_facts=1000):
    print("Loading STaRK-Prime dataset...")
    ds = load_dataset("snap-stanford/stark", "STaRK-Prime")
    
    # Download the SKB (Knowledge Base) node info
    skb_url = "https://huggingface.co/datasets/snap-stanford/stark/resolve/main/skb/prime/processed.zip"
    os.makedirs("data/raw", exist_ok=True)
    zip_path = "data/raw/prime_processed.zip"
    
    if not os.path.exists(zip_path):
        print("Downloading Prime SKB processed zip using curl...")
        import subprocess
        subprocess.run(["curl", "-4", "-L", "-o", zip_path, skb_url], check=True)
    
    extracted_dir = "data/raw/prime_processed"
    if not os.path.exists(extracted_dir):
        print("Extracting Prime SKB processed zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
            
    print("Loading node info...")
    with open(os.path.join(extracted_dir, "node_info.pkl"), 'rb') as f:
        node_info = pickle.load(f)
        
    qa_pairs = []
    
    for split in ["synthesized_all_split", "humen_generated_eval"]:
        if split not in ds:
            continue
            
        print(f"Processing split: {split}")
        for item in ds[split]:
            query = item["query"]
            ans_ids = item["answer_ids"]
            
            if not ans_ids:
                continue
                
            ans_id = ans_ids[0]  # Just take the first valid answer entity
            if ans_id in node_info:
                node = node_info[ans_id]
                name = node.get("name", "")
                type_ = node.get("type", "")
                details = node.get("details", {})
                
                # Construct a simple document text representation of the entity
                doc = f"Entity: {name}\nType: {type_}\n"
                for k, v in details.items():
                    if not str(k).startswith("_"):
                        doc += f"{k}: {v}\n"
                        
                qa_pairs.append({
                    "document": doc.strip(),
                    "query": query,
                    "target_entity": name
                })
                
            if len(qa_pairs) >= num_facts:
                break
                
        if len(qa_pairs) >= num_facts:
            break
            
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/stark_prime_qa.jsonl"
    with open(out_path, "w") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + "\n")
            
    print(f"Saved {len(qa_pairs)} QA pairs to {out_path}")

if __name__ == "__main__":
    generate_qa_pairs_prime(1000)
