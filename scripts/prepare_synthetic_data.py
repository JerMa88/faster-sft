import json
import random
import os

def generate_synthetic_stark(num_samples=1000):
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/synthetic_qa.jsonl"
    
    genes = [f"Gene-{random.randint(1000, 9999)}" for _ in range(500)]
    diseases = [f"Disease-{random.randint(100, 999)}" for _ in range(200)]
    drugs = [f"Drug-{random.randint(10000, 99999)}" for _ in range(300)]
    
    qa_pairs = []
    for i in range(num_samples):
        # Create a synthetic fact
        entity_type = random.choice(["gene", "disease", "drug"])
        if entity_type == "gene":
            entity = random.choice(genes)
            target = random.choice(diseases)
            doc = f"Entity: {entity}\nType: Gene\nFunction: It is a critical biomarker and is known to be overexpressed in {target}. It plays a role in cellular metabolism."
            query = f"Which gene is known to be overexpressed in {target} and plays a role in cellular metabolism?"
        elif entity_type == "disease":
            entity = random.choice(diseases)
            target = random.choice(drugs)
            doc = f"Entity: {entity}\nType: Disease\nTreatment: The primary effective treatment for {entity} is {target}, which targets the underlying pathways."
            query = f"What is the disease that is primarily treated by {target}?"
        else:
            entity = random.choice(drugs)
            target = random.choice(genes)
            doc = f"Entity: {entity}\nType: Drug\nMechanism: This compound directly inhibits {target} and is used in modern therapies."
            query = f"Which drug acts by directly inhibiting {target} in modern therapies?"
            
        qa_pairs.append({
            "document": doc,
            "query": query,
            "target_entity": entity
        })
        
    with open(out_path, "w") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + "\n")
            
    print(f"Saved {num_samples} synthetic QA pairs to {out_path}")

if __name__ == "__main__":
    generate_synthetic_stark(2000)
