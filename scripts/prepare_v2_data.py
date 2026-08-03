import json
import random
import re
import os

def load_data(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def get_stopwords():
    return {"what", "is", "the", "a", "an", "to", "as", "or", "and", "in", "of", "for", "with", "that", "this", "are", "on", "it", "be", "disease", "condition", "related"}

def extract_bridge_entities(query, all_entities):
    # Try to find any known entities in the query
    found = []
    q_lower = query.lower()
    for ent in all_entities:
        if ent.lower() in q_lower:
            found.append(ent)
    
    if not found:
        # Fallback: extract capitalized phrases
        words = query.split()
        cap = [w for w in words if w[0].isupper() and w.lower() not in get_stopwords()]
        if cap:
            found.append(" ".join(cap))
        else:
            # Fallback 2: random noun
            found.append(query.split()[1] if len(query.split()) > 1 else query)
    return found

def get_hard_negative(query, all_entities, current_target, stopwords):
    q_words = set(re.findall(r'\w+', query.lower())) - stopwords
    best_match = None
    best_score = -1
    
    # randomly sample 100 to avoid O(N^2)
    candidates = random.sample(all_entities, min(100, len(all_entities)))
    for cand in candidates:
        if cand == current_target: continue
        c_words = set(re.findall(r'\w+', cand.lower())) - stopwords
        score = len(q_words.intersection(c_words))
        if score > best_score:
            best_score = score
            best_match = cand
    if not best_match:
        best_match = random.choice(candidates)
    return best_match

def extract_meta_path(query):
    q_lower = query.lower()
    path = []
    if "ancestor" in q_lower or "parent" in q_lower or "broader" in q_lower:
        path.append("[IsA]")
    if "descendent" in q_lower or "child" in q_lower or "narrower" in q_lower:
        path.append("[HasA]")
    if "treat" in q_lower or "drug" in q_lower or "therapy" in q_lower:
        path.append("[TreatedBy]")
    if "gene" in q_lower or "protein" in q_lower:
        path.append("[AssociatedGene]")
    if "symptom" in q_lower or "clinical" in q_lower:
        path.append("[HasSymptom]")
    
    if not path:
        path.append("[RelatedTo]")
    
    return " ".join(path)

def process_file(in_path, out_path):
    print(f"Processing {in_path}...")
    data = load_data(in_path)
    all_entities = list(set(d['target_entity'] for d in data))
    stopwords = get_stopwords()
    
    processed = []
    for d in data:
        q = d['query']
        tgt = d['target_entity']
        
        bridge = extract_bridge_entities(q, all_entities)
        bridge = bridge[0] if bridge else ""
        
        hn = get_hard_negative(q, all_entities, tgt, stopwords)
        meta = extract_meta_path(q)
        
        d['bridge_entity'] = bridge
        d['hard_negative'] = hn
        d['meta_path'] = meta
        
        # update p_gen to include meta path prefix
        d['p_gen'] = f"Meta: {meta}\n{d['p_gen']}"
        processed.append(d)
        
    with open(out_path, 'w') as f:
        for d in processed:
            f.write(json.dumps(d) + "\n")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    random.seed(42)
    process_file("data/processed/stark_prime_qa.jsonl", "data/processed/stark_prime_qa_v2.jsonl")
    process_file("data/processed/stark_mag_qa.jsonl", "data/processed/stark_mag_qa_v2.jsonl")
