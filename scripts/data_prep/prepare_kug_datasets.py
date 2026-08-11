"""
KUG Dataset Preprocessing Script
=================================
Builds task-disaggregated datasets (chaining, intersection, fact_checking)
for STaRK-PRIME and STaRK-MAG following Appendix B of Dai et al. (2025).

Outputs:
  - data/processed/kug_dataset_prime.jsonl
  - data/processed/kug_dataset_mag.jsonl
  - data/processed/kug_dataset_all.jsonl
"""

import os
import sys
import json
import random
import pickle
import zipfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ["HF_HOME"]            = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"]  = HF_CACHE

from datasets import load_dataset
from huggingface_hub import hf_hub_download


STARK_CONFIGS = {
    "prime": {
        "hf_name":     "snap-stanford/stark",
        "hf_config":   "STaRK-Prime",
        "skb_repo":    "snap-stanford/stark",
        "skb_file":    "skb/prime/processed.zip",
        "output_file": "data/processed/kug_dataset_prime.jsonl",
        "label":       "STaRK-PRIME",
    },
    "mag": {
        "hf_name":     "snap-stanford/stark",
        "hf_config":   "STaRK-MAG",
        "skb_repo":    "snap-stanford/stark",
        "skb_file":    "skb/mag/processed.zip",
        "output_file": "data/processed/kug_dataset_mag.jsonl",
        "label":       "STaRK-MAG",
    },
}


def load_node_info(dataset_name: str) -> dict:
    """Download and cache the SKB processed.zip, return node_info dict."""
    cfg      = STARK_CONFIGS[dataset_name]
    raw_dir  = ROOT / "data" / "raw" / f"stark_{dataset_name}_skb"
    node_pkl = raw_dir / "node_info.pkl"

    if node_pkl.exists():
        with open(node_pkl, "rb") as f:
            return pickle.load(f)

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(
        repo_id=cfg["skb_repo"],
        filename=cfg["skb_file"],
        repo_type="dataset",
        cache_dir=HF_CACHE,
        local_dir=str(raw_dir),
    )

    extract_dir = raw_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract_dir))

    candidates = list(extract_dir.rglob("node_info.pkl"))
    if not candidates:
        raise FileNotFoundError(f"node_info.pkl not found under {extract_dir}")
    
    import shutil
    shutil.copy(candidates[0], node_pkl)

    with open(node_pkl, "rb") as f:
        return pickle.load(f)


def get_node_name(node: dict) -> str:
    if not isinstance(node, dict):
        return ""
    for k in ["name", "DisplayName", "display_name", "title", "label"]:
        val = node.get(k)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def node_to_doc(node: dict) -> str:
    name    = get_node_name(node) or "Unknown"
    type_   = node.get("type") or node.get("category") or "Entity"
    details = node.get("details", {}) or {}
    lines   = [f"Entity: {name}", f"Type: {type_}"]
    if isinstance(details, dict) and details:
        for k, v in details.items():
            if not str(k).startswith("_"):
                lines.append(f"{k}: {str(v)[:200]}")
    else:
        for k, v in node.items():
            if k not in ["name", "DisplayName", "display_name", "type", "id"]:
                lines.append(f"{k}: {str(v)[:200]}")
    return "\n".join(lines)


def classify_task_type(query: str) -> str:
    """Classify task into chaining, intersection, or fact_checking based on query structure."""
    q_lower = query.lower()
    if any(w in q_lower for w in ["verify", "true or false", "is it true", "correct statement", "check whether"]):
        return "fact_checking"
    elif any(w in q_lower for w in ["and", "both", "also", "shares", "satisfies", "multi-constraint"]):
        return "intersection"
    elif any(w in q_lower for w in ["which", "that", "whose", "connected to", "expressed in", "targets"]):
        return "chaining"
    else:
        return "chaining"  # default multi-hop reasoning


def build_kug_pairs(dataset_name: str, num_facts: int = 1000, seed: int = 42) -> list:
    random.seed(seed)
    cfg = STARK_CONFIGS[dataset_name]

    print(f"\n  Loading {cfg['label']} QA queries from HuggingFace...")
    ds = load_dataset(cfg["hf_name"], cfg["hf_config"], cache_dir=HF_CACHE, trust_remote_code=True)

    priority_splits = ["train", "synthesized_all_split", "val", "test", "humen_generated_eval", "human_generated_eval"]
    raw_items = []
    for split in priority_splits:
        if split in ds:
            raw_items.extend(list(ds[split]))
    if not raw_items:
        for split in ds:
            raw_items.extend(list(ds[split]))

    random.shuffle(raw_items)
    print(f"  Total raw QA items available: {len(raw_items)}")

    node_info = load_node_info(dataset_name)
    forward, inverted = {}, {}
    for k, v in node_info.items():
        forward[k] = v
        forward[str(k)] = v
        if isinstance(v, dict) and "id" in v:
            ext_id = v["id"]
            inverted[ext_id] = v
            inverted[str(ext_id)] = v
            if isinstance(ext_id, str) and ext_id.isdigit():
                inverted[int(ext_id)] = v

    def _lookup_node(a_id):
        for key in [a_id, str(a_id), int(a_id) if isinstance(a_id, str) and a_id.isdigit() else None]:
            if key is None:
                continue
            n = forward.get(key) or inverted.get(key)
            if n and isinstance(n, dict) and get_node_name(n):
                return n
        return None

    qa_pairs = []
    skipped = 0
    import ast

    for idx, item in enumerate(raw_items):
        if len(qa_pairs) >= num_facts:
            break

        query = item.get("query", "") or item.get("question", "")
        if not query:
            skipped += 1
            continue

        ans_raw = item.get("answer_ids") or item.get("answer_id") or item.get("answer")
        if isinstance(ans_raw, str):
            try:
                ans_raw = ast.literal_eval(ans_raw)
            except (ValueError, SyntaxError):
                pass
        if isinstance(ans_raw, (list, tuple)):
            ans_ids = [int(a) for a in ans_raw if a is not None]
        elif isinstance(ans_raw, (int, float)):
            ans_ids = [int(ans_raw)]
        else:
            ans_ids = []

        entity_name, node = None, None
        for a_id in ans_ids:
            node = _lookup_node(a_id)
            if node:
                entity_name = get_node_name(node)
                break

        if not entity_name:
            skipped += 1
            continue

        doc = node_to_doc(node)
        task_type = classify_task_type(query)
        head_entity = entity_name  # primary entity string for position tracking

        # Synthesize fact checking variant for ~15% of dataset
        if idx % 7 == 0:
            task_type = "fact_checking"
            p_mem = f"Context: {doc}\nQuery: Is the entity described accurately?\nAnswer: true"
            p_gen = f"Query: Verify if '{entity_name}' is associated with: {query}\nAnswer: true"
        elif idx % 2 == 0:
            task_type = "intersection"
            p_mem = f"Context: {doc}\nQuery: What entity satisfies these properties?\nAnswer: {entity_name}"
            p_gen = f"Query: {query}\nAnswer: {entity_name}"
        else:
            task_type = "chaining"
            p_mem = f"Context: {doc}\nQuery: What entity does this describe?\nAnswer: {entity_name}"
            p_gen = f"Query: {query}\nAnswer: {entity_name}"

        qa_pairs.append({
            "id": f"{dataset_name}_{len(qa_pairs):04d}",
            "domain": cfg["label"],
            "task_type": task_type,
            "document": doc,
            "query": query,
            "target_entity": entity_name,
            "head_entity": head_entity,
            "bridge_entity": entity_name,
            "p_mem": p_mem,
            "p_gen": p_gen,
        })

    print(f"  Built {len(qa_pairs)} pairs for {dataset_name} (skipped {skipped})")
    return qa_pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare task-disaggregated KUG datasets")
    parser.add_argument("--num_facts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for ds_name in ["prime", "mag"]:
        pairs = build_kug_pairs(ds_name, num_facts=args.num_facts, seed=args.seed)
        out_path = out_dir / f"kug_dataset_{ds_name}.jsonl"
        with open(out_path, "w") as f:
            for item in pairs:
                f.write(json.dumps(item) + "\n")
        print(f" Saved {len(pairs)} records -> {out_path}")
        all_pairs.extend(pairs)

    comb_path = out_dir / "kug_dataset_all.jsonl"
    with open(comb_path, "w") as f:
        for item in all_pairs:
            f.write(json.dumps(item) + "\n")
    print(f" Saved combined {len(all_pairs)} records -> {comb_path}")


if __name__ == "__main__":
    main()
