"""
STaRK Data Preparation Script
================================
Downloads STaRK-Prime and STaRK-MAG from HuggingFace (snap-stanford/stark)
and builds memorization/generalization QA pairs following arXiv:2607.08393
Appendix A.

Usage:
    python scripts/prepare_data.py --dataset prime --num_facts 1000
    python scripts/prepare_data.py --dataset mag   --num_facts 1000
    python scripts/prepare_data.py --dataset both  --num_facts 1000
"""

import os
import sys
import json
import random
import pickle
import zipfile
import argparse
import tempfile
from pathlib import Path

# Redirect HF cache before importing datasets/transformers
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ["HF_HOME"]            = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"]  = HF_CACHE

from datasets import load_dataset
from huggingface_hub import hf_hub_download


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configurations
# ─────────────────────────────────────────────────────────────────────────────

STARK_CONFIGS = {
    "prime": {
        "hf_name":     "snap-stanford/stark",
        "hf_config":   "STaRK-Prime",   # actual HF config name (case-sensitive)
        "skb_repo":    "snap-stanford/stark",
        "skb_file":    "skb/prime/processed.zip",
        "output_file": "data/processed/stark_prime_qa.jsonl",
        "label":       "STaRK-Prime",
    },
    "mag": {
        "hf_name":     "snap-stanford/stark",
        "hf_config":   "STaRK-MAG",     # actual HF config name (case-sensitive)
        "skb_repo":    "snap-stanford/stark",
        "skb_file":    "skb/mag/processed.zip",
        "output_file": "data/processed/stark_mag_qa.jsonl",
        "label":       "STaRK-MAG",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SKB (Knowledge-Base) node info loader
# ─────────────────────────────────────────────────────────────────────────────

def load_node_info(dataset_name: str) -> dict:
    """Download and cache the SKB processed.zip, return node_info dict."""
    cfg         = STARK_CONFIGS[dataset_name]
    raw_dir     = ROOT / "data" / "raw" / f"stark_{dataset_name}_skb"
    node_pkl    = raw_dir / "node_info.pkl"

    if node_pkl.exists():
        print(f"  SKB node_info already extracted: {node_pkl}")
        with open(node_pkl, "rb") as f:
            return pickle.load(f)

    print(f"  Downloading SKB for {cfg['label']} …")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download the processed.zip from the HF dataset repo (LFS-tracked)
    zip_path = hf_hub_download(
        repo_id=cfg["skb_repo"],
        filename=cfg["skb_file"],
        repo_type="dataset",
        cache_dir=HF_CACHE,
        local_dir=str(raw_dir),
    )
    print(f"  Downloaded → {zip_path}")

    # Extract
    extract_dir = raw_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract_dir))

    # The pkl might be directly in extract_dir or in a sub-folder
    candidates = list(extract_dir.rglob("node_info.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"node_info.pkl not found under {extract_dir}. "
            f"Contents: {list(extract_dir.rglob('*'))[:20]}"
        )
    src_pkl = candidates[0]
    import shutil
    shutil.copy(src_pkl, node_pkl)

    with open(node_pkl, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Node text serialiser
# ─────────────────────────────────────────────────────────────────────────────

def node_to_doc(node: dict) -> str:
    """Convert a node_info entry into a short textual document."""
    name    = node.get("name", "Unknown")
    type_   = node.get("type", "Entity")
    details = node.get("details", {}) or {}
    lines   = [f"Entity: {name}", f"Type: {type_}"]
    for k, v in details.items():
        if str(k).startswith("_"):
            continue
        v_str = str(v)[:200]   # truncate long field values
        lines.append(f"{k}: {v_str}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# QA pair builder
# ─────────────────────────────────────────────────────────────────────────────

def build_qa_pairs(dataset_name: str, num_facts: int = 1000,
                   seed: int = 42) -> list[dict]:
    """
    Build (P_mem, P_gen, entity) triplets from a STaRK config.

    P_mem: "Context: <doc>\nQuery: What entity does this describe?\nAnswer: <entity>"
    P_gen: "<original QA query>\nAnswer: <entity>"

    Both share the same target_entity, so entity spans are identifiable.
    """
    random.seed(seed)
    cfg = STARK_CONFIGS[dataset_name]

    # ── Load QA queries ───────────────────────────────────────────────────────
    print(f"\n  Loading {cfg['label']} QA queries from HuggingFace …")
    try:
        ds = load_dataset(
            cfg["hf_name"],
            cfg["hf_config"],
            cache_dir=HF_CACHE,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  [ERROR] Could not load dataset '{cfg['hf_name']}' config "
              f"'{cfg['hf_config']}': {e}")
        raise

    # Print available splits for diagnostics
    print(f"  Available splits: {list(ds.keys())}")
    # Pick all splits in order of preference
    priority_splits = [
        "train", "synthesized_all_split", "val", "test",
        "humen_generated_eval", "human_generated_eval",
    ]
    raw_items = []
    for split in priority_splits:
        if split in ds:
            raw_items.extend(list(ds[split]))
    if not raw_items:
        # Fallback: just take whatever is there
        for split in ds:
            raw_items.extend(list(ds[split]))

    random.shuffle(raw_items)
    print(f"  Total raw QA items available: {len(raw_items)}")

    # ── Load node info ─────────────────────────────────────────────────────────
    node_info = load_node_info(dataset_name)
    print(f"  SKB nodes loaded: {len(node_info)}")

    # ── Determine field names dynamically ─────────────────────────────────────
    if raw_items:
        sample = raw_items[0]
        print(f"  Sample item keys: {list(sample.keys())}")
        # Map to canonical names
        query_key  = next((k for k in ["query", "question", "text"] if k in sample), None)
        ans_key    = next((k for k in ["answer_ids", "answer_id", "answers",
                                       "answer", "label"] if k in sample), None)
        print(f"  Using query_key='{query_key}', ans_key='{ans_key}'")
    else:
        raise RuntimeError("No items found in any split of the dataset.")

    # ── Build universal lookup (forward + inverted) ────────────────────────────
    # Diagnostic: show what key format node_info actually uses
    if node_info:
        sample_keys = list(node_info.keys())[:3]
        first_val   = node_info[sample_keys[0]]
        print(f"  node_info first 3 keys : {sample_keys} (type={type(sample_keys[0]).__name__})")
        if isinstance(first_val, dict):
            print(f"  node_info sample val   : { {k: v for k, v in list(first_val.items())[:4]} }")
        # Build lookup that covers: sequential int key, sequential str key,
        # external-id int (stored in node['id']), external-id str.
        forward  = {}   # sequential key → node
        inverted = {}   # external id   → node
        for k, v in node_info.items():
            forward[k]      = v
            forward[str(k)] = v          # str version of sequential key
            if isinstance(v, dict) and "id" in v:
                ext_id = v["id"]
                inverted[ext_id]      = v
                inverted[str(ext_id)] = v
                if isinstance(ext_id, str) and ext_id.isdigit():
                    inverted[int(ext_id)] = v
    else:
        forward = inverted = {}

    def _lookup_node(a_id):
        """Try all four ID-format combinations to find a node."""
        for key in [a_id,
                    str(a_id),
                    int(a_id) if isinstance(a_id, str) and a_id.isdigit() else None]:
            if key is None:
                continue
            n = forward.get(key) or inverted.get(key)
            if n and isinstance(n, dict) and n.get("name"):
                return n
        return None

    # ── Build pairs ────────────────────────────────────────────────────────────
    qa_pairs      = []
    skipped       = 0
    diag_printed  = False   # print one failed-lookup diagnostic

    for item in raw_items:
        if len(qa_pairs) >= num_facts:
            break

        query = item.get(query_key, "") if query_key else ""
        if not query:
            skipped += 1
            continue

        # Resolve answer entity
        ans_raw = item.get(ans_key) if ans_key else None
        if isinstance(ans_raw, (list, tuple)):
            ans_ids = [a for a in ans_raw if a is not None]
        elif ans_raw is not None:
            ans_ids = [ans_raw]
        else:
            ans_ids = []

        # Find the first answer entity
        entity_name = None
        node        = None
        for a_id in ans_ids:
            node = _lookup_node(a_id)
            if node:
                entity_name = node["name"].strip()
                break

        # One-time diagnostic if all lookups fail
        if not entity_name and ans_ids and not diag_printed:
            print(f"  [DIAG] First failed answer_ids: {ans_ids[:3]} "
                  f"(type={type(ans_ids[0]).__name__}). "
                  f"Checking forward keys sample: {sample_keys[:3]}")
            diag_printed = True

        if not entity_name:
            skipped += 1
            continue

        doc = node_to_doc(node)

        # Memorization prompt: recall the entity from its own document
        p_mem = (
            f"Context: {doc}\n"
            f"Query: What entity does this describe?\n"
            f"Answer: {entity_name}"
        )
        # Generalization prompt: answer the original knowledge-graph QA query
        p_gen = (
            f"Query: {query}\n"
            f"Answer: {entity_name}"
        )

        qa_pairs.append({
            "document":      doc,
            "query":         query,
            "target_entity": entity_name,
            # Memorization / generalization text for direct use
            "p_mem":         p_mem,
            "p_gen":         p_gen,
        })

    print(f"  Built {len(qa_pairs)} pairs  (skipped {skipped} for missing entity/query)")
    return qa_pairs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build STaRK QA pairs")
    parser.add_argument("--dataset", choices=["prime", "mag", "both"],
                        default="both",
                        help="Which STaRK config to download")
    parser.add_argument("--num_facts", type=int, default=1000,
                        help="Number of QA pairs to build per dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    targets = ["prime", "mag"] if args.dataset == "both" else [args.dataset]

    for ds_name in targets:
        cfg = STARK_CONFIGS[ds_name]
        out_path = ROOT / cfg["output_file"]
        print(f"\n{'='*60}")
        print(f"  Processing {cfg['label']} → {out_path}")
        print(f"{'='*60}")

        pairs = build_qa_pairs(ds_name, num_facts=args.num_facts, seed=args.seed)

        with open(out_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        print(f"  ✓ Saved {len(pairs)} pairs → {out_path}")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
