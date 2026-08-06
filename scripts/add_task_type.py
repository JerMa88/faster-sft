"""
scripts/add_task_type.py
========================
Migrates existing JSONL dataset files to include a `task_type` field
('chaining' or 'intersection') derived from the `meta_path` field
in the corresponding v2 files.

Strategy
--------
- v2 files (stark_prime_qa_v2.jsonl, stark_mag_qa_v2.jsonl) already contain
  a `meta_path` field per item, stored as a list of characters that form
  relation-bracketed strings like '[AssociatedGene]' or '[IsA] [HasA]'.
- v1 and v2 files are index-aligned (same 1000 items in the same order).
- Task type is derived from hop count:
    * Single-hop path (one '[...]' bracket group) → 'intersection'
    * Multi-hop path  (two or more bracket groups) → 'chaining'
  This maps to the paper's taxonomy:
    - Chaining: sequential multi-hop reasoning over a bridge entity
    - Intersection: single-hop multi-constraint filtering

The script writes `task_type` into ALL four JSONL files:
    data/processed/stark_prime_qa.jsonl
    data/processed/stark_prime_qa_v2.jsonl
    data/processed/stark_mag_qa.jsonl
    data/processed/stark_mag_qa_v2.jsonl

Idempotent: re-running the script on already-migrated files is safe.

Usage
-----
    python scripts/add_task_type.py [--dry_run] [--data_dir data/processed]

Output
------
Prints a per-file summary table and sample spot-checks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _join_meta_path(raw) -> str:
    """
    Normalise meta_path regardless of storage format.

    The field may be stored as:
      - A string  → '[AssociatedGene] [HasSymptom]'
      - A list of strings (relation tokens)  → ['[AssociatedGene]', '[HasSymptom]']
      - A list of chars (bug in prepare_data.py) → ['[', 'A', 's', ...]
    """
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        joined = "".join(raw).strip()
        return joined
    return str(raw).strip()


def infer_task_type(meta_path_raw) -> str:
    """
    Derive task_type from a meta_path value.

    Counting bracket groups  '[...]'  in the normalised string:
      1 group  → 'intersection'  (single-hop, multi-constraint filtering)
      ≥2 groups → 'chaining'     (multi-hop sequential reasoning)
    """
    path_str = _join_meta_path(meta_path_raw)
    hop_count = len(re.findall(r'\[.*?\]', path_str))
    return "chaining" if hop_count >= 2 else "intersection"


# ---------------------------------------------------------------------------
# File-pair processing
# ---------------------------------------------------------------------------

def migrate_pair(
    v1_path: Path,
    v2_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Reads v2 meta_path values, derives task_type for each item, then writes
    task_type into BOTH v1 and v2 files.

    Returns a summary dict with counts.
    """
    if not v2_path.exists():
        print(f"  [SKIP] v2 file not found: {v2_path}")
        return {}

    # Load v2 (has meta_path)
    v2_items: list[dict] = []
    with open(v2_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v2_items.append(json.loads(line))

    # Derive task_type for every item
    task_types = [infer_task_type(item.get("meta_path", "")) for item in v2_items]

    counts = {"chaining": task_types.count("chaining"),
              "intersection": task_types.count("intersection"),
              "total": len(task_types)}

    # --- Write v2 (always has meta_path) ---
    if not dry_run:
        _write_with_task_type(v2_path, v2_items, task_types)
        print(f"  [OK] {v2_path.name}: "
              f"chaining={counts['chaining']}, intersection={counts['intersection']}")
    else:
        print(f"  [DRY-RUN] {v2_path.name}: "
              f"would write chaining={counts['chaining']}, intersection={counts['intersection']}")

    # --- Write v1 (same items, no meta_path) ---
    if v1_path.exists():
        v1_items: list[dict] = []
        with open(v1_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v1_items.append(json.loads(line))

        if len(v1_items) != len(task_types):
            print(f"  [ERROR] Length mismatch: v1={len(v1_items)} v2={len(task_types)} "
                  f"— skipping v1 write for {v1_path.name}")
        else:
            if not dry_run:
                _write_with_task_type(v1_path, v1_items, task_types)
                print(f"  [OK] {v1_path.name}: task_type written from v2 alignment")
            else:
                print(f"  [DRY-RUN] {v1_path.name}: would write from v2 alignment")
    else:
        print(f"  [SKIP] v1 file not found: {v1_path}")

    return counts


def _write_with_task_type(path: Path, items: list[dict], task_types: list[str]) -> None:
    """Writes items back to path with task_type inserted as the 2nd field for readability."""
    with open(path, "w", encoding="utf-8") as f:
        for item, tt in zip(items, task_types):
            # Insert task_type right after target_entity (if present) or at start
            updated = {}
            for k, v in item.items():
                updated[k] = v
                if k == "target_entity":
                    updated["task_type"] = tt  # insert here for readability
            if "task_type" not in updated:
                updated["task_type"] = tt       # fallback: append
            f.write(json.dumps(updated, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Spot-check printer
# ---------------------------------------------------------------------------

def print_spot_checks(data_path: Path, n: int = 10) -> None:
    """Prints n random items for manual verification."""
    if not data_path.exists():
        return
    import random
    random.seed(0)
    with open(data_path, "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    sample = random.sample(items, min(n, len(items)))
    print(f"\n  Spot-check ({data_path.name}, n={len(sample)}):")
    print(f"  {'task_type':<14}  {'meta_path (if present)':<35}  query[:80]")
    print(f"  {'-'*14}  {'-'*35}  {'-'*80}")
    for item in sample:
        tt   = item.get("task_type", "MISSING")
        mp   = _join_meta_path(item.get("meta_path", "")) if "meta_path" in item else "(none)"
        q    = item.get("query", "")[:80]
        print(f"  {tt:<14}  {mp:<35}  {q}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_file(path: Path) -> bool:
    """Checks every item has a valid task_type. Returns True if all pass."""
    if not path.exists():
        return True  # nothing to validate
    errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [VALIDATE ERROR] line {i+1}: JSON parse error: {e}")
                errors += 1
                continue
            tt = item.get("task_type")
            if tt not in ("chaining", "intersection"):
                print(f"  [VALIDATE ERROR] line {i+1}: invalid task_type={tt!r}")
                errors += 1
    if errors == 0:
        print(f"  [VALIDATE OK] {path.name}: all items have valid task_type")
    else:
        print(f"  [VALIDATE FAIL] {path.name}: {errors} error(s)")
    return errors == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add task_type field to JSONL dataset files."
    )
    parser.add_argument(
        "--data_dir",
        default=str(ROOT / "data" / "processed"),
        help="Directory containing the JSONL files (default: data/processed/)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] data_dir not found: {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  add_task_type.py — JSONL Migration")
    print(f"  data_dir = {data_dir}")
    print(f"  dry_run  = {args.dry_run}")
    print("=" * 60)

    pairs = [
        (data_dir / "stark_prime_qa.jsonl",  data_dir / "stark_prime_qa_v2.jsonl"),
        (data_dir / "stark_mag_qa.jsonl",    data_dir / "stark_mag_qa_v2.jsonl"),
    ]

    all_ok = True
    for v1_path, v2_path in pairs:
        print(f"\nProcessing: {v2_path.stem} → {v1_path.stem}")
        migrate_pair(v1_path, v2_path, dry_run=args.dry_run)

    if not args.dry_run:
        print("\n" + "=" * 60)
        print("  Validation")
        print("=" * 60)
        for v1_path, v2_path in pairs:
            ok_v1 = validate_file(v1_path)
            ok_v2 = validate_file(v2_path)
            all_ok = all_ok and ok_v1 and ok_v2

        print("\n" + "=" * 60)
        print("  Spot-checks")
        print("=" * 60)
        for v1_path, _ in pairs:
            print_spot_checks(v1_path, n=10)

        if all_ok:
            print("\n[SUCCESS] All files validated. task_type migration complete.")
        else:
            print("\n[FAILURE] Some files failed validation. Check errors above.")
            sys.exit(1)
    else:
        print("\n[DRY-RUN complete] No files were modified.")


if __name__ == "__main__":
    main()
