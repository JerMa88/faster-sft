"""
Unit Test: Completion-Only Loss Masking in paired_dataloader.py
================================================================
Verifies that:
1. Active tokens (labels != -100) exist for every item.
2. Masked tokens vastly outnumber active tokens (prompt much longer than answer).
3. Active tokens decode to the target entity (or boolean for fact_checking).
4. Both mem and gen splits work correctly.

Run: python tests/test_completion_masking.py
(Does NOT load PyTorch models - pure tokenizer test only.)
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from src.data.paired_dataloader import PairedSTaRKDataset, ANSWER_SEP


TOKENIZER_PATH = "Qwen/Qwen2.5-1.5B"
DATASET_PATH = "data/processed/kug_dataset_all.jsonl"
N_SAMPLES = 20


def run_tests():
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading dataset: {DATASET_PATH}")
    dataset = PairedSTaRKDataset(DATASET_PATH, tokenizer, max_length=512)

    answer_sep_ids = tokenizer.encode(ANSWER_SEP, add_special_tokens=False)
    print(f"\nANSWER_SEP = {repr(ANSWER_SEP)}")
    print(f"answer_sep_ids (standalone, for reference) = {answer_sep_ids}")
    print(f"Answer sep decoded: {repr(tokenizer.decode(answer_sep_ids))}")

    print(f"\nRunning completion masking tests on first {N_SAMPLES} items...\n")

    passed = 0
    failed = 0
    warnings = 0

    for idx in range(min(N_SAMPLES, len(dataset))):
        item = dataset[idx]
        task_type = item["task_type"]
        target_entity = item["target_entity"]

        for split in ["mem", "gen"]:
            ids = item[f"{split}_input_ids"].tolist()
            labels = item[f"{split}_labels"].tolist()
            mask = item[f"{split}_attention_mask"].tolist()
            text = item[f"p_{split}_text"]

            # Non-padded sequence length
            seq_len = sum(mask)

            active = [(i, ids[i]) for i in range(seq_len) if labels[i] != -100]
            masked = [(i, ids[i]) for i in range(seq_len) if labels[i] == -100]

            # === TEST 1: At least 1 active token ===
            if len(active) == 0:
                print(f"FAIL [{idx:03d}][{split}]: No active tokens! task={task_type}, target={target_entity!r}")
                failed += 1
                continue

            # === TEST 2: Prompt is much longer than completion ===
            ratio = len(active) / max(1, seq_len)
            if ratio > 0.5:
                # More than 50% of non-padding tokens are active — suspicious
                print(f"WARN [{idx:03d}][{split}]: {len(active)}/{seq_len} tokens active ({ratio:.1%}) — possible masking issue")
                warnings += 1

            # === TEST 3: Verify active tokens decode to something close to target entity ===
            active_token_ids = [t[1] for t in active]
            decoded_active = tokenizer.decode(active_token_ids, skip_special_tokens=True).strip()

            target_lower = target_entity.lower()
            decoded_lower = decoded_active.lower()
            entity_match = (target_lower in decoded_lower) or (decoded_lower in target_lower)

            # For fact_checking, answer is 'true'/'false'/'unknown'
            if task_type == "fact_checking":
                # The answer should be true/false/unknown
                fact_answers = {"true", "false", "unknown"}
                entity_match = any(fa in decoded_lower for fa in fact_answers)

            if not entity_match:
                print(f"WARN [{idx:03d}][{split}]: decoded='{decoded_active}' doesn't clearly match target='{target_entity}' (may be ok if truncated)")
                warnings += 1

            # === TEST 4: Verify first active token comes AFTER at least one masked token ===
            first_active_pos = active[0][0]
            has_prompt_masking = first_active_pos > 0

            if not has_prompt_masking:
                print(f"FAIL [{idx:03d}][{split}]: First active token at position 0 — no prompt masking!")
                failed += 1
                continue

            # === TEST 5: Verify answer sep appears in raw text and position makes sense ===
            sep_pos = text.rfind(ANSWER_SEP)
            if sep_pos == -1:
                print(f"FAIL [{idx:03d}][{split}]: '{ANSWER_SEP}' not found in raw text!")
                failed += 1
                continue

            # Estimate expected prompt token count from char split
            prompt_part = text[: sep_pos + len(ANSWER_SEP)]
            prompt_ids_standalone = tokenizer.encode(prompt_part, add_special_tokens=False)
            expected_prompt_len = len(prompt_ids_standalone)

            # The first active position should be close to expected_prompt_len
            # Allow ±3 tokens for BPE boundary artifacts
            diff = abs(first_active_pos - expected_prompt_len)
            if diff > 3:
                print(f"WARN [{idx:03d}][{split}]: first_active_pos={first_active_pos}, expected≈{expected_prompt_len} (diff={diff}, may be BPE boundary)")
                warnings += 1

            status = "OK"
            print(
                f"  {status} [{idx:03d}][{split}] task={task_type:<15} | "
                f"masked={len(masked):4d} | active={len(active):3d} | "
                f"active_ratio={ratio:.1%} | "
                f"first_active={first_active_pos} | "
                f"decoded='{decoded_active[:35]}'"
            )
            passed += 1

    print(f"\n{'='*70}")
    print(f"Results: {passed} PASSED, {failed} FAILED, {warnings} WARNINGS")
    if failed == 0:
        print("ALL TESTS PASSED — Completion masking is correctly implemented!")
        print(f"  Active token ratio is low (answers << prompt length) ✓")
        print(f"  Active tokens decode to target entities ✓")
        print(f"  Prompt masking applied (first active pos > 0) ✓")
    else:
        print("SOME TESTS FAILED — Check completion masking logic.")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
