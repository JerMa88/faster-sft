"""
RLVR Verifier Module for KUG Benchmark
=======================================
Provides automated, verifiable reward evaluation for candidate LLM rollouts
without leaking answers into training prompts.

Tasks Supported:
  1. chaining:       Multi-hop entity query -> Exact/normalized target entity match.
  2. intersection:   Parallel entity query  -> Exact/normalized target entity match.
  3. fact_checking:  Binary verification    -> Boolean ('true'/'false') extraction and match.
"""

import re
import string


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, articles, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_answer_text(completion: str) -> str:
    """Extract candidate text after the last '\\nAnswer:' separator if present."""
    sep = "\nAnswer:"
    pos = completion.rfind(sep)
    if pos != -1:
        return completion[pos + len(sep):].strip()
    
    # Fallback for completions that start directly with "Answer:"
    if completion.startswith("Answer:"):
        return completion[len("Answer:"):].strip()

    return completion.strip()


def verify_entity_target(completion: str, target_entity: str) -> float:
    """
    Verify candidate entity completion against ground truth target entity.
    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    ans = extract_answer_text(completion)
    norm_ans = normalize_text(ans)
    norm_tgt = normalize_text(target_entity)

    if not norm_ans or not norm_tgt:
        return 0.0

    # 1. Exact normalized match
    if norm_ans == norm_tgt:
        return 1.0

    # 2. Leading/contained match if answer starts with target or matches target boundary
    # e.g., target="invasive breast carcinoma", answer="invasive breast carcinoma."
    if norm_ans.startswith(norm_tgt) or norm_tgt.startswith(norm_ans):
        # Prevent trivial single-letter match
        if min(len(norm_ans), len(norm_tgt)) >= 3:
            return 1.0

    # 3. Exact word boundary substring
    if re.search(r"\b" + re.escape(norm_tgt) + r"\b", norm_ans):
        return 1.0

    return 0.0


def verify_fact_checking(completion: str, target_label: str) -> float:
    """
    Verify candidate fact checking completion against ground truth boolean ('true' or 'false').
    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    ans = extract_answer_text(completion).lower().strip()
    norm_tgt = target_label.lower().strip()

    # If target_label is not a valid boolean, normalize or fallback
    if norm_tgt not in ["true", "false"]:
        if "true" in norm_tgt:
            norm_tgt = "true"
        elif "false" in norm_tgt:
            norm_tgt = "false"

    # Match first valid boolean token
    pred_bool = None
    first_word = ans.split()[0] if ans.split() else ""
    first_word = first_word.rstrip(".,;!?")

    if first_word in ["true", "false"]:
        pred_bool = first_word
    else:
        # Search for explicit 'true' or 'false'
        has_true = bool(re.search(r"\btrue\b", ans))
        has_false = bool(re.search(r"\bfalse\b", ans))
        if has_true and not has_false:
            pred_bool = "true"
        elif has_false and not has_true:
            pred_bool = "false"
        else:
            pred_bool = None

    if pred_bool is not None and pred_bool == norm_tgt:
        return 1.0

    return 0.0


def compute_verifiable_reward(
    completion: str,
    target_entity: str,
    task_type: str,
    bridge_entity: str = "",
    chain_hops: list = None,
    fc_label: str = ""
) -> float:
    """
    Main entry point for computing verifiable RLVR reward with step-wise breadcrumbs.
    
    Reward Tiers:
      - Chaining:
          * Target Entity (Final Hop) Match:       R = 1.00
          * Bridge Entity (Hop 1/2 Intermediate):  R = 0.50
          * Intermediate Entity in chain_hops:     R = 0.25
          * Off-path / Hallucination:              R = 0.00
      - Intersection:
          * Target Entity Match:                   R = 1.00
          * Mismatch:                              R = 0.00
      - Fact Checking:
          * Correct Boolean (fc_label):            R = 1.00
          * Incorrect Boolean:                     R = 0.00
    """
    if task_type == "fact_checking":
        label = fc_label if fc_label else target_entity
        return verify_fact_checking(completion, label)

    # 1. Check Full Target Match (1.00)
    if verify_entity_target(completion, target_entity) > 0.0:
        return 1.0

    # 2. For Chaining, check Step-Wise Breadcrumb Rewards
    if task_type == "chaining":
        # 2a. Bridge Entity Match (0.50)
        if bridge_entity and verify_entity_target(completion, bridge_entity) > 0.0:
            return 0.50

        # 2b. Intermediate Chain Hop Entity Match (0.25)
        if chain_hops:
            for hop in chain_hops:
                # Hop format: "Entity1 --[rel]--> Entity2"
                parts = re.split(r"\s+--\[.*?\]-->\s+", hop)
                for ent in parts:
                    ent = ent.strip()
                    if ent and verify_entity_target(completion, ent) > 0.0:
                        return 0.25

    return 0.0
