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


def split_cot_completion(completion: str) -> tuple:
    """
    Splits a candidate completion into (thought_text, answer_text).
    Supports formats:
      1. <think> thought </think> Answer: answer
      2. Reasoning: thought \nAnswer: answer
      3. Step 1: thought \nAnswer: answer
      4. Plain answer (thought is empty string)
    """
    if not completion:
        return "", ""

    # Look for Answer: or Final Answer: separator
    match = re.search(r"(?:</think>|\n|^)\s*(?:Answer|Final Answer):\s*", completion, re.IGNORECASE)
    if match:
        thought = completion[: match.start()].strip()
        answer = completion[match.end() :].strip()
        thought = re.sub(r"^<think>\s*", "", thought, flags=re.IGNORECASE).strip()
        return thought, answer

    # Fallback: check for explicit <think> ... </think> tags
    think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL | re.IGNORECASE)
    if think_match:
        thought = think_match.group(1).strip()
        answer = completion[think_match.end() :].strip()
        return thought, answer

    # If no separator found, extract answer from text
    return "", extract_answer_text(completion)


def compute_verifiable_reward(
    completion: str,
    target_entity: str,
    task_type: str,
    bridge_entity: str = "",
    chain_hops: list = None,
    fc_label: str = "",
    epoch: int = 1,
    curriculum_anneal: bool = True
) -> float:
    """
    Main entry point for computing verifiable RLVR reward with Bridge-Penalized Curriculum Annealing
    and 2-Step CoT Scratchpad support.
    
    Curriculum Annealing Schedule (for Chaining):
      - Discovery Phase (Epoch <= 25):
          * Full Target Match:                R = 1.00
          * Bridge in Thought:                R = 0.50
          * Bridge in Answer:                 R = 0.40
          * Intermediate Hop:                 R = 0.20
      - Annealing Phase (Epoch 26..35):
          * Full Target Match:                R = 1.00
          * Bridge in Thought:                R = 0.50 * (35 - epoch) / 10
          * Bridge in Answer:                 R = 0.40 * (35 - epoch) / 10
          * Intermediate Hop:                 R = 0.20 * (35 - epoch) / 10
      - Exploitation & Penalty Phase (Epoch >= 36):
          * Full Target Match:                R = 1.00
          * Bridge in Answer (Stop at Hop 1): R = -0.30  (Active penalty)
          * Bridge in Thought without Target: R = 0.00
          * Hallucination / Off-Path:         R = 0.00
    """
    thought, answer = split_cot_completion(completion)

    if task_type == "fact_checking":
        label = fc_label if fc_label else target_entity
        return verify_fact_checking(answer if answer else completion, label)

    # Intersection evaluation
    if task_type == "intersection":
        target_in_ans = verify_entity_target(answer, target_entity) > 0.0 or verify_entity_target(completion, target_entity) > 0.0
        return 1.0 if target_in_ans else 0.0

    # Chaining (Multi-Hop) evaluation with Curriculum Annealing
    if task_type == "chaining":
        norm_thought = normalize_text(thought)
        norm_bridge = normalize_text(bridge_entity) if bridge_entity else ""
        norm_target = normalize_text(target_entity) if target_entity else ""

        target_in_ans = verify_entity_target(answer, target_entity) > 0.0 or (norm_target and norm_target in normalize_text(answer))
        bridge_in_thought = bool(norm_bridge and norm_thought and (norm_bridge in norm_thought or verify_entity_target(thought, bridge_entity) > 0.0))
        bridge_in_ans = bool(bridge_entity and (verify_entity_target(answer, bridge_entity) > 0.0 or verify_entity_target(completion, bridge_entity) > 0.0))

        # 1. Full Target Match (Always 1.00)
        if target_in_ans:
            return 1.00

        if verify_entity_target(completion, target_entity) > 0.0:
            return 0.80

        # Curriculum schedule for partial / intermediate rewards
        if curriculum_anneal:
            if epoch <= 25:
                # Phase 1: Exploration / Discovery
                if bridge_in_thought:
                    return 0.50
                if bridge_in_ans:
                    return 0.40
                if chain_hops and norm_thought:
                    for hop in chain_hops:
                        parts = re.split(r"\s+--\[.*?\]-->\s+", hop)
                        for ent in parts:
                            clean_ent = normalize_text(ent)
                            if len(clean_ent) >= 3 and clean_ent in norm_thought:
                                return 0.20
            elif 26 <= epoch <= 35:
                # Phase 2: Linear Annealing
                decay = max(0.0, (35.0 - epoch) / 10.0)
                if bridge_in_thought:
                    return 0.50 * decay
                if bridge_in_ans:
                    return 0.40 * decay
                if chain_hops and norm_thought:
                    for hop in chain_hops:
                        parts = re.split(r"\s+--\[.*?\]-->\s+", hop)
                        for ent in parts:
                            clean_ent = normalize_text(ent)
                            if len(clean_ent) >= 3 and clean_ent in norm_thought:
                                return 0.20 * decay
            else:
                # Phase 3 (Epoch >= 36): Active Penalty for stopping at bridge!
                if bridge_in_ans:
                    return -0.30  # Penalize stopping at intermediate hop instead of target!
                return 0.00
        else:
            # Fixed Breadcrumbs without annealing
            if bridge_in_thought:
                return 0.40
            if bridge_in_ans:
                return 0.30

    return 0.0
