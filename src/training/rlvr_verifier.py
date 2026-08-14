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


def compute_verifiable_reward(completion: str, target_entity: str, task_type: str) -> float:
    """
    Main entry point for computing verifiable RLVR reward R in {0.0, 1.0}.
    
    Args:
        completion: The generated model response text.
        target_entity: Ground truth entity or boolean label.
        task_type: 'chaining', 'intersection', or 'fact_checking'.
    
    Returns:
        1.0 if candidate is verifiably correct, 0.0 otherwise.
    """
    if task_type == "fact_checking":
        return verify_fact_checking(completion, target_entity)
    else:
        return verify_entity_target(completion, target_entity)
