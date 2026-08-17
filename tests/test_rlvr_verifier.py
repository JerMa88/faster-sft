"""
Unit tests for RLVR Verifier Module
"""

import pytest
from src.training.rlvr_verifier import (
    normalize_text,
    extract_answer_text,
    split_cot_completion,
    verify_entity_target,
    verify_fact_checking,
    compute_verifiable_reward,
)


def test_normalize_text():
    assert normalize_text("Invasive Breast Carcinoma.") == "invasive breast carcinoma"
    assert normalize_text("  The COVID-19 Vaccine! ") == "covid19 vaccine"
    assert normalize_text("A metabolic pathway") == "metabolic pathway"


def test_extract_answer_text():
    raw_prompt_and_ans = "Query: What is the entity?\nAnswer: Tamoxifen"
    assert extract_answer_text(raw_prompt_and_ans) == "Tamoxifen"

    completion_only = "Answer: Tamoxifen"
    assert extract_answer_text(completion_only) == "Tamoxifen"

    raw_text = "Tamoxifen."
    assert extract_answer_text(raw_text) == "Tamoxifen."


def test_verify_entity_target():
    target = "invasive breast carcinoma"
    
    # Exact match
    assert verify_entity_target("invasive breast carcinoma", target) == 1.0
    # Match with prefix / suffix punctuation
    assert verify_entity_target(" Invasive Breast Carcinoma. ", target) == 1.0
    assert verify_entity_target("Query: ...\nAnswer: invasive breast carcinoma", target) == 1.0
    
    # Partial word / incorrect
    assert verify_entity_target("breast cancer", target) == 0.0
    assert verify_entity_target("lung adenocarcinoma", target) == 0.0
    assert verify_entity_target("", target) == 0.0


def test_verify_fact_checking():
    # True target
    assert verify_fact_checking("true", "true") == 1.0
    assert verify_fact_checking("True.", "true") == 1.0
    assert verify_fact_checking("Answer: true", "true") == 1.0
    assert verify_fact_checking("The statement is true.", "true") == 1.0
    assert verify_fact_checking("false", "true") == 0.0

    # False target
    assert verify_fact_checking("false", "false") == 1.0
    assert verify_fact_checking("False.", "false") == 1.0
    assert verify_fact_checking("Answer: false", "false") == 1.0
    assert verify_fact_checking("The statement is false.", "false") == 1.0
    assert verify_fact_checking("true", "false") == 0.0


def test_split_cot_completion():
    # 1. <think> format
    c1 = "<think> Step 1: Laser cooling of solids </think>\nAnswer: Resolved sideband cooling"
    t1, a1 = split_cot_completion(c1)
    assert t1 == "Step 1: Laser cooling of solids"
    assert a1 == "Resolved sideband cooling"

    # 2. Reasoning: ... \nAnswer: ...
    c2 = "Reasoning: Amir writes Laser cooling of solids.\nAnswer: Resolved sideband cooling"
    t2, a2 = split_cot_completion(c2)
    assert "Laser cooling of solids" in t2
    assert a2 == "Resolved sideband cooling"

    # 3. Direct Answer only
    c3 = "Answer: Resolved sideband cooling"
    t3, a3 = split_cot_completion(c3)
    assert t3 == ""
    assert a3 == "Resolved sideband cooling"


def test_compute_verifiable_reward():
    # 1. Chaining 2-Step CoT Success (1.00)
    c1 = "<think> Step 1: Laser cooling of solids </think>\nAnswer: Resolved sideband cooling"
    assert compute_verifiable_reward(c1, "Resolved sideband cooling", "chaining", bridge_entity="Laser cooling of solids") == 1.00

    # 2. Chaining Direct Target Match (0.80)
    c2 = "Answer: Resolved sideband cooling"
    assert compute_verifiable_reward(c2, "Resolved sideband cooling", "chaining", bridge_entity="Laser cooling of solids") == 0.80

    # 3. Chaining Step 1 Bridge Solved in Thought, wrong final answer (0.40)
    c3 = "Reasoning: Laser cooling of solids is the intermediate paper.\nAnswer: Some Wrong Target"
    assert compute_verifiable_reward(c3, "Resolved sideband cooling", "chaining", bridge_entity="Laser cooling of solids") == 0.40

    # 4. Chaining stopped at bridge in final answer (0.30)
    c4 = "Answer: Laser cooling of solids"
    assert compute_verifiable_reward(c4, "Resolved sideband cooling", "chaining", bridge_entity="Laser cooling of solids") == 0.30

    # 5. Chaining Off-path Hallucination (0.00)
    c5 = "Reasoning: completely random hallucination\nAnswer: wrong answer"
    assert compute_verifiable_reward(c5, "Resolved sideband cooling", "chaining", bridge_entity="Laser cooling of solids") == 0.00

    # 6. Intersection Exact Match (1.00) vs Mismatch (0.00)
    c_inter = "<think> Finding joint paper </think>\nAnswer: Tamoxifen"
    assert compute_verifiable_reward(c_inter, "tamoxifen", "intersection") == 1.0
    assert compute_verifiable_reward("Answer: Wrong Entity", "tamoxifen", "intersection") == 0.0

    # 7. Fact Checking with fc_label
    c_fc = "<think> Checking relation: expression present </think>\nAnswer: false"
    assert compute_verifiable_reward(c_fc, "Random Entity Name", "fact_checking", fc_label="false") == 1.0
    assert compute_verifiable_reward("Answer: true", "Random Entity Name", "fact_checking", fc_label="false") == 0.0


if __name__ == "__main__":
    test_normalize_text()
    test_extract_answer_text()
    test_verify_entity_target()
    test_verify_fact_checking()
    test_split_cot_completion()
    test_compute_verifiable_reward()
    print("All 2-Step CoT RLVR verifier unit tests passed successfully!")
