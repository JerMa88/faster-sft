"""
Unit tests for RLVR Verifier Module
"""

import pytest
from src.training.rlvr_verifier import (
    normalize_text,
    extract_answer_text,
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


def test_compute_verifiable_reward():
    assert compute_verifiable_reward("Tamoxifen", "tamoxifen", "chaining") == 1.0
    assert compute_verifiable_reward("Tamoxifen", "tamoxifen", "intersection") == 1.0
    assert compute_verifiable_reward("True", "true", "fact_checking") == 1.0
    assert compute_verifiable_reward("False", "true", "fact_checking") == 0.0
    assert compute_verifiable_reward("Wrong Entity", "tamoxifen", "chaining") == 0.0


if __name__ == "__main__":
    test_normalize_text()
    test_extract_answer_text()
    test_verify_entity_target()
    test_verify_fact_checking()
    test_compute_verifiable_reward()
    print("All RLVR verifier unit tests passed successfully!")
