"""
test_pipeline.py
----------------
Unit tests that verify the pipeline logic WITHOUT making any real API calls.

Run with:
    python -m pytest tests/test_pipeline.py -v
    # or simply
    python tests/test_pipeline.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ---- Make sure the project root is on the path ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_client import extract_label, extract_thinking, JudgeResponse
from src.templates import build_prompt, TEMPLATES, CRITERIA
from src.dataset import PairRecord, DIFFICULTY_LEVELS
from src.metrics import compute_position_bias, compute_pairwise_agreement, compute_thinking_accuracy


# ===========================================================================
# Label extraction tests
# ===========================================================================

def test_extract_label_bare():
    assert extract_label("The better response is A.") == ("A", True)
    assert extract_label("B") == ("B", True)
    assert extract_label("Both are equally good. C") == ("C", True)

def test_extract_label_verdict_line():
    text = "Response A is more helpful.\nVerdict: B"
    label, ok = extract_label(text)
    assert label == "B" and ok

def test_extract_label_with_thinking():
    text = "<think>A is better because it is more detailed.</think>\nB"
    label, ok = extract_label(text)
    assert label == "B" and ok

def test_extract_label_fallback():
    text = "I cannot determine a clear winner."
    label, ok = extract_label(text)
    # No A/B/C present → should return None with parse failure
    assert label is None and ok is False

def test_extract_thinking():
    text = "<think>step 1\nstep 2</think>\nA"
    thinking = extract_thinking(text)
    assert thinking == "step 1\nstep 2"

def test_extract_thinking_absent():
    assert extract_thinking("Just output A") is None


# ===========================================================================
# Template tests
# ===========================================================================

def test_all_templates_build():
    pair = PairRecord("id1", "What is 2+2?", "4", "5", "A")
    for tmpl_id in TEMPLATES:
        sys_p, usr_p = build_prompt(tmpl_id, pair.prompt, pair.response_a, pair.response_b)
        assert len(sys_p) > 10, f"System prompt too short for {tmpl_id}"
        assert "4" in usr_p and "5" in usr_p, f"Responses not in user prompt for {tmpl_id}"

def test_unknown_template_raises():
    try:
        build_prompt("nonexistent_template", "q", "a", "b")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ===========================================================================
# Dataset tests
# ===========================================================================

def test_pair_record_flip():
    pair = PairRecord("p1", "Question?", "Good answer", "Bad answer", "A")
    flipped = pair.flipped()
    assert flipped.response_a == "Bad answer"
    assert flipped.response_b == "Good answer"
    assert flipped.gold_label == "B"
    assert flipped.prompt_id == "p1_flipped"

def test_pair_record_flip_tie():
    pair = PairRecord("p2", "Q?", "Ans1", "Ans2", "C")
    assert pair.flipped().gold_label == "C"

def test_pair_record_difficulty_field():
    pair = PairRecord("p1", "Q?", "A", "B", "A", difficulty="hard")
    assert pair.difficulty == "hard"
    assert pair.flipped().difficulty == "hard"

def test_pair_record_difficulty_optional():
    pair = PairRecord("p1", "Q?", "A", "B", "A")
    assert pair.difficulty is None

def test_difficulty_levels_constant():
    assert set(DIFFICULTY_LEVELS) == {"easy", "medium", "hard", "impossible"}


# ===========================================================================
# Metrics tests
# ===========================================================================

def test_compute_position_bias_perfect():
    results = [
        {"prompt_id": "p1", "condition": "AB", "label": "A", "experiment_id": "x"},
        {"prompt_id": "p1", "condition": "BA", "label": "B", "experiment_id": "x"},  # consistent
        {"prompt_id": "p2", "condition": "AB", "label": "B", "experiment_id": "x"},
        {"prompt_id": "p2", "condition": "BA", "label": "A", "experiment_id": "x"},  # consistent
    ]
    m = compute_position_bias(results)
    assert m["position_consistency"] == 1.0
    assert m["position_bias_rate"] == 0.0

def test_compute_position_bias_full_bias():
    results = [
        {"prompt_id": "p1", "condition": "AB", "label": "A", "experiment_id": "x"},
        {"prompt_id": "p1", "condition": "BA", "label": "A", "experiment_id": "x"},  # bias!
    ]
    m = compute_position_bias(results)
    assert m["position_consistency"] == 0.0
    assert m["position_bias_rate"] == 1.0
    assert m["bias_toward_first_position"] == 1.0

def test_compute_position_bias_accuracy():
    # Gold: p1 -> A (better response is A), p2 -> A
    gold = {"p1": "A", "p2": "A"}
    results = [
        {"prompt_id": "p1", "condition": "AB", "label": "A"},  # correct
        {"prompt_id": "p1", "condition": "BA", "label": "B"},  # correct (gold flips to B in BA)
        {"prompt_id": "p2", "condition": "AB", "label": "A"},  # correct
        {"prompt_id": "p2", "condition": "BA", "label": "A"},  # wrong (should be B)
    ]
    m = compute_position_bias(results, gold_labels=gold)
    assert m["accuracy"]["ab_accuracy"] == 1.0
    assert m["accuracy"]["ba_accuracy"] == 0.5
    assert m["accuracy"]["accuracy_gap"] == 0.5

def test_compute_position_bias_no_gold_no_accuracy():
    results = [
        {"prompt_id": "p1", "condition": "AB", "label": "A"},
        {"prompt_id": "p1", "condition": "BA", "label": "B"},
    ]
    m = compute_position_bias(results)
    assert "accuracy" not in m

def test_compute_pairwise_agreement_perfect():
    results = [
        {"prompt_id": "p1", "condition": "tmpl1", "label": "A"},
        {"prompt_id": "p1", "condition": "tmpl2", "label": "A"},
        {"prompt_id": "p2", "condition": "tmpl1", "label": "B"},
        {"prompt_id": "p2", "condition": "tmpl2", "label": "B"},
    ]
    m = compute_pairwise_agreement(results)
    assert m["overall_pairwise_agreement"] == 1.0

def test_compute_pairwise_agreement_zero():
    results = [
        {"prompt_id": "p1", "condition": "tmpl1", "label": "A"},
        {"prompt_id": "p1", "condition": "tmpl2", "label": "B"},
    ]
    m = compute_pairwise_agreement(results)
    assert m["overall_pairwise_agreement"] == 0.0

def test_compute_thinking_accuracy():
    gold = {"p1": "A", "p2": "B"}
    results = [
        {"prompt_id": "p1", "condition": "no_reasoning", "label": "A", "latency_s": 1.0},
        {"prompt_id": "p2", "condition": "no_reasoning", "label": "B", "latency_s": 1.0},
        {"prompt_id": "p1", "condition": "think_512",   "label": "A", "latency_s": 2.0},
        {"prompt_id": "p2", "condition": "think_512",   "label": "A", "latency_s": 2.0},  # wrong
    ]
    m = compute_thinking_accuracy(results, gold)
    assert m["accuracy_vs_gold"]["no_reasoning"] == 1.0
    assert m["accuracy_vs_gold"]["think_512"] == 0.5


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    tests = [
        test_extract_label_bare,
        test_extract_label_verdict_line,
        test_extract_label_with_thinking,
        test_extract_label_fallback,
        test_extract_thinking,
        test_extract_thinking_absent,
        test_all_templates_build,
        test_unknown_template_raises,
        test_pair_record_flip,
        test_pair_record_flip_tie,
        test_pair_record_difficulty_field,
        test_pair_record_difficulty_optional,
        test_difficulty_levels_constant,
        test_compute_position_bias_perfect,
        test_compute_position_bias_full_bias,
        test_compute_position_bias_accuracy,
        test_compute_position_bias_no_gold_no_accuracy,
        test_compute_pairwise_agreement_perfect,
        test_compute_pairwise_agreement_zero,
        test_compute_thinking_accuracy,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
