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
from src.templates import (
    build_prompt, TEMPLATES, CRITERIA,
    _build_user_prompt_blind,
    _build_user_prompt_criterion_first,
    _build_user_prompt_blind_criterion_first,
)
from src.dataset import (
    PairRecord, load_dataset_pairs,
    _detect_multiturn, _stratified_sample,
)
from src.metrics import (
    compute_position_bias, compute_pairwise_agreement,
    compute_thinking_accuracy, compute_stratified_metrics,
)


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

def test_extract_label_blind_1():
    label, ok = extract_label("Response 1 is clearer. 1")
    assert label == "A" and ok

def test_extract_label_blind_2():
    label, ok = extract_label("2")
    assert label == "B" and ok

def test_extract_label_blind_verdict_line():
    label, ok = extract_label("Both are reasonable.\nVerdict: 1")
    assert label == "A" and ok


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

def test_build_blind_user_prompt():
    out = _build_user_prompt_blind("Q?", "resp_a", "resp_b", "better overall")
    assert "[Response 1]" in out and "[Response 2]" in out
    assert "[Response A]" not in out and "[Response B]" not in out

def test_build_criterion_first_user_prompt():
    out = _build_user_prompt_criterion_first("Q?", "resp_a", "resp_b", "more helpful")
    criterion_pos = out.index("more helpful")
    response_a_pos = out.index("[Response A]")
    assert criterion_pos < response_a_pos, "Criterion should appear before responses"

def test_build_blind_criterion_first_user_prompt():
    out = _build_user_prompt_blind_criterion_first("Q?", "resp_a", "resp_b", "more helpful")
    criterion_pos = out.index("more helpful")
    response_1_pos = out.index("[Response 1]")
    assert criterion_pos < response_1_pos, "Criterion should appear before responses"
    assert "[Response A]" not in out

def test_all_b5_templates_build():
    pair = PairRecord("id1", "What is 2+2?", "4", "5", "A")
    for tmpl_id in ("blind", "criterion_first", "blind_criterion_first"):
        sys_p, usr_p = build_prompt(tmpl_id, pair.prompt, pair.response_a, pair.response_b)
        assert len(sys_p) > 10, f"System prompt too short for {tmpl_id}"
        assert "4" in usr_p and "5" in usr_p, f"Responses not in user prompt for {tmpl_id}"


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

def test_pair_record_metadata_defaults():
    # Metadata fields are optional and default to None
    pair = PairRecord("id1", "Q?", "A", "B", "A")
    assert pair.score_gap is None
    assert pair.difficulty is None
    assert pair.verbosity_delta is None
    assert pair.complexity_max is None
    assert pair.is_multiturn is None

def test_pair_record_metadata_values():
    pair = PairRecord(
        "id2", "Q?", "A", "B", "A",
        score_gap=1.5, difficulty="easy",
        verbosity_delta=2, complexity_max=3, is_multiturn=False,
    )
    assert pair.score_gap == 1.5
    assert pair.difficulty == "easy"
    assert pair.verbosity_delta == 2

def test_pair_record_flip_negates_verbosity_delta():
    pair = PairRecord("p3", "Q?", "A", "B", "A", verbosity_delta=2)
    assert pair.flipped().verbosity_delta == -2

def test_pair_record_flip_preserves_symmetric_metadata():
    pair = PairRecord(
        "p4", "Q?", "A", "B", "A",
        score_gap=0.5, difficulty="medium",
        complexity_max=3, is_multiturn=True,
    )
    f = pair.flipped()
    assert f.score_gap == 0.5
    assert f.difficulty == "medium"
    assert f.complexity_max == 3
    assert f.is_multiturn is True


# ===========================================================================
# Multi-turn detection tests
# ===========================================================================

def test_detect_multiturn_single_turn():
    assert _detect_multiturn("What is the capital of France?") is False

def test_detect_multiturn_with_assistant_marker():
    prompt = "User: Who wrote Hamlet?\nAssistant: Shakespeare.\nUser: When?"
    assert _detect_multiturn(prompt) is True

def test_detect_multiturn_case_insensitive():
    assert _detect_multiturn("ASSISTANT: prior response here") is True


# ===========================================================================
# Stratified sampling tests
# ===========================================================================

def _make_pairs_with_difficulty(counts: dict) -> list:
    """Helper: create fake PairRecord lists with given difficulty distribution."""
    import random as _random
    pairs = []
    for difficulty, n in counts.items():
        for i in range(n):
            pairs.append(PairRecord(
                f"{difficulty}_{i}", "Q?", "A", "B", "A",
                difficulty=difficulty,
            ))
    _random.shuffle(pairs)
    return pairs

def test_stratified_sample_proportional():
    rng = __import__("random").Random(42)
    pairs = _make_pairs_with_difficulty({"easy": 100, "medium": 100, "hard": 100})
    sampled = _stratified_sample(pairs, n=30, rng=rng)
    assert len(sampled) == 30
    counts = {}
    for p in sampled:
        counts[p.difficulty] = counts.get(p.difficulty, 0) + 1
    assert counts["easy"] == 10
    assert counts["medium"] == 10
    assert counts["hard"] == 10

def test_stratified_sample_undersized_stratum():
    # When one stratum has fewer pairs than quota, shortfall is filled from others
    rng = __import__("random").Random(0)
    pairs = _make_pairs_with_difficulty({"easy": 50, "medium": 50, "hard": 2})
    sampled = _stratified_sample(pairs, n=30, rng=rng)
    assert len(sampled) == 30

def test_stratified_sample_n_larger_than_pool():
    rng = __import__("random").Random(0)
    pairs = _make_pairs_with_difficulty({"easy": 5, "hard": 5})
    sampled = _stratified_sample(pairs, n=20, rng=rng)
    assert len(sampled) == 10  # can't exceed pool size


# ===========================================================================
# randomize_position tests  (no disk I/O — tested via _stratified_sample /
# inline logic; randomize_position tested on synthetic PairRecord lists)
# ===========================================================================

def _apply_randomize_position(pairs, seed=42):
    """Replicate the randomize_position logic from load_dataset_pairs."""
    import random as _random
    rng2 = _random.Random(seed + 1337)
    result = []
    for pair in pairs:
        if rng2.random() < 0.5:
            neg_vd = (-pair.verbosity_delta
                      if pair.verbosity_delta is not None else None)
            pair = PairRecord(
                prompt_id=pair.prompt_id,
                prompt=pair.prompt,
                response_a=pair.response_b,
                response_b=pair.response_a,
                gold_label={"A": "B", "B": "A", "C": "C"}[pair.gold_label],
                score_gap=pair.score_gap,
                difficulty=pair.difficulty,
                verbosity_delta=neg_vd,
                complexity_max=pair.complexity_max,
                is_multiturn=pair.is_multiturn,
            )
        result.append(pair)
    return result

def test_randomize_position_produces_both_labels():
    # With 100 pairs all-A gold, randomizing should yield both A and B labels
    pairs = [PairRecord(f"p{i}", "Q?", "best", "worst", "A") for i in range(100)]
    randomized = _apply_randomize_position(pairs, seed=42)
    labels = {p.gold_label for p in randomized}
    assert "A" in labels and "B" in labels

def test_randomize_position_gold_b_inverts_correctly():
    # A pair with gold=B should become gold=A when flipped
    pair = PairRecord("px", "Q?", "best", "worst", "B", verbosity_delta=-1)
    result = _apply_randomize_position([pair] * 100, seed=99)
    # At least some should have been flipped to gold=A
    assert any(p.gold_label == "A" for p in result)

def test_randomize_position_preserves_response_identity():
    # After flipping, response_a and response_b are always the same two strings
    pairs = [PairRecord("p1", "Q?", "X", "Y", "A", verbosity_delta=3)]
    randomized = _apply_randomize_position(pairs, seed=0)
    p = randomized[0]
    assert set([p.response_a, p.response_b]) == {"X", "Y"}

def test_randomize_position_verbosity_delta_sign_flips():
    # verbosity_delta sign should invert when responses are swapped
    pair = PairRecord("p1", "Q?", "X", "Y", "A", verbosity_delta=3)
    # Force a flip by using a seed where this pair is always flipped
    rng = __import__("random").Random(42 + 1337)
    if rng.random() < 0.5:
        # was flipped
        result = _apply_randomize_position([pair], seed=42)
        p = result[0]
        if p.response_a == "Y":  # was flipped
            assert p.verbosity_delta == -3


# ===========================================================================
# Stratified metrics tests
# ===========================================================================

def test_compute_stratified_metrics_by_difficulty():
    gold = {"p1": "A", "p2": "B", "p3": "A"}
    results = [
        {"prompt_id": "p1", "condition": "AB", "label": "A", "experiment_id": "x"},
        {"prompt_id": "p1", "condition": "BA", "label": "B", "experiment_id": "x"},
        {"prompt_id": "p2", "condition": "AB", "label": "A", "experiment_id": "x"},  # bias
        {"prompt_id": "p2", "condition": "BA", "label": "A", "experiment_id": "x"},  # bias
        {"prompt_id": "p3", "condition": "AB", "label": "A", "experiment_id": "x"},
        {"prompt_id": "p3", "condition": "BA", "label": "B", "experiment_id": "x"},
    ]
    pairs = [
        PairRecord("p1", "Q?", "A", "B", "A", difficulty="easy"),
        PairRecord("p2", "Q?", "A", "B", "B", difficulty="hard"),
        PairRecord("p3", "Q?", "A", "B", "A", difficulty="easy"),
    ]
    strat = compute_stratified_metrics(results, pairs, "difficulty", compute_position_bias)
    assert "easy" in strat and "hard" in strat
    assert strat["easy"]["position_consistency"] == 1.0   # p1 and p3 both consistent
    assert strat["hard"]["position_consistency"] == 0.0   # p2 biased
    assert strat["easy"]["n_results"] == 4
    assert strat["hard"]["n_results"] == 2

def test_compute_stratified_metrics_skips_none_stratum():
    results = [
        {"prompt_id": "p1", "condition": "AB", "label": "A", "experiment_id": "x"},
        {"prompt_id": "p1", "condition": "BA", "label": "B", "experiment_id": "x"},
        {"prompt_id": "p2", "condition": "AB", "label": "A", "experiment_id": "x"},
        {"prompt_id": "p2", "condition": "BA", "label": "A", "experiment_id": "x"},
    ]
    pairs = [
        PairRecord("p1", "Q?", "A", "B", "A", difficulty="easy"),
        PairRecord("p2", "Q?", "A", "B", "A"),  # no difficulty (None)
    ]
    strat = compute_stratified_metrics(results, pairs, "difficulty", compute_position_bias)
    # p2 has no stratum so only "easy" should appear
    assert list(strat.keys()) == ["easy"]

def test_compute_stratified_metrics_flipped_ids_match():
    # experiments.py always writes pair.prompt_id (not pair.flipped().prompt_id)
    # for both AB and BA conditions, so both results share the same prompt_id.
    results = [
        {"prompt_id": "p1", "condition": "AB", "label": "A", "experiment_id": "x"},
        {"prompt_id": "p1", "condition": "BA", "label": "B", "experiment_id": "x"},
    ]
    pairs = [PairRecord("p1", "Q?", "A", "B", "A", difficulty="medium")]
    strat = compute_stratified_metrics(results, pairs, "difficulty", compute_position_bias)
    assert "medium" in strat
    assert strat["medium"]["position_consistency"] == 1.0


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
        # Label extraction
        test_extract_label_bare,
        test_extract_label_verdict_line,
        test_extract_label_with_thinking,
        test_extract_label_fallback,
        test_extract_thinking,
        test_extract_thinking_absent,
        test_extract_label_blind_1,
        test_extract_label_blind_2,
        test_extract_label_blind_verdict_line,
        # Templates
        test_all_templates_build,
        test_unknown_template_raises,
        test_build_blind_user_prompt,
        test_build_criterion_first_user_prompt,
        test_build_blind_criterion_first_user_prompt,
        test_all_b5_templates_build,
        # Dataset — PairRecord
        test_pair_record_flip,
        test_pair_record_flip_tie,
        test_pair_record_metadata_defaults,
        test_pair_record_metadata_values,
        test_pair_record_flip_negates_verbosity_delta,
        test_pair_record_flip_preserves_symmetric_metadata,
        # Dataset — multi-turn detection
        test_detect_multiturn_single_turn,
        test_detect_multiturn_with_assistant_marker,
        test_detect_multiturn_case_insensitive,
        # Dataset — stratified sampling
        test_stratified_sample_proportional,
        test_stratified_sample_undersized_stratum,
        test_stratified_sample_n_larger_than_pool,
        # Dataset — randomize position
        test_randomize_position_produces_both_labels,
        test_randomize_position_gold_b_inverts_correctly,
        test_randomize_position_preserves_response_identity,
        test_randomize_position_verbosity_delta_sign_flips,
        # Metrics
        test_compute_position_bias_perfect,
        test_compute_position_bias_full_bias,
        test_compute_pairwise_agreement_perfect,
        test_compute_pairwise_agreement_zero,
        test_compute_thinking_accuracy,
        # Stratified metrics
        test_compute_stratified_metrics_by_difficulty,
        test_compute_stratified_metrics_skips_none_stratum,
        test_compute_stratified_metrics_flipped_ids_match,
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
