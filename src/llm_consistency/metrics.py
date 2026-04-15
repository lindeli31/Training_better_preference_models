from typing import List, Counter

import numpy as np
import dspy



def consistency_score(answers: List[str]):
    if answers[0].strip().lower() == "a" and answers[1].strip().lower() == "b":
        return 1.0
    elif answers[0].strip().lower() == "b" and answers[1].strip().lower() == "a":
        return 1.0
    elif answers[0].strip().lower() == "c" and answers[1].strip().lower() == "c":
        return 1.0
    elif answers[0].strip().lower() == answers[1].strip().lower():
        return 0.0
    else:
        return 0.0


def accuracy_score(answers: List[str], ground_truth: str):
    if ground_truth == 0:
        gts = ["b", "a"]
    elif ground_truth == 1:
        gts = ["a", "b"]
    else:
        gts = ["equal", "equal"]
    score = 0
    for count, _ in enumerate(answers):
        if answers[count].strip().lower() == gts[count].strip().lower():
            score += 1
    return score / len(answers)


def harmonic_mean(x1: float, x2: float):
    return 2 * (x1 * x2) / (x1 + x2 + 1e-8)

def arithmetic_mean(x1: float, x2: float):
    return (x1 + x2) / 2


# ---------------------------------------------------------------------------
# Position-bias-aware metrics
# ---------------------------------------------------------------------------

_REMAP = {"a": "b", "b": "a", "equal": "equal"}


def accuracy_ab(answers: List[str], ground_truth: int) -> float:
    """Accuracy on the AB ordering only (permutation 0)."""
    expected = "a" if ground_truth == 1 else ("b" if ground_truth == 0 else "equal")
    return 1.0 if answers[0].strip().lower() == expected else 0.0


def accuracy_ba_remapped(answers: List[str], ground_truth: int) -> float:
    """Accuracy on the BA ordering, remapped back to original response labels."""
    remapped = _REMAP.get(answers[1].strip().lower(), answers[1].strip().lower())
    expected = "a" if ground_truth == 1 else ("b" if ground_truth == 0 else "equal")
    return 1.0 if remapped == expected else 0.0


def debiased_accuracy(answers: List[str], ground_truth: int) -> float:
    """Average accuracy across AB and BA orderings — robust to position bias."""
    return 0.5 * accuracy_ab(answers, ground_truth) + 0.5 * accuracy_ba_remapped(answers, ground_truth)


def flip_rate(answers: List[str]) -> float:
    """1.0 if the model flips its answer when inputs are swapped (position-inconsistent)."""
    remapped_ba = _REMAP.get(answers[1].strip().lower(), answers[1].strip().lower())
    return 0.0 if answers[0].strip().lower() == remapped_ba else 1.0


# ---------------------------------------------------------------------------
# DSPy metric wrappers
# ---------------------------------------------------------------------------

def eval_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return arithmetic_mean(
        accuracy_score(pred.answers, gold.ground_truth),
        consistency_score(pred.answers)
    )

def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    cons = consistency_score(pred.answers)
    acc = accuracy_score(pred.answers, gold.ground_truth)
    score = arithmetic_mean(acc, cons)
    acc_ab = accuracy_ab(pred.answers, gold.ground_truth)
    acc_ba = accuracy_ba_remapped(pred.answers, gold.ground_truth)
    return dspy.Prediction(score=score, feedback=f"Combined score (acc+cons mean): {score:.2f}. Accuracy averaged across both orderings: {acc:.2f} (AB: {acc_ab:.2f}, BA remapped: {acc_ba:.2f}). Consistency (1.0 = correctly flips answer when inputs swap): {cons:.2f}.")

def accuracy_score_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return accuracy_score(pred.answers, gold.ground_truth)

def consistency_score_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return consistency_score(pred.answers)

def accuracy_ab_metric(gold, pred, trace=None):
    return accuracy_ab(pred.answers, gold.ground_truth)

def debiased_accuracy_metric(gold, pred, trace=None):
    return debiased_accuracy(pred.answers, gold.ground_truth)

def flip_rate_metric(gold, pred, trace=None):
    return flip_rate(pred.answers)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores: List[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) with percentile bootstrap confidence interval."""
    arr = np.array(scores)
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    return float(np.mean(arr)), float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))



# ---------------------------------------------------------------------------
# Aggregate position bias report
# ---------------------------------------------------------------------------

def position_bias_report(examples, predictions):
    """Print primacy bias and position-conditioned accuracy across the dataset."""
    ab_answers = [p.answers[0].strip().lower() for p in predictions]
    gts = [e.ground_truth for e in examples]

    # Primacy bias: P(model picks "a" in AB ordering) - 0.5
    # Recency bias: P(model picks "b" in AB ordering) - 0.5
    # Note: with ties possible, primacy + recency != 0
    frac_a = sum(1 for a in ab_answers if a == "a") / len(ab_answers)
    frac_b = sum(1 for a in ab_answers if a == "b") / len(ab_answers)
    frac_equal = sum(1 for a in ab_answers if a == "equal") / len(ab_answers)
    primacy = frac_a - 0.5
    recency = frac_b - 0.5

    # Position-conditioned accuracy in AB ordering
    # gold in pos A: ground_truth==1 (response1 better, sits in slot a)
    # gold in pos B: ground_truth==0 (response2 better, sits in slot b)
    gold_in_a = [(a, gt) for a, gt in zip(ab_answers, gts) if gt == 1]
    gold_in_b = [(a, gt) for a, gt in zip(ab_answers, gts) if gt == 0]

    acc_gold_a = (sum(1 for a, _ in gold_in_a if a == "a") / len(gold_in_a)) if gold_in_a else float("nan")
    acc_gold_b = (sum(1 for a, _ in gold_in_b if a == "b") / len(gold_in_b)) if gold_in_b else float("nan")
    gap = acc_gold_a - acc_gold_b

    # Inconsistent primacy: picks "a" in AB AND "a" in BA (always anchors to first position)
    # Inconsistent recency: picks "b" in AB AND "b" in BA (always anchors to last position)
    ba_answers = [p.answers[1].strip().lower() for p in predictions]
    incons_primacy = sum(
        1 for ab, ba in zip(ab_answers, ba_answers) if ab == "a" and ba == "a"
    ) / len(ab_answers)
    incons_recency = sum(
        1 for ab, ba in zip(ab_answers, ba_answers) if ab == "b" and ba == "b"
    ) / len(ab_answers)

    print(f"\n  --- Position Bias ---")
    print(f"  Primacy bias (P(pick A) - 0.5):     {primacy:+.3f}  (>0 = favours first)")
    print(f"  Recency bias (P(pick B) - 0.5):     {recency:+.3f}  (>0 = favours last)")
    print(f"  Inconsistent primacy rate:           {incons_primacy:.3f}  (always picks first-presented)")
    print(f"  Inconsistent recency rate:           {incons_recency:.3f}  (always picks last-presented)")
    print(f"  Equal rate:                          {frac_equal:.3f}")
    print(f"  Acc when gold in position A:         {acc_gold_a:.3f}  (n={len(gold_in_a)})")
    print(f"  Acc when gold in position B:         {acc_gold_b:.3f}  (n={len(gold_in_b)})")
    print(f"  Position-conditioned gap (A - B):    {gap:+.3f}  (0 = unbiased)")
