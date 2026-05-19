import math
from typing import List, Counter

import dspy

from src.eval.EvalProcessor import _get_score


def consistency_score(answers: List[str]):
    # Extract labels from responses
    extracted_labels = []
    parse_failures = 0
    for answer in answers:
        label, parse_ok = _get_score(answer)
        extracted_labels.append(label)
        if not parse_ok:
            parse_failures += 1

    if parse_failures > 0:
        return 0  # answer is wrong

    # Compute consistency metrics
    shift = abs(extracted_labels[0] + extracted_labels[1])  # AB - (-BA) = AB + BA
    return shift


def accuracy_score(answers: List[str], ground_truth: str):
    # Extract labels from responses
    extracted_labels = []
    parse_failures = 0
    for answer in answers:
        label, parse_ok = _get_score(answer)
        extracted_labels.append(label)
        if not parse_ok:
            parse_failures += 1

    if parse_failures > 0:
        return 0  # answer is wrong

    # Compute accuracy metrics
    gold_ab_label = ground_truth
    gold_ba_label = ground_truth * -1
    ab_difference = abs(extracted_labels[0] - gold_ab_label)
    ba_difference = abs(extracted_labels[1] - gold_ba_label)

    return arithmetic_mean(ab_difference, ba_difference)

def harmonic_mean(x1: float, x2: float):
    return 2 * (x1 * x2) / (x1 + x2 + 1e-8)

def arithmetic_mean(x1: float, x2: float):
    return (x1 + x2) / 2


def eval_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return arithmetic_mean(
        accuracy_score(pred.answers, gold.ground_truth),
        consistency_score(pred.answers)
    )

def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    accuracy = accuracy_score(pred.answers, gold.ground_truth)
    consistency = consistency_score(pred.answers)
    scale = 2
    score = math.exp(-(accuracy + consistency) / scale)
    return dspy.Prediction(
        score=score,
        feedback=f"The accuracy of the answers is {accuracy} points different from the ground truth, the consistency of the answers is {consistency} points away from each other. The final score (exponential decay) is {score:.2f}. Remember to make generalizable adjustments, not overfit on specific examples."
    )

def accuracy_score_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    accuracy = accuracy_score(pred.answers, gold.ground_truth)
    scale = 2
    score = math.exp(-accuracy / scale)
    return score

def consistency_score_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    consistency = consistency_score(pred.answers)
    scale = 2
    score = math.exp(-consistency / scale)
    return score