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
        return 0 # answer is wrong

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

# def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
#     accuracy = accuracy_score(pred.answers, gold.ground_truth)
#     consistency = consistency_score(pred.answers)
#     scale = 2
#     score = math.exp(-(accuracy + consistency) / scale)
#     return dspy.Prediction(
#         score=score,
#         feedback=f"The accuracy of the answers is {accuracy} points different from the ground truth, the consistency of the answers is {consistency} points away from each other. The final score (exponential decay) is {score:.2f}. Remember to make generalizable adjustments, not overfit on specific examples."
#     )

def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ab_label, ab_ok = _get_score(pred.answers[0])
    ba_label, ba_ok = _get_score(pred.answers[1])

    if not (ab_ok and ba_ok):
        return dspy.Prediction(
            score=0.0,
            feedback=(
                "PARSE FAILURE: the response did not end in an extractable numeric label in "
                "the required schema. A malformed answer cannot be scored and counts as fully "
                "wrong (score = 0.0). Fix the output format robustly — the final verdict must "
                "always appear in the expected schema, regardless of the content of the pair."
            )
        )

    accuracy = accuracy_score(pred.answers, gold.ground_truth)
    consistency = consistency_score(pred.answers)
    scale = 2
    score = math.exp(-(accuracy + consistency) / scale)

    feedback = (
        f"This judge is evaluated on a single pair of candidate responses and must output a "
        f"preference label. The same underlying pair is shown to the judge twice in different "
        f"presentations; the judge does not know this and should not try to infer it. The "
        f"scoring combines two error terms:\n"
        f"  - Accuracy error = {accuracy}: mean absolute distance between the judge's labels "
        f"and the ground-truth preference across both presentations. 0 is perfect; higher "
        f"means the verdict disagrees with the true preference.\n"
        f"  - Consistency error = {consistency}: measures whether the judge reaches the same "
        f"underlying verdict regardless of how the pair is presented. 0 is perfect; non-zero "
        f"means the verdict depends on presentation rather than on the responses' content. "
        f"This is an external diagnostic — the judge cannot detect inconsistency from a single "
        f"call, but it can avoid causing it by judging on substance.\n"
        f"  - Combined score = {score:.3f} (exponential decay of total error; 1.0 is perfect, "
        f"approaches 0 as error grows).\n"
        # f"Improvement targets: if accuracy error dominates, the rubric or reasoning is "
        # f"pointing the verdict the wrong way and the criteria need sharpening. If consistency "
        # f"error dominates, the judgement is leaking from presentation cues rather than "
        # f"content — e.g. anchoring on whichever response is read first, weighting surface "
        # f"features like length or assertiveness, or applying asymmetric scrutiny. The prompt "
        # f"should push the judge to ground its verdict in concrete, content-level criteria "
        # f"that would yield the same answer no matter which response was labelled which. Make "
        # f"generalizable prompt changes — do not overfit to this example."
    )
    return dspy.Prediction(score=score, feedback=feedback)

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