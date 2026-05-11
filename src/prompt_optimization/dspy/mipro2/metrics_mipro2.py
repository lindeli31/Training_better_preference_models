from typing import List, Counter

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
    # clean_answers = [ans.strip().lower() for ans in answers]
    # counts = Counter(clean_answers)
    # majority, majority_count = counts.most_common(1)[0]
    # return majority_count / len(answers)


def accuracy_score(answers: List[str], ground_truth: str):
    if ground_truth == 0:
        gts = ["a", "b"]
    elif ground_truth == 1:
        gts = ["b", "a"]
    else:
        gts = ["equal", "equal"]
    # perms = list(itertools.permutations(facts, len(facts)))
    # correct_answers = [ans for ans in answers if ans.strip().lower() == ground_truth.strip().lower()]
    # return len(correct_answers) / len(answers)
    score = 0
    for count, _ in enumerate(answers):
        if answers[count].strip().lower() == gts[count].strip().lower():
            score += 1
    return score / len(answers)

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
    score = arithmetic_mean(accuracy, consistency)
    return dspy.Prediction(score=score, feedback=f"The accuracy of the answers is {accuracy:.2f}%, the consistency of the answers is {consistency:.2f}%. The final score (arithmetic mean) is {score:.2f}.")

def accuracy_score_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return accuracy_score(pred.answers, gold.ground_truth)

def consistency_score_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return consistency_score(pred.answers)