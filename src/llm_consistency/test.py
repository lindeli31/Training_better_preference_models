import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import dspy
import wandb
import weave
from datasets import disable_caching
from dotenv import load_dotenv

from src.llm_consistency.BatchPermutationWrapper import BatchPermutationWrapper
from src.llm_consistency.ConsistentQA import ConsistentQA
from src.llm_consistency.adapt_HelpSteer3_filtered import adapt_HelpSteer3_filtered
from src.llm_consistency.metrics import (
    eval_metric, metric_with_feedback,
    accuracy_score_metric, accuracy_ab_metric, debiased_accuracy_metric, flip_rate_metric,
    consistency_score_metric,
    position_bias_report, bootstrap_ci,
)

disable_caching()
load_dotenv()

WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
if WANDB_API_KEY:
    wandb.login(WANDB_API_KEY)
    weave.init(project_name="DSPy")

CSCS_SERVING_API = os.environ.get('CSCS_SERVING_API') or os.environ['SWISSAI_API_KEY']
base_url = "https://api.swissai.svc.cscs.ch/v1"
task_lm = dspy.LM(
    "openai/meta-llama/Llama-3.3-70B-Instruct",
    temperature=1.0,
    api_key=CSCS_SERVING_API,
    api_base=base_url,
    cache=False,
)
reflection_lm = dspy.LM(
    "openai/meta-llama/Llama-3.3-70B-Instruct",
    temperature=1.0,
    api_key=CSCS_SERVING_API,
    api_base=base_url,
    cache=False,
)

dspy.configure(lm=task_lm)

dataset = adapt_HelpSteer3_filtered()
list_dataset = dataset.to_list()
adapted_dataset = [
    dspy.Example(
        answer_candidates=row['answer_candidates'],
        conversation_history=row['conversation_history'],
        ground_truth=row['ground_truth']
    ).with_inputs('answer_candidates', 'conversation_history')
    for row in list_dataset
]
train_dataset      = adapted_dataset[:150]
validation_dataset = adapted_dataset[150:450]
test_dataset       = adapted_dataset[450:750]

student_model = ConsistentQA()
wrapper_model = BatchPermutationWrapper(student_model)


# ---------------------------------------------------------------------------
# Rollout tracker
# ---------------------------------------------------------------------------

class RolloutTracker:
    """Wraps metric_with_feedback to record per-call scores during GEPA optimization."""

    def __init__(self, n_divisions: int = 12):
        self.n_divisions = n_divisions
        self._scores: list[float] = []

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
        result = metric_with_feedback(gold, pred, trace, pred_name, pred_trace)
        self._scores.append(result.score)
        return result

    def rollout_means(self) -> list[float]:
        """Divide total calls evenly into n_divisions chunks and return mean per chunk."""
        chunk_size = len(self._scores) // self.n_divisions
        if chunk_size == 0:
            return []
        return [
            float(np.mean(self._scores[i * chunk_size:(i + 1) * chunk_size]))
            for i in range(self.n_divisions)
        ]

    def plot(self, output_path: str = "results/gepa_debiased_accuracy.png"):
        if not self._scores:
            print("No scores recorded — nothing to plot.")
            return
        rollouts = self.rollout_means()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        n_calls = len(self._scores)
        xs = np.arange(1, n_calls + 1)
        running_best = np.maximum.accumulate(self._scores)
        chunk_size = n_calls // self.n_divisions if n_calls >= self.n_divisions else 0

        seg_means = rollouts if rollouts else [float(np.mean(self._scores))]
        seg_xs = [int((i + 0.5) * chunk_size) for i in range(len(seg_means))] if chunk_size > 0 else [n_calls // 2]
        running_best_segs = list(np.maximum.accumulate(seg_means))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: running best of segment means
        axes[0].plot(seg_xs, running_best_segs, color="green", linewidth=1.5, marker="o", label="Running best (seg mean)")
        axes[0].plot(seg_xs, seg_means, color="steelblue", linewidth=1, marker="x", alpha=0.6, label="Segment mean")
        axes[0].set_xlabel("Rollout")
        axes[0].set_ylabel("Combined score (acc+cons)")
        axes[0].set_title("Running Best of Segment Means")
        axes[0].set_ylim(0, 1)
        axes[0].set_xlim(1, n_calls)
        axes[0].legend()

        # Right: all per-call scores with segment boundaries
        axes[1].plot(xs, self._scores, alpha=0.3, label="Per-call score")
        axes[1].plot(xs, running_best, color="green", linewidth=1, alpha=0.7, label="Running best (per-call)")
        if chunk_size > 0:
            for i in range(1, self.n_divisions):
                axes[1].axvline(i * chunk_size, color="red", linestyle="--", alpha=0.3)
        axes[1].set_xlabel("Rollout")
        axes[1].set_ylabel("Combined score (acc+cons)")
        axes[1].set_title(f"Per-rollout Scores (red = 1/{self.n_divisions} budget boundary)")
        axes[1].set_ylim(0, 1)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        print(f"\nPlot saved to {output_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def collect_predictions(model, dataset, num_threads=32):
    """Run model on dataset in parallel, returns predictions in original order."""
    results = [None] * len(dataset)

    def run(i, ex):
        return i, model(answer_candidates=ex.answer_candidates, conversation_history=ex.conversation_history)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(run, i, ex): i for i, ex in enumerate(dataset)}
        for future in as_completed(futures):
            i, pred = future.result()
            results[i] = pred
    return results


def run_full_eval(model, dataset, label="Evaluation"):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    preds = collect_predictions(model, dataset)

    rows = [
        ("Accuracy (both perms)", mean(accuracy_score_metric(g, p)    for g, p in zip(dataset, preds))),
        ("Accuracy (AB only)",    mean(accuracy_ab_metric(g, p)       for g, p in zip(dataset, preds))),
        ("Debiased accuracy",     mean(debiased_accuracy_metric(g, p) for g, p in zip(dataset, preds))),
        ("Flip rate",             mean(flip_rate_metric(g, p)         for g, p in zip(dataset, preds))),
        ("Consistency",           mean(consistency_score_metric(g, p) for g, p in zip(dataset, preds))),
        ("Combined (acc+cons)",   mean(eval_metric(g, p)              for g, p in zip(dataset, preds))),
    ]
    for name, val in rows:
        print(f"  {name:<30} {val:.3f}")

    position_bias_report(dataset, preds)


def run_test_eval(model, dataset, label="Test Evaluation"):
    """Like run_full_eval but reports bootstrap 95% CIs — use only on held-out test set."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    preds = collect_predictions(model, dataset)

    metric_fns = [
        ("Accuracy (both perms)", accuracy_score_metric),
        ("Accuracy (AB only)",    accuracy_ab_metric),
        ("Debiased accuracy",     debiased_accuracy_metric),
        ("Flip rate",             flip_rate_metric),
        ("Consistency",           consistency_score_metric),
        ("Combined (acc+cons)",   eval_metric),
    ]
    for name, fn in metric_fns:
        scores = [fn(g, p) for g, p in zip(dataset, preds)]
        mu, lo, hi = bootstrap_ci(scores, ci=0.99)
        print(f"  {name:<30} {mu:.3f}  99% CI [{lo:.3f}, {hi:.3f}]")

    position_bias_report(dataset, preds)


# ---------------------------------------------------------------------------
# Logging to file
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

_log_path = Path("results/eval_log.txt")
_log_path.parent.mkdir(parents=True, exist_ok=True)
_log_file = open(_log_path, "a")
_log_file.write(f"\n{'#'*55}\n# Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'#'*55}\n")
sys.stdout = _Tee(sys.__stdout__, _log_file)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

response = wrapper_model(
    answer_candidates=list_dataset[0]['answer_candidates'],
    conversation_history=list_dataset[0]['conversation_history'],
)
print("answers:", response.answers)
print("history: ", task_lm.inspect_history(n=1))


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

run_full_eval(wrapper_model, validation_dataset, "Baseline — Validation Set")
run_test_eval(wrapper_model, test_dataset, "Baseline — Test Set")


# ---------------------------------------------------------------------------
# GEPA optimisation
# ---------------------------------------------------------------------------

tracker = RolloutTracker(n_divisions=12)  # matches auto="medium" n=12

optimizer = dspy.GEPA(
    metric=tracker,
    reflection_lm=reflection_lm,
    auto="heavy"
)

optimized_wrapper = optimizer.compile(
    student=wrapper_model,
    trainset=train_dataset,
    valset=validation_dataset,
)

tracker.plot("results/gepa_consistency.png")


# ---------------------------------------------------------------------------
# Post-optimisation evaluation
# ---------------------------------------------------------------------------

run_full_eval(optimized_wrapper, validation_dataset, "After GEPA Optimisation")
run_test_eval(optimized_wrapper, test_dataset, "Test Set — After GEPA Optimisation")

sys.stdout = sys.__stdout__
_log_file.close()
print(f"Eval log saved to {_log_path}")
