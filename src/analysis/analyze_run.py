from datasets import load_dataset, Dataset


def analyze_run(
        repo_id: str,
        dataset_name: str,
        split_name: str,
) -> str:
    dataset = load_dataset(path=repo_id, name=dataset_name, split=split_name)

    shift = _get_shift(dataset)
    bias_toward_first_position = _get_bias_toward_first_position(dataset)
    bias_toward_second_position = _get_bias_toward_second_position(dataset)
    tie_inconsistency_rate = _get_tie_inconsistency_rate(dataset)

    hard_accuracy = _get_hard_accuracy(dataset)
    soft_accuracy = _get_soft_accuracy(dataset)
    ab_accuracy = _get_ab_accuracy(dataset)
    ba_accuracy = _get_ba_accuracy(dataset)
    accuracy_gap = ab_accuracy - ba_accuracy

    plot_ab_difference(
        dataset,
        title=f"{repo_id} | {split_name}",
        save_path=f"{split_name}_ab_difference.png",
    )
    plt.show()

    plot_ab_ba_confusion(
        dataset,
        title=f"{repo_id} | {split_name}",
        save_path=f"{split_name}_ab_ba_confusion.png",
        normalize="all",
    )
    plt.show()

    plot_extracted_labels_confusion(
        dataset,
        title=f"{repo_id} | {split_name}",
        save_path=f"{split_name}_extracted_labels_confusion.png",
        normalize="all",
    )
    plt.show()

    plot_shift(
        dataset,
        title=f"{repo_id} | {split_name}",
        save_path=f"{split_name}_shift.png",
    )
    plt.show()

    return f"""
==================================================
Run Analysis: {repo_id} | dataset: {dataset_name} | Split: {split_name}
==================================================

[ Accuracy Metrics ]
--------------------------------------------------
Hard Accuracy:            {hard_accuracy:.2f}
Soft Accuracy:            {soft_accuracy}
AB Accuracy:              {ab_accuracy:.2f}
BA Accuracy:              {ba_accuracy:.2f}
Accuracy Gap:             {accuracy_gap:.2f}

[ Bias & Consistency ]
--------------------------------------------------
Shift:                    {shift}
Bias (First Position):    {bias_toward_first_position:.2f}
Bias (Second Position):   {bias_toward_second_position:.2f}
Tie Inconsistency Rate:   {tie_inconsistency_rate:.2f}
==================================================
""".strip()

def _get_shift(dataset: Dataset):
    df = dataset.select_columns(["shift"]).to_pandas()
    counts = df["shift"].value_counts(normalize=True)
    return counts

def _get_bias_toward_first_position(dataset: Dataset) -> float:
    filtered_dataset = dataset.filter(
        lambda x: x["bias_toward_first_position"] == True
    )
    return len(filtered_dataset) / len(dataset)

def _get_bias_toward_second_position(dataset: Dataset) -> float:
    filtered_dataset = dataset.filter(
        lambda x: x["bias_toward_second_position"] == True
    )
    return len(filtered_dataset) / len(dataset)

def _get_tie_inconsistency_rate(dataset: Dataset) -> float:
    filtered_dataset = dataset.filter(
        lambda x: x["tie_inconsistency"] == True
    )
    return len(filtered_dataset) / len(dataset)

def _get_hard_accuracy(dataset: Dataset) -> float:
    """Both orderings exactly match the ground truth."""
    filtered = dataset.filter(
        lambda x: x["ab_difference"] == 0 and x["ba_difference"] == 0
    )
    return len(filtered) / len(dataset)

def _get_soft_accuracy(dataset: Dataset):
    # """AB ordering within 1 step of the ground truth."""
    # filtered = dataset.filter(
    #     lambda x: x["ab_difference"] == 0
    # )
    # return len(filtered) / len(dataset)
    df = dataset.select_columns(["ab_difference"]).to_pandas()
    counts = df["ab_difference"].value_counts(normalize=True)
    return counts

def _get_ab_accuracy(dataset: Dataset) -> float:
    # filtered_correct_dataset = dataset.filter(
    #     lambda x: (x["ab_correct"] == True and x["ground_truth"] == 0) # 0 -> "A"
    #            or (x["ba_correct"] == True and x["ground_truth"] == 1) # 1 -> "B"
    # )
    # filtered_total_dataset = dataset.filter(
    #     lambda x: x["ground_truth"] == 0 or  x["ground_truth"] == 1
    # )
    # return len(filtered_correct_dataset) / len(filtered_total_dataset)
    """First-position accuracy: exact-match rate when the preferred answer is shown first.

        - ground_truth < 0  -> A is preferred; A sits first in AB order  -> check ab_difference == 0
        - ground_truth > 0  -> B is preferred; B sits first in BA order  -> check ba_difference == 0
        - ground_truth == 0 -> tie, excluded from this metric
        """
    filtered_correct = dataset.filter(
        lambda x: (x["ab_difference"] == 0 and x["ground_truth"] < 0)
                  or (x["ba_difference"] == 0 and x["ground_truth"] > 0)
    )
    filtered_total = dataset.filter(lambda x: x["ground_truth"] != 0)
    if len(filtered_total) == 0:
        return float("nan")
    return len(filtered_correct) / len(filtered_total)

def _get_ba_accuracy(dataset: Dataset) -> float:
    # filtered_correct_dataset = dataset.filter(
    #     lambda x: (x["ba_correct"] == True and x["ground_truth"] == 0) # 0 -> "A"
    #            or (x["ab_correct"] == True and x["ground_truth"] == 1) # 1 -> "B"
    # )
    # filtered_total_dataset = dataset.filter(
    #     lambda x: x["ground_truth"] == 0 or  x["ground_truth"] == 1
    # )
    # return len(filtered_correct_dataset) / len(filtered_total_dataset)
    """Second-position accuracy: exact-match rate when the preferred answer is shown second."""
    filtered_correct = dataset.filter(
        lambda x: (x["ba_difference"] == 0 and x["ground_truth"] < 0)
                  or (x["ab_difference"] == 0 and x["ground_truth"] > 0)
    )
    filtered_total = dataset.filter(lambda x: x["ground_truth"] != 0)
    if len(filtered_total) == 0:
        return float("nan")
    return len(filtered_correct) / len(filtered_total)

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset


def plot_ab_difference(
    dataset: Dataset,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar plot of ab_difference distribution with hard-accuracy overlay on the 0-bar.

    The light-blue bar at x=0 shows the fraction of examples where ab_difference == 0.
    The dark-blue overlay on top of it shows the hard-accuracy fraction
    (both ab_difference == 0 AND ba_difference == 0), which is a subset of that bar.
    """
    x_range = list(range(-3, 4))  # -3..+3 inclusive
    total = len(dataset)

    # ab_difference distribution
    ab_counts = {x: 0 for x in x_range}
    hard_correct = 0
    for ex in dataset:
        ab = ex.get("ab_difference")
        ba = ex.get("ba_difference")
        if ab is None:
            continue
        if ab in ab_counts:
            ab_counts[ab] += 1
        if ab == 0 and ba == 0:
            hard_correct += 1

    ab_pct = [ab_counts[x] / total for x in x_range]
    hard_pct = hard_correct / total

    fig, ax = plt.subplots(figsize=(8, 5))

    # Light blue base bars
    ax.bar(x_range, ab_pct, color="#9ecae1", edgecolor="white",
           label="ab_difference distribution")

    # Dark blue overlay on the 0-bar for hard accuracy
    ax.bar([0], [hard_pct], color="#08519c", edgecolor="white",
           label=f"Hard accuracy ({hard_pct:.1%})")

    # Annotate each bar with its percentage
    for x, pct in zip(x_range, ab_pct):
        if pct > 0:
            ax.text(x, pct + 0.005, f"{pct:.1%}",
                    ha="center", va="bottom", fontsize=9)
    if hard_pct > 0:
        ax.text(0, hard_pct / 2, f"{hard_pct:.1%}",
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")

    ax.set_xticks(x_range)
    ax.set_xlabel("ab_difference (predicted − ground truth, AB ordering)")
    ax.set_ylabel("Fraction of examples")
    ax.set_title(title or "AB Difference Distribution")
    ax.set_ylim(0, max(ab_pct) * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset


def plot_ab_ba_confusion(
    dataset: Dataset,
    title: str | None = None,
    save_path: str | None = None,
    normalize: str = "all",  # "all", "row", "col", or "none"
) -> plt.Figure:
    """Confusion matrix between ab_difference (rows) and ba_difference (columns).

    Each cell is the fraction of examples falling at that (ab_diff, ba_diff) pair.
    The diagonal at (0, 0) is hard accuracy.

    normalize:
        "all"  — fractions of total dataset (cells sum to 1)
        "row"  — fractions within each ab_difference row
        "col"  — fractions within each ba_difference column
        "none" — raw counts
    """
    values = list(range(-3, 4))  # -3..+3 inclusive
    n = len(values)
    val_to_idx = {v: i for i, v in enumerate(values)}

    counts = np.zeros((n, n), dtype=int)
    total = 0
    out_of_range = 0
    for ex in dataset:
        ab = ex.get("ab_difference")
        ba = ex.get("ba_difference")
        if ab is None or ba is None:
            continue
        total += 1
        if ab in val_to_idx and ba in val_to_idx:
            counts[val_to_idx[ab], val_to_idx[ba]] += 1
        else:
            out_of_range += 1

    if normalize == "all":
        matrix = counts / total if total else counts.astype(float)
        fmt = lambda v: f"{v:.1%}" if v > 0 else ""
        cbar_label = "Fraction of all examples"
    elif normalize == "row":
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.where(row_sums > 0, counts / row_sums, 0.0)
        fmt = lambda v: f"{v:.1%}" if v > 0 else ""
        cbar_label = "Fraction within ab_difference row"
    elif normalize == "col":
        col_sums = counts.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.where(col_sums > 0, counts / col_sums, 0.0)
        fmt = lambda v: f"{v:.1%}" if v > 0 else ""
        cbar_label = "Fraction within ba_difference column"
    else:  # "none"
        matrix = counts.astype(float)
        fmt = lambda v: f"{int(v)}" if v > 0 else ""
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")

    # Cell labels — switch text color on dark cells for readability
    vmax = matrix.max() if matrix.size else 1.0
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if v == 0:
                continue
            color = "white" if v > 0.5 * vmax else "black"
            ax.text(j, i, fmt(v), ha="center", va="center",
                    fontsize=9, color=color)

    # Highlight the (0, 0) cell — hard accuracy
    zero_idx = val_to_idx[0]
    ax.add_patch(plt.Rectangle(
        (zero_idx - 0.5, zero_idx - 0.5), 1, 1,
        fill=False, edgecolor="#08519c", linewidth=2.5,
    ))

    ax.set_xticks(range(n))
    ax.set_xticklabels(values)
    ax.set_yticks(range(n))
    ax.set_yticklabels(values)
    ax.set_xlabel("ba_difference (BA ordering)")
    ax.set_ylabel("ab_difference (AB ordering)")
    ax.set_title(title or "AB vs BA Difference Confusion Matrix")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    if out_of_range:
        ax.text(0.0, -0.12, f"{out_of_range} example(s) outside [-3, +3] not shown",
                transform=ax.transAxes, fontsize=9,
                color="#a32d2d", ha="left", va="top")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset


def plot_extracted_labels_confusion(
    dataset: Dataset,
    title: str | None = None,
    save_path: str | None = None,
    normalize: str = "all",  # "all", "row", "col", or "none"
) -> plt.Figure:
    """Confusion matrix of the raw extracted scores: AB-ordering (rows) vs BA-ordering (columns).

    Both axes are on the model's score scale, -2..+2.

    Patterns to read:
      - Anti-diagonal (top-right to bottom-left, where extracted[0] == -extracted[1])
        is the consistency line — the model gives mirrored scores when the order flips.
      - Main diagonal (where extracted[0] == extracted[1]) is the opposite:
        the model returned the same score regardless of order — strong positional bias.
      - Mass in the (0, 0) cell means the model defaults to "tie" in both orderings.
      - Mass on rows/columns ±2 means the model saturates to extreme scores.

    normalize:
        "all"  — fractions of total examples (cells sum to 1)
        "row"  — fractions within each AB row    ("given AB=X, what did BA do?")
        "col"  — fractions within each BA column ("given BA=Y, what did AB do?")
        "none" — raw counts
    """
    values = list(range(-2, 3))  # -2..+2 inclusive
    n = len(values)
    val_to_idx = {v: i for i, v in enumerate(values)}

    counts = np.zeros((n, n), dtype=int)
    out_of_range = 0
    skipped = 0
    for ex in dataset:
        labels = ex.get("extracted_labels")
        if not labels or len(labels) != 2 or labels[0] is None or labels[1] is None:
            skipped += 1
            continue
        ab, ba = labels[0], labels[1]
        if ab in val_to_idx and ba in val_to_idx:
            counts[val_to_idx[ab], val_to_idx[ba]] += 1
        else:
            out_of_range += 1

    total = counts.sum()
    if normalize == "all":
        matrix = counts / total if total else counts.astype(float)
        fmt = lambda v: f"{v:.1%}" if v > 0 else ""
        cbar_label = "Fraction of all examples"
    elif normalize == "row":
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.where(row_sums > 0, counts / row_sums, 0.0)
        fmt = lambda v: f"{v:.1%}" if v > 0 else ""
        cbar_label = "Fraction within AB row"
    elif normalize == "col":
        col_sums = counts.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.where(col_sums > 0, counts / col_sums, 0.0)
        fmt = lambda v: f"{v:.1%}" if v > 0 else ""
        cbar_label = "Fraction within BA column"
    else:  # "none"
        matrix = counts.astype(float)
        fmt = lambda v: f"{int(v)}" if v > 0 else ""
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")

    vmax = matrix.max() if matrix.size else 1.0
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if v == 0:
                continue
            color = "white" if v > 0.5 * vmax else "black"
            ax.text(j, i, fmt(v), ha="center", va="center",
                    fontsize=9, color=color)

    # Anti-diagonal = consistency line (extracted[0] == -extracted[1])
    for v in values:
        i = val_to_idx[v]
        j = val_to_idx[-v]
        ax.add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="#08519c", linewidth=2.0,
        ))

    ax.set_xticks(range(n))
    ax.set_xticklabels(values)
    ax.set_yticks(range(n))
    ax.set_yticklabels(values)
    ax.set_xlabel("extracted_labels[1] (BA ordering score)")
    ax.set_ylabel("extracted_labels[0] (AB ordering score)")
    ax.set_title(title or "AB vs BA Extracted Scores")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    notes = []
    if skipped:
        notes.append(f"{skipped} example(s) skipped (parse failure)")
    if out_of_range:
        notes.append(f"{out_of_range} example(s) outside [-2, +2] not shown")
    if notes:
        ax.text(0.0, -0.12, " · ".join(notes),
                transform=ax.transAxes, fontsize=9,
                color="#a32d2d", ha="left", va="top")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset


def plot_shift(
    dataset: Dataset,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar plot of shift distribution with hard-accuracy overlay on the 0-bar.

    shift = extracted[0] + extracted[1], range [-4, +4].
      shift == 0 -> perfectly consistent (mirrored scores under order-swap)
      shift  < 0 -> model leans toward first position in both orderings
      shift  > 0 -> model leans toward second position in both orderings

    The light-blue bar at x=0 is the consistency rate.
    The dark-blue overlay on it is hard accuracy (consistent AND correct,
    which is a strict subset of consistency).
    """
    x_range = list(range(-4, 5))  # -4..+4 inclusive
    total = len(dataset)

    shift_counts = {x: 0 for x in x_range}
    hard_correct = 0
    out_of_range = 0
    for ex in dataset:
        s = ex.get("shift")
        if s is None:
            continue
        if s in shift_counts:
            shift_counts[s] += 1
        else:
            out_of_range += 1
        if ex.get("ab_difference") == 0 and ex.get("ba_difference") == 0:
            hard_correct += 1

    shift_pct = [shift_counts[x] / total for x in x_range]
    hard_pct = hard_correct / total

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x_range, shift_pct, color="#9ecae1", edgecolor="white",
           label="shift distribution")

    ax.bar([0], [hard_pct], color="#08519c", edgecolor="white",
           label=f"Hard accuracy ({hard_pct:.1%})")

    for x, pct in zip(x_range, shift_pct):
        if pct > 0:
            ax.text(x, pct + 0.005, f"{pct:.1%}",
                    ha="center", va="bottom", fontsize=9)
    if hard_pct > 0:
        ax.text(0, hard_pct / 2, f"{hard_pct:.1%}",
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")

    # Light vertical guides separating the three regions
    ax.axvline(-0.5, color="#888888", linestyle=":", linewidth=0.8)
    ax.axvline(0.5, color="#888888", linestyle=":", linewidth=0.8)

    ymax = max(shift_pct) * 1.15 if max(shift_pct) > 0 else 0.1
    ax.set_ylim(0, ymax)
    ax.text(-2.5, ymax * 0.97, "first-position bias",
            ha="center", va="top", fontsize=9, color="#666666", style="italic")
    ax.text(0, ymax * 0.97, "consistent",
            ha="center", va="top", fontsize=9, color="#666666", style="italic")
    ax.text(2.5, ymax * 0.97, "second-position bias",
            ha="center", va="top", fontsize=9, color="#666666", style="italic")

    ax.set_xticks(x_range)
    ax.set_xlabel("shift = extracted[0] + extracted[1]")
    ax.set_ylabel("Fraction of examples")
    ax.set_title(title or "Shift Distribution")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if out_of_range:
        ax.text(0.0, -0.12, f"{out_of_range} example(s) outside [-4, +4] not shown",
                transform=ax.transAxes, fontsize=9,
                color="#a32d2d", ha="left", va="top")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig