"""
dataset.py
----------
Loads and pre-processes preference datasets for the judge pipeline.

Supported datasets
------------------
  - nvidia/HelpSteer2 (default)
  - openai/summarize_from_feedback
  - Anthropic/hh-rlhf

Each dataset is normalised to a list of PairRecord objects with fields:
    prompt_id : str
    prompt    : str
    response_a: str   (the "chosen" / higher-rated response in original data)
    response_b: str   (the "rejected" / lower-rated response in original data)
    gold_label: str   ("A", "B", or "C") derived from human preference labels

Usage
-----
    from dataset import load_dataset_pairs, PairRecord
    pairs = load_dataset_pairs("nvidia/HelpSteer2", split="validation", n=200)
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PairRecord:
    prompt_id: str
    prompt: str
    response_a: str        # original A (not flipped)
    response_b: str        # original B (not flipped)
    gold_label: str        # "A" = A is better, "B" = B is better, "C" = tie

    def flipped(self) -> "PairRecord":
        """Return a new PairRecord with A and B swapped, gold label adjusted."""
        flipped_label = {"A": "B", "B": "A", "C": "C"}.get(self.gold_label, self.gold_label)
        return PairRecord(
            prompt_id=self.prompt_id + "_flipped",
            prompt=self.prompt,
            response_a=self.response_b,
            response_b=self.response_a,
            gold_label=flipped_label,
        )


def _make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# HelpSteer2 loader
# ---------------------------------------------------------------------------

def _load_helpsteer2(split: str, n: Optional[int]) -> list[PairRecord]:
    """
    HelpSteer2 is a scored dataset (not pairwise).
    We construct pairs by taking the top-scored and bottom-scored responses
    for the same prompt.

    Columns: prompt, response, helpfulness, correctness, coherence,
             complexity, verbosity  (all 0-4 Likert)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    ds = load_dataset("nvidia/HelpSteer2", split=split)
    if n:
        ds = ds.select(range(min(n * 5, len(ds))))   # oversample, we'll filter below

    # Group by prompt
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for row in ds:
        groups[row["prompt"]].append(row)

    pairs: list[PairRecord] = []
    for prompt, rows in groups.items():
        if len(rows) < 2:
            continue
        # Score = mean of helpfulness + correctness + coherence
        def score(r):
            return (r["helpfulness"] + r["correctness"] + r["coherence"]) / 3.0

        rows_sorted = sorted(rows, key=score, reverse=True)
        best, worst = rows_sorted[0], rows_sorted[-1]

        s_best  = score(best)
        s_worst = score(worst)

        if s_best == s_worst:
            gold = "C"
        elif s_best > s_worst:
            gold = "A"
        else:
            gold = "B"

        pairs.append(PairRecord(
            prompt_id=_make_id(prompt),
            prompt=prompt,
            response_a=best["response"],
            response_b=worst["response"],
            gold_label=gold,
        ))

        if n and len(pairs) >= n:
            break

    logger.info("Loaded %d pairs from HelpSteer2 (%s split)", len(pairs), split)
    return pairs


# ---------------------------------------------------------------------------
# HH-RLHF loader (Anthropic)
# ---------------------------------------------------------------------------

def _load_hh_rlhf(split: str, n: Optional[int]) -> list[PairRecord]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    if n:
        ds = ds.select(range(min(n, len(ds))))

    pairs = []
    for i, row in enumerate(ds):
        chosen   = row.get("chosen", "")
        rejected = row.get("rejected", "")
        # Extract last assistant turn as the response
        def last_assistant(text: str) -> tuple[str, str]:
            """Returns (prompt_part, response_part)."""
            parts = text.split("\n\nAssistant:")
            response = parts[-1].strip() if len(parts) > 1 else text
            prompt   = "\n\nAssistant:".join(parts[:-1]).strip() if len(parts) > 1 else ""
            return prompt, response

        prompt_c, resp_chosen   = last_assistant(chosen)
        _,         resp_rejected = last_assistant(rejected)

        pairs.append(PairRecord(
            prompt_id=f"hh_{i:05d}",
            prompt=prompt_c,
            response_a=resp_chosen,
            response_b=resp_rejected,
            gold_label="A",               # HH-RLHF: chosen is always preferred
        ))

    logger.info("Loaded %d pairs from hh-rlhf (%s split)", len(pairs), split)
    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LOADERS = {
    "nvidia/HelpSteer2":                  _load_helpsteer2,
    "Anthropic/hh-rlhf":                  _load_hh_rlhf,
}


def load_dataset_pairs(
    dataset_name: str = "nvidia/HelpSteer2",
    split: str = "validation",
    n: Optional[int] = 200,
    seed: int = 42,
) -> list[PairRecord]:
    """
    Load a preference dataset and return normalised PairRecord list.

    Parameters
    ----------
    dataset_name : HuggingFace dataset identifier
    split        : dataset split ("train", "validation", "test")
    n            : max number of pairs to return (None = all)
    seed         : random seed for shuffling before truncation
    """
    if dataset_name not in LOADERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(LOADERS)}"
        )

    import random
    pairs = LOADERS[dataset_name](split, n=None)   # load all, then sample

    random.seed(seed)
    random.shuffle(pairs)

    if n:
        pairs = pairs[:n]

    return pairs
