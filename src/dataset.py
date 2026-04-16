"""
dataset.py
----------
Tools to download HelpSteer2 from HuggingFace and save processed pairs to data/.
The processed data will have the following format:
- prompt_id: a short ID for the prompt (for tracking and debugging)
- prompt: the original prompt text
- response_a: the text of response A
- response_b: the text of response B
- gold_label: "A", "B", or "C" (if A is better, B is better, or tie)
If you run this file directly, it will download both train and validation splits and save them to data/helpsteer2_{split}.json (we have also saved the full version with all scores and metadata for analysis). You can then load the pairs for your experiments using the load_dataset_pairs function.

Remark: If you want to change the score, you can modify the _score function to compute a different score from the original helpfulness, correctness, and coherence. For example, you could give more weight to correctness if you think it's more important for your experiments.
"""

# Standard library imports
import csv
import hashlib
import json
import random
from collections import defaultdict

from itertools import combinations
from pathlib import Path
from typing import Optional
from datasets import load_dataset

# Where to save processed data
DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Data structure. Each data point of the experiment is composed by:
# - prompt_id: a short ID for the prompt (for tracking and debugging)
# - prompt: the original prompt text
# - response_a: the text of response A
# - response_b: the text of response B
# - gold_label: "A", "B", or "C" (if A is better, B is better, or tie)
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = ("easy", "medium", "hard")


class PairRecord:
    def __init__(self, prompt_id: str, prompt: str, response_a: str, response_b: str,
                 gold_label: str, difficulty: Optional[str] = None, **_kwargs):
        self.prompt_id = prompt_id
        self.prompt = prompt
        self.response_a = response_a
        self.response_b = response_b
        self.gold_label = gold_label
        self.difficulty = difficulty  # "easy", "medium", "hard" — only set when loaded from full dataset

    def flipped(self) -> "PairRecord":
        flipped_label = {"A": "B", "B": "A", "C": "C"}[self.gold_label]
        return PairRecord(
            prompt_id=self.prompt_id + "_flipped",
            prompt=self.prompt,
            response_a=self.response_b,
            response_b=self.response_a,
            gold_label=flipped_label,
            difficulty=self.difficulty,
        )

    def __repr__(self):
        return f"PairRecord(prompt_id={self.prompt_id!r}, gold_label={self.gold_label!r})"

    def __eq__(self, other):
        if not isinstance(other, PairRecord):
            return False
        return (self.prompt_id == other.prompt_id and self.prompt == other.prompt
                and self.response_a == other.response_a and self.response_b == other.response_b
                and self.gold_label == other.gold_label)

# ---------------------------------------------------------------------------
# Helper functions. Make IDs create an Id for each prompt, score(row) compute
# the score of the answer(average of helpfulness, correctness, coherence), 
# and save(records, path) save the records to JSON and CSV.
# ---------------------------------------------------------------------------

def _make_id(text: str) -> str:
    '''Make a short ID (10 characters) from the prompt text'''
    return hashlib.md5(text.encode()).hexdigest()[:10]

def _score(row):
    '''Compute average score across helpfulness, correctness, coherence.'''
    return (row["helpfulness"] + row["correctness"] + row["coherence"]) / 3.0

def _save(records: list[dict], path: Path):
    """Save a list of dicts to JSON + CSV."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} pairs to {path} and {csv_path}")

# ---------------------------------------------------------------------------
# Download: all pairwise combinations
# - Download HelpSteer2 from HuggingFace
# - Group by prompt, build all pairwise combinations of responses
# - Calculate the best and worst scores
# - Randomly assign best/worst to A/B to avoid position bias    
# - Save to data/helpsteer2_{split}.json (we have also saved the full version with all scores and metadata for analysis)
# ---------------------------------------------------------------------------

def download_dataset(split: str = "validation", full: bool = False):
    """Download HelpSteer2, build all pairwise combinations, save to data/."""
    ds = load_dataset("nvidia/HelpSteer2", split=split)

    groups: dict[str, list] = defaultdict(list)
    for row in ds:
        groups[row["prompt"]].append(row)

    records = []
    for prompt, rows in groups.items():
        if len(rows) < 2:
            continue

        for a, b in combinations(rows, 2):
            s_a, s_b = _score(a), _score(b)

            if s_a > s_b:
                best, worst = a, b
            elif s_b > s_a:
                best, worst = b, a
            else:
                best, worst = a, b

            # Randomly assign best/worst to position A/B to avoid position bias
            if s_a == s_b:
                gold = "C"
            elif random.random() < 0.5:
                best, worst = worst, best
                gold = "B"
            else:
                gold = "A"

            record = {
                "prompt_id": _make_id(prompt),
                "prompt": prompt,
                "response_a": best["response"],
                "response_b": worst["response"],
                "gold_label": gold,
            }

            if full:
                record["helpfulness_a"] = best["helpfulness"]
                record["correctness_a"] = best["correctness"]
                record["coherence_a"] = best["coherence"]
                record["complexity_a"] = best["complexity"]
                record["verbosity_a"] = best["verbosity"]
                record["score_a"] = round(_score(best), 4)
                record["helpfulness_b"] = worst["helpfulness"]
                record["correctness_b"] = worst["correctness"]
                record["coherence_b"] = worst["coherence"]
                record["complexity_b"] = worst["complexity"]
                record["verbosity_b"] = worst["verbosity"]
                record["score_b"] = round(_score(worst), 4)
                record["score_gap"] = round(abs(s_a - s_b), 4)
                record["len_a"] = len(best["response"])
                record["len_b"] = len(worst["response"])
                gap = abs(s_a - s_b)
                if gap > 1.0:
                    record["difficulty"] = "easy"
                elif gap > 0.33:
                    record["difficulty"] = "medium"
                else:
                    record["difficulty"] = "hard"

            records.append(record)

    suffix = "_full" if full else ""
    _save(records, DATA_DIR / f"helpsteer2_{split}{suffix}.json")
    return records


# ---------------------------------------------------------------------------
# load_dataset_pairs  (useful when you want to run the experiments)
# - Load pairs from local JSON file in data/
# - Shuffle and optionally subsample for experiments (if you do not want to run on all pairs n: Optional[int] = 200),
# - Return a list of PairRecord objects (see data structure above)
# ---------------------------------------------------------------------------

def load_dataset_pairs(
    split: str = "validation",
    n: Optional[int] = 200,
    seed: int = 42,
    difficulty: Optional[str] = None,
) -> list[PairRecord]:
    """Load pairs from local JSON file in data/.

    If difficulty is given ("easy", "medium", "hard"), loads the full dataset
    variant (which carries the difficulty tag) and filters before sampling.
    """
    if difficulty is not None and difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"difficulty must be one of {DIFFICULTY_LEVELS}, got {difficulty!r}")

    use_full = difficulty is not None
    suffix = "_full" if use_full else ""
    path = DATA_DIR / f"helpsteer2_{split}{suffix}.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = [PairRecord(**row) for row in data]

    if difficulty is not None:
        pairs = [p for p in pairs if p.difficulty == difficulty]

    random.seed(seed)
    random.shuffle(pairs)
    if n:
        pairs = pairs[:n]
    return pairs


# ---------------------------------------------------------------------------
# Run directly to download train + validation
# python -m src.dataset 
# when you are in the root of the project.
# Ater running this, you can visualize dataset with dataset.analysis.ipynb.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for split in ["train", "validation"]:
        download_dataset(split)
        download_dataset(split, full=True)
