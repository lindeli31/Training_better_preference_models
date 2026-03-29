"""
dataset.py
----------
Download HelpSteer2 from HuggingFace and save processed pairs to data/.
Load pairs from local JSON files for experiments.
"""

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional
from datasets import load_dataset


DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class PairRecord:
    prompt_id: str
    prompt: str
    response_a: str
    response_b: str
    gold_label: str  # "A", "B", "C"

    def flipped(self) -> "PairRecord":
        flipped_label = {"A": "B", "B": "A", "C": "C"}[self.gold_label]
        return PairRecord(
            prompt_id=self.prompt_id + "_flipped",
            prompt=self.prompt,
            response_a=self.response_b,
            response_b=self.response_a,
            gold_label=flipped_label,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_id(text: str) -> str:
    '''Make a short ID from the prompt text'''
    return hashlib.md5(text.encode()).hexdigest()[:10]

def _score(row):
    '''Compute average score across helpfulness, correctness, coherence.'''
    return (row["helpfulness"] + row["correctness"] + row["coherence"]) / 3.0


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

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
                gold = "A"
            elif s_b > s_a:
                best, worst = b, a
                gold = "A"
            else:
                best, worst = a, b
                gold = "C"

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
# Load: data/*.json -> list[PairRecord]
# ---------------------------------------------------------------------------

def load_dataset_pairs(
    split: str = "validation",
    n: Optional[int] = 200,
    seed: int = 42,
) -> list[PairRecord]:
    """Load pairs from local JSON file in data/."""
    path = DATA_DIR / f"helpsteer2_{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = [PairRecord(**row) for row in data]

    random.seed(seed)
    random.shuffle(pairs)
    if n:
        pairs = pairs[:n]
    return pairs


# ---------------------------------------------------------------------------
# Run directly to download train + validation
# python -m src.dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for split in ["train", "validation"]:
        download_dataset(split)
        download_dataset(split, full=True)
