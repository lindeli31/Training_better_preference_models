"""
dataset.py
----------
Download HelpSteer2 from HuggingFace and save processed pairs to data/.
Load pairs from local JSON files for experiments.

Stratification
--------------
download_dataset() writes five stratification fields alongside every pair:
  score_gap       : |score_a - score_b| (float)
  difficulty      : "easy" / "medium" / "hard" derived from score_gap
  verbosity_delta : verbosity_a - verbosity_b (signed int); positive = better
                    response is more verbose
  complexity_max  : max(complexity_a, complexity_b) (int 0-4)
  is_multiturn    : True if the prompt contains a prior assistant response

load_dataset_pairs() exposes two sampling options:
  stratify=True          : proportional sampling across difficulty strata so
                           that hard/medium/easy pairs are equally represented
  randomize_position=True: randomly flip which response is placed in position A
                           so gold_label is ~50% A / ~50% B; this is required
                           for valid positional bias measurement (without it,
                           gold is always A and A-preference looks like accuracy)
"""

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, fields as dc_fields
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
    gold_label: str                        # "A", "B", "C"

    # Stratification metadata (None when loaded from legacy files)
    score_gap: Optional[float] = None      # |score_a - score_b|
    difficulty: Optional[str] = None       # "easy" / "medium" / "hard"
    verbosity_delta: Optional[int] = None  # verbosity_a - verbosity_b (signed)
    complexity_max: Optional[int] = None   # max(complexity_a, complexity_b)
    is_multiturn: Optional[bool] = None    # prior assistant turn in prompt?

    def flipped(self) -> "PairRecord":
        """Return a copy with A/B swapped and gold label inverted.

        verbosity_delta sign is negated to stay consistent with the new
        response_a / response_b assignment.  All other metadata is symmetric.
        """
        flipped_label = {"A": "B", "B": "A", "C": "C"}[self.gold_label]
        neg_vd = (-self.verbosity_delta) if self.verbosity_delta is not None else None
        return PairRecord(
            prompt_id=self.prompt_id + "_flipped",
            prompt=self.prompt,
            response_a=self.response_b,
            response_b=self.response_a,
            gold_label=flipped_label,
            score_gap=self.score_gap,
            difficulty=self.difficulty,
            verbosity_delta=neg_vd,
            complexity_max=self.complexity_max,
            is_multiturn=self.is_multiturn,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_id(text: str) -> str:
    """Short stable ID from prompt text."""
    return hashlib.md5(text.encode()).hexdigest()[:10]

def _score(row) -> float:
    """Composite quality score: mean of helpfulness, correctness, coherence."""
    return (row["helpfulness"] + row["correctness"] + row["coherence"]) / 3.0

def _detect_multiturn(prompt: str) -> bool:
    """Return True if the prompt contains a prior assistant response.

    HelpSteer2 multi-turn conversations include the full dialogue history in
    the prompt field.  Detecting an assistant marker indicates at least one
    prior exchange exists before the final user turn.
    """
    return bool(re.search(r"(?i)\bassistant\s*:", prompt))


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
    """Download HelpSteer2, build all pairwise combinations, save to data/.

    The standard file (full=False) now includes five stratification fields:
    score_gap, difficulty, verbosity_delta, complexity_max, is_multiturn.

    The full file (full=True) additionally includes per-response attribute
    scores, composite scores, and response lengths.
    """
    ds = load_dataset("nvidia/HelpSteer2", split=split)

    groups: dict[str, list] = defaultdict(list)
    for row in ds:
        groups[row["prompt"]].append(row)

    records = []
    for prompt, rows in groups.items():
        if len(rows) < 2:
            continue

        multiturn = _detect_multiturn(prompt)

        for a, b in combinations(rows, 2):
            s_a, s_b = _score(a), _score(b)

            if s_a > s_b:
                best, worst = a, b
            elif s_b > s_a:
                best, worst = b, a
            else:
                best, worst = a, b

            gap = abs(s_a - s_b)
            if gap > 1.0:
                difficulty = "easy"
            elif gap > 0.33:
                difficulty = "medium"
            else:
                difficulty = "hard"

            record = {
                "prompt_id": _make_id(prompt),
                "prompt": prompt,
                "response_a": best["response"],
                "response_b": worst["response"],
                "gold_label": "C" if s_a == s_b else "A",
                # stratification fields (always included)
                "score_gap": round(gap, 4),
                "difficulty": difficulty,
                "verbosity_delta": int(best["verbosity"]) - int(worst["verbosity"]),
                "complexity_max": int(max(best["complexity"], worst["complexity"])),
                "is_multiturn": multiturn,
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
                record["len_a"] = len(best["response"])
                record["len_b"] = len(worst["response"])

            records.append(record)

    suffix = "_full" if full else ""
    _save(records, DATA_DIR / f"helpsteer2_{split}{suffix}.json")
    return records


# ---------------------------------------------------------------------------
# Stratified sampling helper
# ---------------------------------------------------------------------------

def _stratified_sample(
    pairs: list[PairRecord], n: int, rng: random.Random
) -> list[PairRecord]:
    """Sample n pairs with proportional representation across difficulty strata.

    If a stratum has fewer pairs than its quota, the shortfall is filled from
    other strata.  Always returns exactly min(n, len(pairs)) records, shuffled.
    """
    strata: dict[str, list[PairRecord]] = defaultdict(list)
    for pair in pairs:
        strata[pair.difficulty or "unknown"].append(pair)

    for v in strata.values():
        rng.shuffle(v)

    stratum_names = sorted(strata)
    n_strata = len(stratum_names)
    base = n // n_strata
    remainder = n % n_strata

    sampled: list[PairRecord] = []
    for i, name in enumerate(stratum_names):
        quota = base + (1 if i < remainder else 0)
        sampled.extend(strata[name][:quota])

    # Top up from overflow pool if any stratum was under-quota
    if len(sampled) < n:
        selected_ids = {p.prompt_id for p in sampled}
        leftover = [p for p in pairs if p.prompt_id not in selected_ids]
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])

    rng.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# Load: data/*.json -> list[PairRecord]
# ---------------------------------------------------------------------------

_PAIR_FIELDS = {f.name for f in dc_fields(PairRecord)}


def load_dataset_pairs(
    split: str = "validation",
    n: Optional[int] = 200,
    seed: int = 42,
    stratify: bool = False,
    randomize_position: bool = False,
) -> list[PairRecord]:
    """Load pairs from local JSON file in data/.

    Args:
        split:             Dataset split ("train" / "validation").
        n:                 Maximum number of pairs to return (None = all).
        seed:              Random seed for reproducibility.
        stratify:          If True, sample proportionally across difficulty
                           strata (easy / medium / hard) instead of uniformly.
                           Requires re-running download_dataset() to generate
                           stratification metadata; falls back to random sampling
                           with a warning when metadata is absent.
        randomize_position: If True, randomly flip 50% of pairs so that
                           gold_label is ~50% A / ~50% B.  Required for valid
                           positional bias measurement — without this, gold is
                           always A and any A-preference in the judge looks like
                           accuracy rather than positional bias.

    Returns:
        list of PairRecord.  Extra JSON fields not in PairRecord are silently
        ignored (backward-compatible with legacy files).
    """
    path = DATA_DIR / f"helpsteer2_{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter to known fields so legacy files and full files both load cleanly
    pairs = [PairRecord(**{k: v for k, v in row.items() if k in _PAIR_FIELDS})
             for row in data]

    rng = random.Random(seed)

    if stratify:
        has_difficulty = any(p.difficulty is not None for p in pairs)
        if not has_difficulty:
            import warnings
            warnings.warn(
                "stratify=True but no difficulty labels found in the data file. "
                "Re-run `python -m src.dataset` to regenerate with stratification "
                "metadata.  Falling back to random sampling.",
                UserWarning,
                stacklevel=2,
            )
            rng.shuffle(pairs)
            pairs = pairs[:n] if n else pairs
        else:
            pairs = _stratified_sample(pairs, n if n else len(pairs), rng)
    else:
        rng.shuffle(pairs)
        if n:
            pairs = pairs[:n]

    if randomize_position:
        rng2 = random.Random(seed + 1337)  # separate stream for position flips
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
        pairs = result

    return pairs


# ---------------------------------------------------------------------------
# Run directly to download train + validation
# python -m src.dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for split in ["train", "validation"]:
        download_dataset(split)
        download_dataset(split, full=True)
