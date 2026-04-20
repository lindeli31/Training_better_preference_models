"""
lookup_prompt.py
----------------
Look up one or more prompt_ids from the processed HelpSteer2 JSON and print
the original prompt + responses.

Usage
-----
    python lookup_prompt.py 1d7a53ffeb ce1271a1f3
    python lookup_prompt.py 1d7a53ffeb --split validation --full
    python lookup_prompt.py 1d7a53ffeb --max-chars 2000

Notes
-----
- prompt_id = md5(prompt_text)[:10], so multiple pairs can share the same id
  (all pair-combinations generated from the same prompt). This script prints
  each matching pair with its own response_a / response_b / gold_label.
- Default split is "train" because it has 10166 pairs; validation has 519.
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path("data")


def parse_args():
    p = argparse.ArgumentParser(description="Look up prompt(s) by prompt_id")
    p.add_argument("ids", nargs="+", help="One or more prompt_ids to look up")
    p.add_argument("--split", default="train",
                   help="Dataset split: train / validation (default: train)")
    p.add_argument("--full", action="store_true",
                   help="Print full text without truncation")
    p.add_argument("--max-chars", type=int, default=600,
                   help="Truncate each field to this many chars (default: 600)")
    return p.parse_args()


def _truncate(s: str, n: int, full: bool) -> str:
    if full or len(s) <= n:
        return s
    return s[:n] + f"\n... [truncated — full length: {len(s)} chars]"


def main():
    args = parse_args()
    path = DATA_DIR / f"helpsteer2_{args.split}.json"
    if not path.exists():
        print(f"ERROR: {path} does not exist. Download the dataset first:")
        print("       python -m src.dataset")
        return

    with open(path) as f:
        data = json.load(f)

    wanted = set(args.ids)
    matches: dict[str, list[dict]] = {pid: [] for pid in wanted}
    for row in data:
        if row["prompt_id"] in wanted:
            matches[row["prompt_id"]].append(row)

    for pid in args.ids:
        rows = matches[pid]
        print(f"\n{'=' * 80}")
        if not rows:
            print(f"prompt_id: {pid}  →  NOT FOUND in {path}")
            continue
        print(f"prompt_id: {pid}  |  {len(rows)} pair(s) for this prompt  |  split={args.split}")
        print(f"{'=' * 80}")
        print(f"\n[PROMPT]\n{_truncate(rows[0]['prompt'], args.max_chars, args.full)}")
        for i, r in enumerate(rows, 1):
            print(f"\n--- pair {i}/{len(rows)}  gold_label={r['gold_label']} ---")
            print(f"\n[response_a]\n{_truncate(r['response_a'], args.max_chars, args.full)}")
            print(f"\n[response_b]\n{_truncate(r['response_b'], args.max_chars, args.full)}")


if __name__ == "__main__":
    main()
