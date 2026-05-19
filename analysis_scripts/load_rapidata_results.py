"""
load_rapidata_results.py
------------------------
Fetch a completed Rapidata pairwise-comparison job and convert its results
into a HelpSteer2-compatible JSON file usable by the existing pipeline
(load_dataset_pairs / load_stratified_pairs / B1 / B3 / B4).

Inputs
------
- The sidecar saved at job-creation time:
      data/rapidata_jobs/<job_name>.json
  containing per-datapoint metadata (prompt_id, gold_helpsteer, difficulty,
  score_gap, ...). Produced by build_rapidata_dataset.py.

- The Rapidata API itself, which we query with the job_id stored in the
  sidecar.

Per-pair processing
-------------------
For each pair we collect the votes from the 20 (or however many) raters
and compute:

- counts: votes_A, votes_B  (we treat the comparison as binary; the UI
  forces A or B; we count any non-A as B)
- gold_label_human:
      "A"  if votes_A >  votes_B
      "B"  if votes_B >  votes_A
      "C"  if votes_A == votes_B  (rare, only on even N)
- agreement: max(votes_A, votes_B) / total_votes  ∈ [0.5, 1]
- difficulty_human: a four-bucket label derived from agreement
      easy   if agreement ≥ 0.85
      medium if 0.70 ≤ agreement < 0.85
      hard   if 0.55 ≤ agreement < 0.70
      tie    if agreement < 0.55

Output
------
A JSON file in the same shape as data/helpsteer2_train_full.json, augmented
with two new fields per record:

    "gold_label_human":   "A" | "B" | "C"
    "difficulty_human":   "easy" | "medium" | "hard" | "tie"
    "human_agreement":    float in [0, 1]
    "votes_a":            int
    "votes_b":            int

Default path:
    data/rapidata_helpsteer_<job_name>.json

Once written, you can load these pairs with the existing
load_dataset_pairs / load_stratified_pairs by pointing them to this file
(or by adding a small loader that prefers gold_label_human / difficulty_human
over the original HelpSteer2 fields — see README.md).

Usage
-----
    python analysis_scripts/load_rapidata_results.py \\
        --job-name helpsteer2_b1_validation
    python analysis_scripts/load_rapidata_results.py \\
        --job-name helpsteer2_b1_validation --output data/my_eval.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Make 'src' importable + load .env so RapidataClient picks up credentials.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_SIDECAR_DIR = Path("data/rapidata_jobs")
DEFAULT_HELPSTEER_PATH = Path("data/helpsteer2_train_full.json")
DEFAULT_OUTPUT_TEMPLATE = "data/rapidata_helpsteer_{job_name}.json"

# Agreement → bucket thresholds.
# Min possible agreement is 0.5 (perfect 50/50 split between A and B).
DIFFICULTY_THRESHOLDS = (
    ("easy",   0.85),  # agreement >= 0.85
    ("medium", 0.70),  # 0.70 <= agreement < 0.85
    ("hard",   0.55),  # 0.55 <= agreement < 0.70
    ("tie",    0.0),   # agreement < 0.55
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-name", required=True,
                   help="Job name as used by build_rapidata_dataset.py "
                        "(matches the sidecar filename without .json).")
    p.add_argument("--sidecar-dir", type=Path, default=DEFAULT_SIDECAR_DIR,
                   help=f"Directory holding the job sidecar (default {DEFAULT_SIDECAR_DIR}).")
    p.add_argument("--helpsteer-path", type=Path, default=DEFAULT_HELPSTEER_PATH,
                   help=f"HelpSteer2 train JSON to merge into "
                        f"(default {DEFAULT_HELPSTEER_PATH}).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path (default: data/rapidata_helpsteer_<job_name>.json).")
    p.add_argument("--results-cache", type=Path, default=None,
                   help="Path to a previously fetched raw results JSON. "
                        "If given, we skip the Rapidata API call.")
    p.add_argument("--job-id", default=None,
                   help="Override the job_id stored in the sidecar. Use this "
                        "when the order was started from the Rapidata dashboard "
                        "(after running build_rapidata_dataset.py without "
                        "--assign): copy the job_id from the dashboard URL or "
                        "the order page.")
    return p.parse_args()


def bucket_from_agreement(agreement: float) -> str:
    for name, threshold in DIFFICULTY_THRESHOLDS:
        if agreement >= threshold:
            return name
    return "tie"


def _build_client():
    """Construct a RapidataClient, preferring env-var auth."""
    import os
    from rapidata import RapidataClient
    cid = os.environ.get("RAPIDATA_CLIENT_ID")
    csec = os.environ.get("RAPIDATA_CLIENT_SECRET")
    if cid and csec:
        return RapidataClient(client_id=cid, client_secret=csec)
    return RapidataClient()


def fetch_results(job_id: str) -> dict:
    """Read-only fetch of job/order results by id.

    The Rapidata SDK exposes two parallel endpoints:
      - client.job.get_job_by_id        (for jobs created via
        audience.assign_job(...))
      - client.order.get_order_by_id    (for orders created from the
        dashboard or via client.order.create(...); their ids start with
        'ord_' as in 'ord_1InxNKx...')
    We try the matching one based on the prefix, falling back to the
    other if it fails.
    """
    client = _build_client()
    looks_like_order = isinstance(job_id, str) and job_id.startswith("ord_")
    primary, secondary = (
        ("order", "job") if looks_like_order else ("job", "order")
    )
    last_err = None
    for endpoint in (primary, secondary):
        try:
            if endpoint == "order":
                obj = client.order.get_order_by_id(job_id)
            else:
                obj = client.job.get_job_by_id(job_id)
            return obj.get_results()
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Failed to fetch results for id {job_id!r} from both job and "
        f"order endpoints. Last error: {last_err}"
    )


def find_job_id_by_name(job_name: str) -> str:
    """Find a running Rapidata job/order whose name matches `job_name`.

    Tries client.job.find_jobs and client.order.find_orders. When the same
    task surfaces as both a Job and an Order (common: the Order is the
    canonical run, the Job is its audience binding), we prefer the
    Order (`ord_*` id). If multiple distinct tasks match, we still raise
    so the caller disambiguates explicitly via --job-id.
    """
    client = _build_client()
    candidates: list = []
    try:
        candidates.extend(client.job.find_jobs(job_name))
    except Exception:
        pass
    finder = getattr(getattr(client, "order", None), "find_orders", None)
    if callable(finder):
        try:
            candidates.extend(finder(job_name))
        except Exception:
            pass
    if not candidates:
        raise RuntimeError(
            f"No Rapidata job/order found by name {job_name!r}. "
            f"Did you start the order from the dashboard? "
            f"(Or pass --job-id explicitly.)"
        )

    def _id(obj):
        return getattr(obj, "id", None) or getattr(obj, "job_id", None)

    ids = [_id(o) for o in candidates]
    orders = [i for i in ids if i and i.startswith("ord_")]
    jobs = [i for i in ids if i and i.startswith("job_")]

    # Common SDK quirk: the same task is exposed as both a Job and an
    # Order with the same name. Prefer the Order in that case.
    if len(orders) == 1 and len(jobs) == 1 and len(ids) == 2:
        chosen = orders[0]
        print(f"  Found 1 Order + 1 Job; preferring Order {chosen}.")
        return chosen
    if len(ids) == 1:
        return ids[0]
    raise RuntimeError(
        f"Found {len(candidates)} Rapidata jobs/orders named "
        f"{job_name!r}: {ids}. Pick one and pass it via --job-id."
    )


_NEITHER_BOTH_KEYS = {
    "neither", "both", "tie", "equal", "unsure",
    "neither / both", "neither/both",
}


def _split_aggregated(aggregated: dict, option_a_text: str,
                      option_b_text: str) -> tuple[int, int, int]:
    """
    Split an aggregatedResults / summedUserScores dict into three buckets:
    A votes, B votes, and "neither / both / tie" votes (the extra option
    enabled by AllowNeitherBothSetting).

    Returns (votes_a, votes_b, votes_c) as numeric values.
    The dict values may be int (aggregatedResults) or float
    (summedUserScores); we preserve the type via float() and let the caller
    cast as needed.
    """
    if not aggregated:
        return 0, 0, 0

    a = aggregated.get(option_a_text)
    b = aggregated.get(option_b_text)
    matched = {option_a_text, option_b_text} & set(aggregated)

    # Defensive fallback when the keys don't match exactly: take the two
    # entries with the largest text overlap with our submitted options
    # (ignored here for simplicity — if this triggers, something is off).
    if a is None or b is None:
        # As a last resort, assume the first two keys map to A/B in
        # submission order.
        keys = list(aggregated.keys())
        a = float(aggregated[keys[0]]) if len(keys) >= 1 else 0
        b = float(aggregated[keys[1]]) if len(keys) >= 2 else 0
        matched = set(keys[:2])

    # Anything not matching A or B is bucketed into "C" (Neither / Both /
    # tie option from AllowNeitherBothSetting).
    c = 0.0
    for k, v in aggregated.items():
        if k in matched:
            continue
        # Only count keys that look like an explicit neither/both choice;
        # otherwise treat as A/B mismatch (already handled).
        if isinstance(k, str) and k.strip().lower() in _NEITHER_BOTH_KEYS:
            c += float(v)
        else:
            # Unrecognised extra key — sum into C anyway, but warn.
            print(f"WARNING: unrecognised aggregatedResults key {k!r} "
                  f"with value {v!r}; counting as 'C'.")
            c += float(v)

    return float(a), float(b), float(c)


def parse_result_entry(entry: dict, option_a_text: str, option_b_text: str
                       ) -> tuple[int, int, int, float, float, float, str]:
    """
    Parse one Rapidata result entry into:
        (votes_a, votes_b, votes_c, weighted_a, weighted_b, weighted_c,
         gold_human)

    Where votes_* are raw rater counts (votes_c bundles "Neither / Both"
    votes, present when AllowNeitherBothSetting is enabled), weighted_*
    are summedUserScores (rater-reliability-weighted), and gold_human is
    the winner label derived from raw counts:
        "A" if votes_a is the unique max
        "B" if votes_b is the unique max
        "C" if votes_c is the unique max OR there is a tie at the top

    Schema reference:
        results[i].aggregatedResults: {option_text: int_count, ...}
        results[i].summedUserScores:  {option_text: float_weighted, ...}
        results[i].winner_index:      0 | 1 (computed by Rapidata from
                                             summedUserScores; we re-derive
                                             our gold from raw counts so it
                                             matches the agreement metric.)
    """
    ag = entry.get("aggregatedResults") or {}
    sw = entry.get("summedUserScores") or {}

    a, b, c = _split_aggregated(ag, option_a_text, option_b_text)
    wa, wb, wc = _split_aggregated(sw, option_a_text, option_b_text)
    votes_a, votes_b, votes_c = int(round(a)), int(round(b)), int(round(c))

    triples = [(votes_a, "A"), (votes_b, "B"), (votes_c, "C")]
    triples.sort(key=lambda x: -x[0])
    if triples[0][0] == triples[1][0]:
        # Tie at the top → call it "C" (ambiguous human gold).
        gold_human = "C"
    else:
        gold_human = triples[0][1]

    return (votes_a, votes_b, votes_c,
            float(wa), float(wb), float(wc), gold_human)


def main() -> None:
    args = parse_args()

    sidecar_path = args.sidecar_dir / f"{args.job_name}.json"
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Sidecar not found at {sidecar_path}. Did build_rapidata_dataset.py "
            f"finish successfully? Try also looking for "
            f"{args.job_name}.dry_run.json (dry-run sidecar)."
        )

    with open(sidecar_path) as f:
        sidecar = json.load(f)

    # Resolution order for the job_id we need to query Rapidata:
    #   1. --job-id (CLI override)
    #   2. sidecar value (set when build_rapidata_dataset.py ran with --assign)
    #   3. find_jobs(job_name) on Rapidata (when the order was started
    #      from the dashboard)
    job_id = args.job_id or sidecar.get("job_id")
    if not job_id and not args.results_cache:
        print(f"No job_id in --job-id or sidecar. Searching Rapidata for a "
              f"job named {args.job_name!r}...")
        try:
            job_id = find_job_id_by_name(args.job_name)
            print(f"  Found job_id = {job_id}")
        except Exception as e:
            raise RuntimeError(
                f"Could not auto-resolve a job_id for {args.job_name!r}: {e}\n"
                f"Open https://app.rapidata.ai/dashboard, find the running "
                f"order for this job, copy its id, and re-run with "
                f"--job-id <id>."
            )

    # Fetch (or load cached) raw results.
    if args.results_cache and args.results_cache.exists():
        with open(args.results_cache) as f:
            raw = json.load(f)
        print(f"Loaded cached raw results from {args.results_cache}.")
    else:
        print(f"Fetching results for job_id={job_id}...")
        raw = fetch_results(job_id)
        # Cache them next to the sidecar so we don't re-hit the API.
        cache_path = args.results_cache or sidecar_path.with_suffix(".raw_results.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        print(f"Saved raw results → {cache_path}")

    # Rapidata returns a top-level dict with keys "info", "summary",
    # "results". Each entry in "results" corresponds to one datapoint and
    # is in the same order as we submitted. We index it by position.
    if isinstance(raw, dict) and "results" in raw:
        results_list = raw["results"]
    elif isinstance(raw, list):
        results_list = raw
    else:
        raise RuntimeError(
            f"Unexpected raw results shape: {type(raw).__name__}. "
            f"Expected the documented Rapidata schema with a 'results' list."
        )

    if len(results_list) != len(sidecar["datapoints"]):
        print(f"NOTE: {len(results_list)} result entries vs "
              f"{len(sidecar['datapoints'])} sidecar datapoints. "
              f"Will match by `context` (prompt), with index as fallback.")

    # Build a lookup from option_a_text -> sidecar entry. The option text
    # is unique per pair (it embeds prompt + response_a) and is preserved
    # verbatim by Rapidata as a key in aggregatedResults / summedUserScores.
    # That makes it the most reliable join key when upload failures or
    # platform-side reordering shift the index.
    by_option_a: dict = {dp["option_a_text"]: dp for dp in sidecar["datapoints"]}

    def lookup_sidecar(entry: dict, fallback_idx: int):
        ag = entry.get("aggregatedResults") or {}
        for key in ag:
            dp = by_option_a.get(key)
            if dp is not None:
                return dp
        if 0 <= fallback_idx < len(sidecar["datapoints"]):
            return sidecar["datapoints"][fallback_idx]
        return None

    human_by_pid = {}
    for i, entry in enumerate(results_list):
        dp = lookup_sidecar(entry, i)
        if dp is None:
            print(f"WARNING: result entry {i} could not be matched to a "
                  f"sidecar datapoint; skipping.")
            continue
        pid = dp["prompt_id"]
        a, b, c, wa, wb, wc, gold_human = parse_result_entry(
            entry, dp["option_a_text"], dp["option_b_text"]
        )
        total = a + b + c
        if total == 0:
            print(f"WARNING: zero votes for prompt_id={pid}; skipping.")
            continue
        # Agreement = share of the top option (A, B or C) — works whether
        # the panel landed on a winner or on the Neither/Both option.
        agreement = max(a, b, c) / total
        human_by_pid[pid] = {
            "gold_label_human": gold_human,
            "difficulty_human": bucket_from_agreement(agreement),
            "human_agreement": round(agreement, 4),
            "votes_a": a,
            "votes_b": b,
            "votes_c": c,
            "weighted_a": round(wa, 4),
            "weighted_b": round(wb, 4),
            "weighted_c": round(wc, 4),
        }

    print(f"Built human metrics for {len(human_by_pid)} pairs.")

    # Merge into HelpSteer2 records, keeping only the labelled subset.
    with open(args.helpsteer_path) as f:
        helpsteer = json.load(f)
    keep = []
    for r in helpsteer:
        h = human_by_pid.get(r["prompt_id"])
        if h is None:
            continue
        merged = {**r, **h}
        keep.append(merged)
    print(f"Merged into {len(keep)} HelpSteer2 records.")

    out_path = args.output or Path(DEFAULT_OUTPUT_TEMPLATE.format(job_name=args.job_name))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_path}")

    # Quick descriptive stats
    print("\n=== Summary ===")
    by_diff_h = Counter(r["difficulty_human"] for r in keep)
    print("Pairs per human-difficulty bucket:")
    for k in ("easy", "medium", "hard", "tie"):
        print(f"  {k:<8} {by_diff_h.get(k, 0)}")

    # Agreement between HelpSteer score-based gold and human gold
    agree_ab = sum(1 for r in keep
                   if r["gold_label_human"] == r["gold_label"]
                   and r["gold_label"] in ("A", "B"))
    n_ab = sum(1 for r in keep if r["gold_label"] in ("A", "B"))
    if n_ab:
        print(f"\nAgreement HelpSteer score-gold vs human gold (non-tie pairs): "
              f"{agree_ab}/{n_ab} = {agree_ab/n_ab:.3f}")


if __name__ == "__main__":
    main()
