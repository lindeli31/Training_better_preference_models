"""
build_rapidata_dataset.py
-------------------------
Submit a Rapidata pairwise-comparison job over a stratified subset of
HelpSteer2 train, so that a panel of human raters labels each pair (A vs B)
~20 times. The resulting agreement distribution is used downstream to:

  1. Cross-check our HelpSteer2 score-based gold against direct human
     preferences.
  2. Re-bucket pairs by *human agreement* instead of score gap (more
     robust definition of "easy" / "medium" / "hard" / "tie").

Sampling
--------
We keep the four-bucket scheme (tie / hard / medium / easy) defined in
src/datasets/dataset.py and draw an equal number of pairs from each:

    n_per_bucket = N_PAIRS // 4   (default N_PAIRS=1248 → 312 per bucket)
    20 ratings/pair × 1248 pairs = 24,960 responses → fits the 25k Rapidata
    free tier.

The deterministic seed mirrors the rest of the codebase (default 42).

Datapoint format
----------------
Each Rapidata datapoint is a 2-element list of strings:

    [
      "Prompt:\\n<prompt>\\n\\nResponse:\\n<response_a>",
      "Prompt:\\n<prompt>\\n\\nResponse:\\n<response_b>"
    ]

so the rater sees the prompt at the top of each option (small redundancy is
preferable to ambiguity about which prompt the response is answering).

Local sidecar
-------------
After job creation we save:

    data/rapidata_jobs/<job_name>.json

containing job_id + the (prompt_id, gold_label, difficulty, score_gap,
position_a, position_b) mapping for every datapoint, indexed by datapoint
position. load_rapidata_results.py uses this to reconstruct per-pair
metadata when the job completes.

Usage
-----
    pip install -U rapidata
    export RAPIDATA_API_KEY=...                 # whatever Rapidata expects
    python analysis_scripts/build_rapidata_dataset.py \\
        --n-pairs 1248 \\
        --responses-per-datapoint 20 \\
        --job-name helpsteer2_b1_validation \\
        --preview                              # opens the UI preview before submitting
        --dry-run                              # builds + saves sidecar but does not submit

After the job is created, monitor it via the Rapidata dashboard or the
returned job_id. When it completes, run load_rapidata_results.py.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make 'src' importable regardless of how this file is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load credentials from .env so RAPIDATA_CLIENT_ID / RAPIDATA_CLIENT_SECRET
# (or the existing rapidata credentials cache at ~/.config/rapidata/credentials.json)
# are picked up automatically.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.datasets.dataset import DIFFICULTY_LEVELS, load_dataset_pairs


def _build_rapidata_client():
    """
    Build a RapidataClient using one of the supported auth flows:

    1. RAPIDATA_CLIENT_ID + RAPIDATA_CLIENT_SECRET env vars (programmatic).
    2. The cached credentials at ~/.config/rapidata/credentials.json
       (set up the first time RapidataClient() runs interactively).
    3. Otherwise the SDK opens a browser window for interactive login.
    """
    from rapidata import RapidataClient
    cid = os.environ.get("RAPIDATA_CLIENT_ID")
    csec = os.environ.get("RAPIDATA_CLIENT_SECRET")
    if cid and csec:
        return RapidataClient(client_id=cid, client_secret=csec)
    return RapidataClient()

DEFAULT_N_PAIRS = 1248          # 312 / bucket × 4 buckets
DEFAULT_RESPONSES = 20          # 1248 × 20 = 24,960 ≤ 25,000 free tier
DEFAULT_OUTPUT_DIR = Path("data/rapidata_jobs")

# Rapidata limits (as of 2025) — text datapoints are capped at 4096 chars.
RAPIDATA_TEXT_LIMIT = 4096


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS,
                   help=f"Total pairs to label (default {DEFAULT_N_PAIRS}, "
                        f"divisible by 4 for equal stratification).")
    p.add_argument("--responses-per-datapoint", type=int, default=DEFAULT_RESPONSES,
                   help=f"Human ratings per pair (default {DEFAULT_RESPONSES}).")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed (default 42 — same as the rest of the codebase).")
    p.add_argument("--job-name", default=None,
                   help="Rapidata job name (default: helpsteer2_b1_<UTC timestamp>).")
    p.add_argument("--audience-id", default="global",
                   help="Rapidata audience id (default 'global').")
    p.add_argument("--instruction",
                   default="Which response better answers the prompt?",
                   help="Instruction shown to raters. Keep it short — each "
                        "labeler session is capped at 25 seconds.")
    p.add_argument("--a-b-names", nargs=2, default=["Response A", "Response B"],
                   metavar=("LEFT", "RIGHT"),
                   help="Custom labels for the two options shown on the buttons "
                        "(default: 'Response A' 'Response B').")
    p.add_argument("--confidence-threshold", type=float, default=None,
                   help="If set (e.g. 0.99), Rapidata will stop collecting "
                        "responses early when statistical consensus is reached "
                        "(uses labeler trust scores). Saves credits on clear-cut "
                        "pairs. Mutually exclusive with --quorum-threshold. "
                        "Default: collect all --responses-per-datapoint responses.")
    p.add_argument("--quorum-threshold", type=int, default=None,
                   help="If set (e.g. 14), Rapidata will stop collecting "
                        "responses once that many raters agree on the same "
                        "answer. Simpler than --confidence-threshold but "
                        "less statistically grounded. Mutually exclusive with "
                        "--confidence-threshold. Note: for our use case (where "
                        "the 'agreement ratio' metric assumes a comparable N "
                        "across pairs), early stopping is OFF by default.")
    p.add_argument("--no-allow-neither-both", action="store_true",
                   help="Disable the 'Neither / Both' button. By default, raters "
                        "can express tied preferences, which is essential for "
                        "honest agreement on the tie bucket.")
    p.add_argument("--no-markdown", action="store_true",
                   help="Disable Markdown rendering of the response text.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Local sidecar directory (default {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--preview", action="store_true",
                   help="Open the Rapidata UI preview after creating the job "
                        "definition.")
    p.add_argument("--assign", action="store_true",
                   help="Programmatically assign the job to the audience and "
                        "start the order. Default: OFF — the script creates "
                        "the job definition, shows the preview, and stops, "
                        "leaving you to start the order from the Rapidata "
                        "dashboard. This avoids unintended credit spend.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the job + save the sidecar, but do NOT call any "
                        "Rapidata API (useful for testing the sampling logic).")
    return p.parse_args()


def _pair_fits_rapidata(pair) -> bool:
    """True if both rendered datapoint strings fit within Rapidata's text limit."""
    a, b = render_datapoint(pair)
    return len(a) <= RAPIDATA_TEXT_LIMIT and len(b) <= RAPIDATA_TEXT_LIMIT


def stratified_sample(n_total: int, seed: int) -> list:
    """Return n_total // 4 pairs from each of the four buckets, deterministic.

    Pairs whose rendered datapoint would exceed Rapidata's text limit are
    skipped at sampling time, and we oversample from the same bucket to
    backfill them. If a bucket's pool is exhausted before reaching the
    target, we emit a warning and return what is available.
    """
    if n_total % 4 != 0:
        raise ValueError(f"--n-pairs must be divisible by 4, got {n_total}")
    n_per_bucket = n_total // 4

    pairs = []
    for difficulty in DIFFICULTY_LEVELS:
        # Pull the entire bucket so we can backfill within it.
        bucket_pairs = load_dataset_pairs(
            split="train", n=None, seed=seed,
            difficulty=difficulty, full=True,
        )
        kept = []
        skipped = 0
        for p in bucket_pairs:
            if len(kept) >= n_per_bucket:
                break
            if _pair_fits_rapidata(p):
                kept.append(p)
            else:
                skipped += 1
        if len(kept) < n_per_bucket:
            print(f"WARNING: bucket {difficulty!r}: kept {len(kept)} pairs, "
                  f"skipped {skipped} (over {RAPIDATA_TEXT_LIMIT} chars), "
                  f"could not reach the target of {n_per_bucket}.")
        elif skipped:
            print(f"  bucket {difficulty!r}: kept {len(kept)} pairs "
                  f"(skipped {skipped} oversize, backfilled).")
        pairs.extend(kept)

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs


def render_datapoint(pair) -> list[str]:
    """Build a 2-element list of strings as Rapidata expects.

    The prompt is embedded directly in each option because Rapidata's
    `contexts=` parameter caps each context at 400 chars, while many
    HelpSteer prompts are longer (median 237 char, p75 ≈ 1100, max
    ~4900). Embedding inside the datapoint avoids the limit. The cost
    is that the prompt is shown twice (once per option), but with
    `MarkdownSetting` the layout is clean enough.
    """
    p = pair.prompt.strip()
    a = pair.response_a.strip()
    b = pair.response_b.strip()
    return [
        f"**Prompt:**\n{p}\n\n**Response:**\n{a}",
        f"**Prompt:**\n{p}\n\n**Response:**\n{b}",
    ]


def main() -> None:
    args = parse_args()
    if args.confidence_threshold is not None and args.quorum_threshold is not None:
        raise SystemExit(
            "ERROR: --confidence-threshold and --quorum-threshold are "
            "mutually exclusive (Rapidata supports only one early-stopping "
            "strategy at a time). Pass exactly one, or neither."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_name = args.job_name or f"helpsteer2_b1_{timestamp}"

    pairs = stratified_sample(args.n_pairs, args.seed)
    print(f"Sampled {len(pairs)} pairs from {len(DIFFICULTY_LEVELS)} buckets.")

    # Each datapoint is a 2-element list with the prompt embedded inside
    # both options (see render_datapoint for the rationale). We do NOT
    # pass `contexts=` because Rapidata caps it at 400 chars and many
    # HelpSteer prompts exceed that.
    datapoints = [render_datapoint(p) for p in pairs]

    # Sidecar metadata: per-datapoint mapping, indexed by position in the
    # job. We persist the rendered option strings so
    # load_rapidata_results.py can match Rapidata's result keys
    # (= the literal datapoint texts for text jobs) back to A/B.
    sidecar = {
        "job_name": job_name,
        "audience_id": args.audience_id,
        "instruction": args.instruction,
        "a_b_names": list(args.a_b_names),
        "responses_per_datapoint": args.responses_per_datapoint,
        "confidence_threshold": args.confidence_threshold,
        "quorum_threshold": args.quorum_threshold,
        "allow_neither_both": not args.no_allow_neither_both,
        "markdown": not args.no_markdown,
        "seed": args.seed,
        "n_pairs": len(pairs),
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "datapoints": [
            {
                "index": i,
                "prompt_id": p.prompt_id,
                "prompt": p.prompt.strip(),
                "gold_label_helpsteer": p.gold_label,
                "difficulty_helpsteer": p.difficulty,
                "score_gap": p.score_gap,
                "option_a_text": dp[0],   # = pair.response_a (stripped)
                "option_b_text": dp[1],   # = pair.response_b (stripped)
            }
            for i, (p, dp) in enumerate(zip(pairs, datapoints))
        ],
    }

    max_responses = len(datapoints) * args.responses_per_datapoint
    if args.confidence_threshold is not None:
        note = (f"(early stopping ON: confidence_threshold="
                f"{args.confidence_threshold}; actual responses may be fewer)")
    elif args.quorum_threshold is not None:
        note = (f"(early stopping ON: quorum_threshold="
                f"{args.quorum_threshold}; actual responses may be fewer)")
    else:
        note = "(no early stopping: exactly this many responses)"
    print(f"Job size: {len(datapoints)} datapoints × "
          f"{args.responses_per_datapoint} responses = up to "
          f"{max_responses} responses {note}.")

    if args.dry_run:
        print(f"[DRY RUN] Would create Rapidata job '{job_name}'.")
        sidecar_path = args.output_dir / f"{job_name}.dry_run.json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
        print(f"Saved sidecar (dry-run) → {sidecar_path}")
        return

    # Lazy import so dry-run works without rapidata installed.
    settings_list: list = []
    if not args.no_markdown:
        from rapidata import MarkdownSetting
        settings_list.append(MarkdownSetting())
    if not args.no_allow_neither_both:
        from rapidata import AllowNeitherBothSetting
        settings_list.append(AllowNeitherBothSetting())

    client = _build_rapidata_client()
    audience = client.audience.get_audience_by_id(args.audience_id)

    job_kwargs: dict = dict(
        name=job_name,
        instruction=args.instruction,
        datapoints=datapoints,
        responses_per_datapoint=args.responses_per_datapoint,
        data_type="text",
        a_b_names=list(args.a_b_names),
        settings=settings_list,
    )
    if args.confidence_threshold is not None:
        job_kwargs["confidence_threshold"] = args.confidence_threshold
    elif args.quorum_threshold is not None:
        job_kwargs["quorum_threshold"] = args.quorum_threshold

    # Catch FailedUploadException: for text jobs upload failures should be
    # rare (no asset upload), but if any datapoint is rejected for format
    # reasons the SDK still returns a partial job_definition. We save the
    # full sidecar (with all attempted pairs) and let load_rapidata_results
    # match results by `context` so missing entries are simply skipped.
    try:
        from rapidata.rapidata_client.exceptions import FailedUploadException
    except ImportError:
        FailedUploadException = None  # type: ignore

    try:
        job_definition = client.job.create_compare_job_definition(**job_kwargs)
        partial = False
    except Exception as e:
        if FailedUploadException is not None and isinstance(e, FailedUploadException):
            partial = True
            failures = e.failures_by_reason
            print(f"WARNING: {len(e.failed_uploads)} datapoints failed to "
                  f"upload. Continuing with the partial job_definition.")
            for reason, items in failures.items():
                print(f"  [{len(items):>3}] {reason}")
            sidecar["upload_failures"] = [
                {"reason": reason, "n_items": len(items)}
                for reason, items in failures.items()
            ]
            job_definition = e.job_definition
            if job_definition is None:
                raise
        else:
            raise

    sidecar["job_definition_id"] = (
        getattr(job_definition, "id", None)
        or getattr(job_definition, "job_definition_id", None)
    )
    sidecar["partial_upload"] = partial

    if args.preview:
        print("Opening preview in browser to review the job definition.")
        job_definition.preview()

    if args.assign:
        # User explicitly opted in to programmatic order creation.
        input("About to assign the job to the audience and start the order. "
              "Press enter to confirm (Ctrl-C to abort)...")
        job = audience.assign_job(job_definition)
        sidecar["job_id"] = (
            getattr(job, "id", None) or getattr(job, "job_id", None)
        )

    sidecar_path = args.output_dir / f"{job_name}.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
    print(f"\nSaved sidecar → {sidecar_path}")
    print(f"  job_definition_id: {sidecar['job_definition_id']}")
    if "job_id" in sidecar:
        print(f"  job_id:            {sidecar['job_id']}")
        print(f"\nOrder is running. Monitor at https://app.rapidata.ai/dashboard")
        print(f"When complete, fetch results with:")
        print(f"  python analysis_scripts/load_rapidata_results.py "
              f"--job-name {job_name}")
    else:
        print(f"\nJob definition created but NOT yet assigned to a labeling order.")
        print(f"Next steps:")
        print(f"  1. Open https://app.rapidata.ai/dashboard")
        print(f"     Find the job definition (search by name '{job_name}') "
              f"and start an order from it.")
        print(f"  2. Once the order completes, fetch its job_id from the "
              f"dashboard and run:")
        print(f"       python analysis_scripts/load_rapidata_results.py "
              f"--job-name {job_name} --job-id <job_id from dashboard>")
        print(f"     (Or rerun this script with --assign to do the assignment "
              f"programmatically.)")


if __name__ == "__main__":
    main()
