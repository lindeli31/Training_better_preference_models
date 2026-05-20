"""
run_b4_build_paraphrase_dataset.py
-----------------------------------
B4 preprocessing: build a paraphrased version of a stratified HelpSteer2 sample.

Samples n pairs evenly across all DIFFICULTY_LEVELS (tie/hard/medium/easy),
then paraphrases response_a and response_b for each pair using a single model.
All paraphrase calls are dispatched in parallel (up to --concurrency at a time).

The prompt includes per-response HelpSteer2 quality scores so the model knows
what level of helpfulness, correctness, coherence, complexity, and verbosity to preserve.

Output: data/helpsteer2_{split}_paraphrased.json
  Mirrors helpsteer2_{split}_full.json with two extra fields per record:
    response_a_para  — paraphrased version of response_a
    response_b_para  — paraphrased version of response_b

Usage
-----
    python run_b4_build_paraphrase_dataset.py
    python run_b4_build_paraphrase_dataset.py --n-pairs 200 --split validation --seed 42
    python run_b4_build_paraphrase_dataset.py --n-pairs 50  --concurrency 16
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer as _ST
    import numpy as _np
    _HAVE_ST = True
except ImportError:
    _HAVE_ST = False

try:
    from rouge_score import rouge_scorer as _rs
    _HAVE_ROUGE = True
except ImportError:
    _HAVE_ROUGE = False

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.dataset import DIFFICULTY_LEVELS, load_stratified_pairs as _load_stratified_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen3-32B-jmacina"
BASE_URL      = "https://api.swissai.cscs.ch/v1"

# First-attempt prompt: light word-level substitutions
PARAPHRASE_PROMPT = (
    "Make minimal edits to the text below: replace individual words and very short "
    "phrases with synonyms. Keep sentence structure and phrasing as close to the "
    "original as possible.\n\n"
    "The original response was annotated with the following quality scores (scale 0–4):\n"
    "  Helpfulness : {helpfulness}\n"
    "  Correctness : {correctness}\n"
    "  Coherence   : {coherence}\n"
    "  Complexity  : {complexity}\n"
    "  Verbosity   : {verbosity}\n\n"
    "Rules:\n"
    "1. Change at most 15–20% of words — word-level substitutions only, no sentence rewrites.\n"
    "2. Every fact, claim, and intent must remain exactly the same.\n"
    "3. Keep approximately the same length as the original.\n"
    "4. Do NOT add commentary, caveats, or new information.\n"
    "5. Output only the edited text, nothing else.\n\n"
    "Text: {text}\n\n"
    "Edited text:"
)

# Retry prompt: shown when quality check fails — includes both metrics and targeted feedback
PARAPHRASE_PROMPT_RETRY = (
    "Make edits to the text below to produce a paraphrase.\n\n"
    "The original response was annotated with the following quality scores (scale 0–4):\n"
    "  Helpfulness : {helpfulness}\n"
    "  Correctness : {correctness}\n"
    "  Coherence   : {coherence}\n"
    "  Complexity  : {complexity}\n"
    "  Verbosity   : {verbosity}\n\n"
    "Your previous attempt scored:\n"
    "  Semantic similarity (cosine) : {cos_sim:.3f}  [target: ≥ {min_cos_sim:.2f}]\n"
    "  Lexical overlap   (ROUGE-L)  : {rouge_l:.3f}  [target: ≤ {max_rouge_l:.2f}]\n\n"
    "{feedback}\n\n"
    "Rules:\n"
    "1. Every fact, claim, and intent must remain exactly the same.\n"
    "2. Keep approximately the same length as the original.\n"
    "3. Do NOT add commentary, caveats, or new information.\n"
    "4. Output only the edited text, nothing else.\n\n"
    "Text: {text}\n\n"
    "Edited text:"
)

_FEEDBACK_COPY = (
    "Problem: ROUGE-L {rouge_l:.3f} exceeds the target (≤ {max_rouge_l:.2f}) — "
    "your output is too close to a copy, you did not change enough.\n"
    "Fix: replace 15–20% of words with synonyms and lightly rephrase a few sentences "
    "while keeping all facts identical."
)
_FEEDBACK_DRIFT = (
    "Problem: cosine similarity {cos_sim:.3f} is below the target (≥ {min_cos_sim:.2f}) — "
    "you changed the meaning too much.\n"
    "Fix: use fewer than 10% word substitutions, keep sentence structure identical, "
    "only swap individual words for exact synonyms."
)
_FEEDBACK_BOTH = (
    "Problems: ROUGE-L {rouge_l:.3f} (target ≤ {max_rouge_l:.2f}) and cosine {cos_sim:.3f} "
    "(target ≥ {min_cos_sim:.2f}) are both off — you restructured sentences while also "
    "drifting in meaning.\n"
    "Fix: do word-level synonym swaps only (10–15% of words), keep every sentence structure "
    "and all facts identical."
)

def _pair_to_dict(p) -> dict:
    return {
        "prompt_id":  p.prompt_id,
        "prompt":     p.prompt,
        "response_a": p.response_a,
        "response_b": p.response_b,
        "gold_label": p.gold_label,
        "difficulty": p.difficulty,
        "score_gap":  p.score_gap,
        **p.extras,
    }


# ---------------------------------------------------------------------------
# Async paraphrase call
# ---------------------------------------------------------------------------

def _build_prompt(
    text: str,
    scores: dict,
    cos_sim: float | None = None,
    rouge_l: float | None = None,
    min_cos_sim: float = 0.90,
    max_rouge_l: float = 0.75,
) -> str:
    if cos_sim is None:
        return PARAPHRASE_PROMPT.format(
            text=text,
            helpfulness=scores.get("helpfulness", "N/A"),
            correctness=scores.get("correctness", "N/A"),
            coherence=scores.get("coherence",     "N/A"),
            complexity=scores.get("complexity",   "N/A"),
            verbosity=scores.get("verbosity",     "N/A"),
        )
    cs   = cos_sim
    rl   = rouge_l if rouge_l is not None else 0.0
    copy  = rl  > max_rouge_l
    drift = cs  < min_cos_sim
    if copy and drift:
        feedback = _FEEDBACK_BOTH.format(
            cos_sim=cs, min_cos_sim=min_cos_sim, rouge_l=rl, max_rouge_l=max_rouge_l)
    elif copy:
        feedback = _FEEDBACK_COPY.format(rouge_l=rl, max_rouge_l=max_rouge_l)
    else:
        feedback = _FEEDBACK_DRIFT.format(cos_sim=cs, min_cos_sim=min_cos_sim)
    return PARAPHRASE_PROMPT_RETRY.format(
        text=text,
        helpfulness=scores.get("helpfulness", "N/A"),
        correctness=scores.get("correctness", "N/A"),
        coherence=scores.get("coherence",     "N/A"),
        complexity=scores.get("complexity",   "N/A"),
        verbosity=scores.get("verbosity",     "N/A"),
        cos_sim=cs, min_cos_sim=min_cos_sim,
        rouge_l=rl, max_rouge_l=max_rouge_l,
        feedback=feedback,
    )


def _cosine(a, b) -> float:
    na, nb = _np.linalg.norm(a), _np.linalg.norm(b)
    return float(_np.dot(a, b) / (na * nb + 1e-9))


def _rouge_l(orig: str, para: str) -> float:
    if not _HAVE_ROUGE:
        return 0.0
    scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(orig, para)["rougeL"].fmeasure


def _quality_ok(cos: float, rl: float, min_cos: float, max_rouge: float) -> bool:
    return cos >= min_cos and rl <= max_rouge


async def _api_call(
    session: aiohttp.ClientSession,
    prompt: str,
    api_key: str,
    model: str,
    max_retries: int,
    base_delay: float,
) -> str:
    """Single API call with exponential-backoff retry on 429 / exceptions."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "/no_think"},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            async with session.post(
                f"{BASE_URL}/chat/completions", json=payload, headers=headers
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Strip Qwen3 thinking blocks if the API returns them anyway
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content
        except Exception as exc:
            if attempt == max_retries - 1:
                logger.error("Failed after %d retries: %s", max_retries, exc)
                return ""
            await asyncio.sleep(base_delay * (2 ** attempt))
    return ""


async def paraphrase(
    session: aiohttp.ClientSession,
    text: str,
    scores: dict,
    api_key: str,
    model: str,
    semaphore: asyncio.Semaphore,
    encoder=None,
    min_cos_sim: float = 0.90,
    max_rouge_l: float = 0.75,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_quality_retries: int = 3,
) -> str:
    loop = asyncio.get_event_loop()

    async with semaphore:
        result = await _api_call(
            session, _build_prompt(text, scores),
            api_key, model, max_retries, base_delay,
        )
        if not result:
            return ""

        if (encoder is None or not _HAVE_ST) and not _HAVE_ROUGE:
            return result

        def _score(candidate: str) -> tuple[float, float]:
            cos = 0.0
            if encoder is not None and _HAVE_ST:
                embs = encoder.encode([text, candidate])
                cos = _cosine(embs[0], embs[1])
            rl = _rouge_l(text, candidate)
            return cos, rl

        best_result = result
        best_cos, best_rl = await loop.run_in_executor(None, lambda: _score(result))
        logger.debug("Initial: cos=%.3f rouge_l=%.3f", best_cos, best_rl)

        for attempt in range(max_quality_retries):
            if _quality_ok(best_cos, best_rl, min_cos_sim, max_rouge_l):
                break
            logger.debug(
                "Quality retry %d/%d: cos=%.3f rouge_l=%.3f",
                attempt + 1, max_quality_retries, best_cos, best_rl,
            )
            candidate = await _api_call(
                session, _build_prompt(
                    text, scores,
                    cos_sim=best_cos, rouge_l=best_rl,
                    min_cos_sim=min_cos_sim, max_rouge_l=max_rouge_l,
                ),
                api_key, model, max_retries, base_delay,
            )
            if not candidate:
                continue
            cos2, rl2 = await loop.run_in_executor(None, lambda: _score(candidate))
            logger.debug("  → cos=%.3f rouge_l=%.3f", cos2, rl2)
            # Accept if strictly better on the primary failing dimension
            if not _quality_ok(best_cos, best_rl, min_cos_sim, max_rouge_l):
                copy_fail  = best_rl  > max_rouge_l
                drift_fail = best_cos < min_cos_sim
                improved = (copy_fail and rl2 < best_rl) or (drift_fail and cos2 > best_cos)
                if improved:
                    best_result, best_cos, best_rl = candidate, cos2, rl2

        if not _quality_ok(best_cos, best_rl, min_cos_sim, max_rouge_l):
            logger.warning(
                "Quality still off after %d retries: cos=%.3f rouge_l=%.3f",
                max_quality_retries, best_cos, best_rl,
            )

        return best_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args):
    api_key = os.environ.get("SWISSAI_API_KEY", "").strip().strip('"')
    if not api_key:
        raise ValueError("SWISSAI_API_KEY not set")

    data_path = Path("data") / f"helpsteer2_{args.split}_full.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found — run: python -m src.datasets.dataset"
        )

    raw_pairs = _load_stratified_pairs(
        split=args.split, n=args.n_pairs,
        seed=args.seed, difficulties=("tie", "hard", "medium", "easy"),
    )
    pairs = [_pair_to_dict(p) for p in raw_pairs]
    dist = {d: sum(1 for p in pairs if p.get("difficulty") == d) for d in DIFFICULTY_LEVELS}
    logger.info("Stratified sample: %d pairs | %s", len(pairs), dist)
    logger.info("Paraphrasing %d pairs with %s (concurrency=%d)",
                len(pairs), args.model, args.concurrency)

    # Load sentence encoder for quality-check retries (optional)
    encoder = None
    if _HAVE_ST and not args.no_quality_check:
        logger.info("Loading sentence encoder for cosine similarity quality check...")
        encoder = _ST("all-MiniLM-L6-v2")
        logger.info("Quality check enabled: cos ≥ %.2f, rouge_l ≤ %.2f",
                    args.min_cos_sim, args.max_rouge_l)
    elif not _HAVE_ST and not args.no_quality_check:
        logger.warning("sentence-transformers not installed — quality check disabled. "
                       "Run: pip install sentence-transformers")

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    total_calls = len(pairs) * 2
    done_count = 0

    async def paraphrase_tracked(*a, **kw):
        nonlocal done_count
        result = await paraphrase(*a, **kw)
        done_count += 1
        if done_count % 50 == 0 or done_count == total_calls:
            elapsed = time.perf_counter() - t0
            rate = done_count / elapsed
            eta = (total_calls - done_count) / rate if rate > 0 else 0
            logger.info("Progress: %d/%d responses (%.0f/s, ETA %.0fmin)",
                        done_count, total_calls, rate, eta / 60)
        return result

    async with aiohttp.ClientSession() as session:
        tasks_a = [
            paraphrase_tracked(
                session, p["response_a"],
                {dim: p[f"{dim}_a"] for dim in ("helpfulness", "correctness", "coherence", "complexity", "verbosity")},
                api_key, args.model, semaphore,
                encoder=encoder, min_cos_sim=args.min_cos_sim,
                max_rouge_l=args.max_rouge_l,
                max_quality_retries=args.max_quality_retries,
            )
            for p in pairs
        ]
        tasks_b = [
            paraphrase_tracked(
                session, p["response_b"],
                {dim: p[f"{dim}_b"] for dim in ("helpfulness", "correctness", "coherence", "complexity", "verbosity")},
                api_key, args.model, semaphore,
                encoder=encoder, min_cos_sim=args.min_cos_sim,
                max_rouge_l=args.max_rouge_l,
                max_quality_retries=args.max_quality_retries,
            )
            for p in pairs
        ]
        paras_a, paras_b = await asyncio.gather(
            asyncio.gather(*tasks_a),
            asyncio.gather(*tasks_b),
        )

    results = [
        {**pair, "response_a_para": pa, "response_b_para": pb, "para_model": args.model}
        for pair, pa, pb in zip(pairs, paras_a, paras_b)
    ]

    output_path = Path("data") / f"helpsteer2_{args.split}_paraphrased.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    failed = sum(1 for r in results if not r["response_a_para"] or not r["response_b_para"])
    logger.info("Done in %.1fs — saved %d pairs to %s (%d failed)",
                time.perf_counter() - t0, len(results), output_path, failed)

    if args.preview:
        _print_preview(results, encoder)


def _print_preview(results: list[dict], encoder) -> None:
    """Print a side-by-side comparison table: original vs paraphrase + quality scores."""
    import textwrap

    scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True) if _HAVE_ROUGE else None

    W = 52  # column width for text
    sep = "─" * (W * 2 + 35)

    header = (f"{'':>3}  {'DIFF':<6}  {'RESP':<4}  "
              f"{'ORIGINAL':<{W}}  {'PARAPHRASE':<{W}}  {'COS':>5}  {'ROU':>5}")
    print("\n" + sep)
    print(header)
    print(sep)

    for i, r in enumerate(results):
        for resp_key in ("a", "b"):
            orig = r.get(f"response_{resp_key}", "")
            para = r.get(f"response_{resp_key}_para", "") or "(failed)"

            cos_str = rou_str = "  n/a"
            if encoder is not None and _HAVE_ST and para != "(failed)":
                embs = encoder.encode([orig, para])
                cos = _cosine(embs[0], embs[1])
                cos_str = f"{cos:.3f}"
            if scorer and para != "(failed)":
                rl = scorer.score(orig, para)["rougeL"].fmeasure
                rou_str = f"{rl:.3f}"

            orig_lines = textwrap.wrap(orig, W) or [""]
            para_lines = textwrap.wrap(para, W) or [""]
            n_lines = max(len(orig_lines), len(para_lines))
            orig_lines += [""] * (n_lines - len(orig_lines))
            para_lines += [""] * (n_lines - len(para_lines))

            for j, (ol, pl) in enumerate(zip(orig_lines, para_lines)):
                if j == 0:
                    print(f"{i+1:>3}  {r.get('difficulty','?'):<6}  resp_{resp_key}  "
                          f"{ol:<{W}}  {pl:<{W}}  {cos_str:>5}  {rou_str:>5}")
                else:
                    print(f"{'':>3}  {'':6}  {'':4}  {ol:<{W}}  {pl:<{W}}")
        print(sep)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs",          type=int,   default=200)
    p.add_argument("--split",                        default="validation")
    p.add_argument("--concurrency",      type=int,   default=16)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--model",                        default=DEFAULT_MODEL)
    p.add_argument("--min-cos-sim",      type=float, default=0.90,
                   help="Minimum cosine similarity (requires sentence-transformers)")
    p.add_argument("--max-rouge-l",      type=float, default=0.85,
                   help="Maximum ROUGE-L to reject near-copies (requires rouge-score)")
    p.add_argument("--max-quality-retries", type=int, default=5,
                   help="Max quality-check retries per response (default: 3)")
    p.add_argument("--no-quality-check", action="store_true",
                   help="Disable cosine similarity quality check and retry")
    p.add_argument("--preview",          action="store_true",
                   help="Print side-by-side comparison table after generation")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
