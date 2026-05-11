"""
opro_tree_position_bias.py
--------------------------
OPRO variant with best-of-N branching at each iteration (tree search,
beam = 1, width = branching_factor).

Differences vs opro_position_bias.py:
  - At each iteration, ask the meta-LM for `branching_factor` candidates
    (diversity comes from temperature sampling).
  - Evaluate ALL candidates on the eval subset with the same scalar OPRO
    score (w_c*consistency + w_a*accuracy - length_hinge).
  - Keep only the best-scoring candidate → becomes the new "current best"
    shown to the meta-LM in the next iteration.

Judge inference is UNCHANGED: single-letter A/B/C output. Only the
proposer strategy differs.

Cost (vs classic OPRO):
  - ~`branching_factor`× more subset evals and meta-LM calls for the same
    n_iterations. The final train/val re-eval is unchanged (1 each).

Pipeline:
  1. Baseline = SEED_PROMPTS[0] on full train + val (matches OPRO).
  2. Evaluate both SEED_PROMPTS on subset; pick the higher-score one as
     the starting "current best".
  3. For N iterations:
       a. Generate `branching_factor` candidates in parallel.
       b. Evaluate each on the subset.
       c. Promote the best-scoring candidate to current.
  4. Re-evaluate the final best prompt on full train + val.
  5. Save results + full branch history.
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from src.dataset.dataset import PairRecord
from src.core.inference_client import SwissAIClient
from src.runall_prompt_optimization.opro_position_bias import (
    META_SYSTEM,
    _build_meta_prompt,
    evaluate_full_metrics,
)
from src.runall_prompt_optimization.scoring import compute_score, SEED_PROMPTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate generation: N parallel meta-LM calls sharing the same history.
# ---------------------------------------------------------------------------

async def _generate_one(client: SwissAIClient, history: list[dict], tag: str) -> str:
    response = await client.judge(
        system_prompt=META_SYSTEM,
        user_prompt=_build_meta_prompt(history),
        prompt_id="meta",
        experiment_id=tag,
        condition="generate",
    )
    return response.raw_text.strip()


async def generate_branch(
    client: SwissAIClient,
    history: list[dict],
    branching_factor: int,
    iter_tag: str,
) -> list[str]:
    """Fire `branching_factor` meta-LM calls in parallel, return all raw prompts."""
    tasks = [
        _generate_one(client, history, f"{iter_tag}_branch_{i}")
        for i in range(branching_factor)
    ]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OproTreeResult:
    best_prompt: str
    best_train_score: float     # scalar OPRO score on the eval subset
    baseline_train: dict
    baseline_val: dict
    train: dict
    val: dict
    history: list[dict]         # per-iteration: winner + all branches
    branching_factor: int


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_opro_tree(
    client: SwissAIClient,
    train_pairs: list[PairRecord],
    val_pairs: list[PairRecord],
    n_iterations: int = 10,
    eval_pairs: int = 30,
    branching_factor: int = 10,
    seed: int = 42,
    output_dir: Path = Path("results/opro_tree"),
    proposer_client: SwissAIClient | None = None,
    precomputed_baseline_train: dict | None = None,
    precomputed_baseline_val: dict | None = None,
) -> OproTreeResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    eval_subset = rng.sample(train_pairs, min(eval_pairs, len(train_pairs)))

    # -------------------------------------------------------------------
    # Baseline on full train + val.
    # -------------------------------------------------------------------
    if precomputed_baseline_train is not None and precomputed_baseline_val is not None:
        baseline_train = precomputed_baseline_train
        baseline_val   = precomputed_baseline_val
        logger.info(
            "Using pre-computed shared baseline: "
            "train cons=%.3f acc=%.3f | val cons=%.3f acc=%.3f",
            baseline_train["position_consistency"], baseline_train["accuracy"],
            baseline_val["position_consistency"],   baseline_val["accuracy"],
        )
    else:
        baseline_prompt = SEED_PROMPTS[0]
        logger.info("Evaluating OPRO-tree baseline on %d train pairs", len(train_pairs))
        baseline_train = await evaluate_full_metrics(
            client, train_pairs, baseline_prompt, experiment_id="opro_tree_baseline_train"
        )
        logger.info(
            "Baseline (train) consistency=%.3f accuracy=%.3f bias_rate=%.3f",
            baseline_train["position_consistency"],
            baseline_train["accuracy"],
            baseline_train["position_bias_rate"],
        )
        baseline_val = await evaluate_full_metrics(
            client, val_pairs, baseline_prompt, experiment_id="opro_tree_baseline_val"
        )
        logger.info(
            "Baseline (val)   consistency=%.3f accuracy=%.3f bias_rate=%.3f",
            baseline_val["position_consistency"],
            baseline_val["accuracy"],
            baseline_val["position_bias_rate"],
        )

    # -------------------------------------------------------------------
    # Seed selection: eval both seeds on subset, keep best as current.
    # -------------------------------------------------------------------
    seed_entries = []
    for i, sp in enumerate(SEED_PROMPTS):
        m = await evaluate_full_metrics(
            client, eval_subset, sp, experiment_id=f"opro_tree_seed_{i}"
        )
        score = compute_score(sp, m)
        seed_entries.append({"prompt": sp, "score": score, "metrics": m})
        logger.info(
            "Seed #%d subset consistency=%.3f accuracy=%.3f score=%.3f",
            i, m["position_consistency"], m["accuracy"], score,
        )
    best_seed = max(seed_entries, key=lambda x: x["score"])
    logger.info("Starting tree search from seed with score=%.3f", best_seed["score"])
    history: list[dict] = [{**best_seed, "branches": seed_entries}]

    meta_client = proposer_client or client

    # -------------------------------------------------------------------
    # Main loop: N parallel candidates per iteration, keep the best.
    # -------------------------------------------------------------------
    for it in range(n_iterations):
        logger.info(
            "Iteration %d/%d — generating %d candidates",
            it + 1, n_iterations, branching_factor,
        )
        candidates = await generate_branch(
            meta_client, history, branching_factor, f"opro_tree_iter_{it + 1}"
        )

        # Evaluate branches in parallel. Each evaluate_full_metrics fans out
        # to eval_pairs*2 calls internally; the client's semaphore gates
        # overall concurrency so this is safe.
        eval_tasks = [
            evaluate_full_metrics(
                client, eval_subset, cand,
                experiment_id=f"opro_tree_iter_{it + 1}_branch_{bi}",
            )
            for bi, cand in enumerate(candidates)
        ]
        branch_metrics = await asyncio.gather(*eval_tasks)

        branch_entries = []
        for bi, (cand, m) in enumerate(zip(candidates, branch_metrics)):
            score = compute_score(cand, m)
            branch_entries.append({"prompt": cand, "score": score, "metrics": m})
            logger.info(
                "  branch %d/%d  consistency=%.3f accuracy=%.3f score=%.3f",
                bi + 1, branching_factor,
                m["position_consistency"], m["accuracy"], score,
            )

        best_branch = max(branch_entries, key=lambda x: x["score"])
        logger.info(
            "Iter %d best branch: score=%.3f (consistency=%.3f accuracy=%.3f)",
            it + 1, best_branch["score"],
            best_branch["metrics"]["position_consistency"],
            best_branch["metrics"]["accuracy"],
        )
        history.append({
            "prompt": best_branch["prompt"],
            "score": best_branch["score"],
            "metrics": best_branch["metrics"],
            "branches": branch_entries,
        })

    # -------------------------------------------------------------------
    # Re-evaluate global best on full train + val.
    # -------------------------------------------------------------------
    best_entry = max(history, key=lambda x: x["score"])
    logger.info("Best over full tree: score=%.3f", best_entry["score"])

    train = await evaluate_full_metrics(
        client, train_pairs, best_entry["prompt"], experiment_id="opro_tree_best_train"
    )
    val = await evaluate_full_metrics(
        client, val_pairs, best_entry["prompt"], experiment_id="opro_tree_best_val"
    )
    logger.info(
        "Best (train) consistency=%.3f accuracy=%.3f bias_rate=%.3f",
        train["position_consistency"], train["accuracy"], train["position_bias_rate"],
    )
    logger.info(
        "Best (val)   consistency=%.3f accuracy=%.3f bias_rate=%.3f",
        val["position_consistency"], val["accuracy"], val["position_bias_rate"],
    )

    result = OproTreeResult(
        best_prompt=best_entry["prompt"],
        best_train_score=best_entry["score"],
        baseline_train=baseline_train,
        baseline_val=baseline_val,
        train=train,
        val=val,
        history=history,
        branching_factor=branching_factor,
    )

    with open(output_dir / "opro_tree_results.json", "w") as f:
        json.dump({
            "best_prompt":       result.best_prompt,
            "best_train_score":  result.best_train_score,
            "baseline_train":    result.baseline_train,
            "baseline_val":      result.baseline_val,
            "train":             result.train,
            "val":               result.val,
            "history":           result.history,
            "branching_factor":  result.branching_factor,
            "n_iterations":      n_iterations,
        }, f, indent=2)
    with open(output_dir / "optimised_prompt.md", "w", encoding="utf-8") as f:
        f.write(result.best_prompt)

    return result
