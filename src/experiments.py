"""
experiments.py
--------------
High-level experiment runners.  Each experiment is a Python async function
that:
  1. Builds a list of judge calls
  2. Dispatches them via SwissAIClient.batch_judge
  3. Saves results to JSONL

Experiments
-----------
  run_position_bias      (B1) - flip A/B, measure consistency
  run_template_sensitivity (B2) - vary system-prompt wording
  run_reasoning_depth    (B3) - compare no reasoning vs. prompted reasoning styles
  run_input_sensitivity  (B4) - minor paraphrasing of templates
  run_repetition_stability (B5) - same prompt N times, measure label distribution
  run_evaluate_accuracy  (Acc) - AB-only, enriched with gold_label and score_gap

All experiments share the same data (list[PairRecord]) and client.
"""

# Standard library imports
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional
from src.inference_client import SwissAIClient, InferenceConfig, JudgeResponse
from src.templates import build_prompt, TEMPLATES, CRITERIA
from src.dataset import PairRecord
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Helper: function to save results to JSONL 
# ---------------------------------------------------------------------------
def save_jsonl(results: list[JudgeResponse], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    logger.info("Saved %d results → %s", len(results), path)
# ---------------------------------------------------------------------------
# Experiment B1: Position Bias
# For each pair, the model judges twice:
#   - Once in the original order (A first, B second) → condition "AB"
#   - Once in the flipped order (B first, A second) → condition "BA
# Finally, it saves the results for analysis (see metrics.py)
# ---------------------------------------------------------------------------
async def run_position_bias(
    client: SwissAIClient,
    pairs: list[PairRecord],
    template_id: str = "expert_rater",
    criterion: str = "overall",
    output_dir: Path = Path("results/position_bias"),
) -> list[JudgeResponse]:
    """
    Experiment B1: Position Bias.

    For each pair, the model judges twice:
      - Once in the original order (A first, B second) → condition "AB"
      - Once in the flipped order (B first, A second) → condition "BA"

    If the model is consistent, flipping the order should flip the label.
    """

    # Step 1: Build all judge calls (2 per pair: original + flipped)
    all_judge_calls = []

    for pair in pairs:

        ## Original order: response A first, response B second
        # build the system and user prompts using the specified template and criterion
        original_system_prompt, original_user_prompt = build_prompt(
            template_id, pair.prompt, pair.response_a, pair.response_b, criterion
        )
        # add the judge call to the list of all judge calls
        all_judge_calls.append(dict(
            system_prompt=original_system_prompt,
            user_prompt=original_user_prompt,
            prompt_id=pair.prompt_id,
            experiment_id="position_bias",
            condition="AB",
        ))

        ## Flipped order: response B first, response A second
        flipped_pair = pair.flipped()
        flipped_system_prompt, flipped_user_prompt = build_prompt(
            template_id, flipped_pair.prompt,
            flipped_pair.response_a, flipped_pair.response_b, criterion
        )
        all_judge_calls.append(dict(
            system_prompt=flipped_system_prompt,
            user_prompt=flipped_user_prompt,
            prompt_id=pair.prompt_id,
            experiment_id="position_bias",
            condition="BA",
        ))

    # Step 2: Send all calls to the model in parallel
    logger.info("[B1] Dispatching %d calls (position bias, template=%s)", len(all_judge_calls), template_id)
    results = await client.batch_judge(all_judge_calls)

    # Step 3: Save results to JSONL
    save_jsonl(results, output_dir / f"{template_id}_{criterion}.jsonl")
    return results












## TODO
# ---------------------------------------------------------------------------
# Experiment B2: Template Sensitivity TODO
# ---------------------------------------------------------------------------
TEMPLATE_SENSITIVITY_VARIANTS = [
    "expert_rater",
    "llm_judge",
    "neutral",
    "academic",
    "minimal",
]
async def run_template_sensitivity(
    client: SwissAIClient,
    pairs: list[PairRecord],
    template_ids: Optional[list[str]] = None,
    criterion: str = "overall",
    output_dir: Path = Path("results/template_sensitivity"),
) -> list[JudgeResponse]:
    """
    Judge each pair with multiple system-prompt templates (same order, same data).
    Consistency across templates reveals template sensitivity.
    """
    template_ids = template_ids or TEMPLATE_SENSITIVITY_VARIANTS

    calls = []
    for tmpl in template_ids:
        for pair in pairs:
            sys_p, usr_p = build_prompt(tmpl, pair.prompt,
                                         pair.response_a, pair.response_b, criterion)
            calls.append(dict(
                system_prompt=sys_p, user_prompt=usr_p,
                prompt_id=pair.prompt_id, experiment_id="template_sensitivity",
                condition=tmpl,
            ))

    logger.info("[B2] Dispatching %d calls (%d templates × %d pairs)",
                len(calls), len(template_ids), len(pairs))
    results = await client.batch_judge(calls)
    save_jsonl(results, output_dir / f"criterion_{criterion}.jsonl")
    return results
# ---------------------------------------------------------------------------
# Experiment B3: Reasoning Depth TODO
# ---------------------------------------------------------------------------
REASONING_CONDITIONS = [
    {"label": "no_reasoning",          "template": "expert_rater"},
    {"label": "reason_then_judge",     "template": "reason_then_judge"},
    {"label": "structured_reasoning",  "template": "structured_reasoning"},
]
async def run_reasoning_depth(
    client: SwissAIClient,
    pairs: list[PairRecord],
    conditions: Optional[list[dict]] = None,
    criterion: str = "overall",
    output_dir: Path = Path("results/reasoning_depth"),
) -> list[JudgeResponse]:
    """
    Compare three prompt-elicited reasoning conditions:
      - no_reasoning          : just output A, B, or C
      - reason_then_judge     : explain reasoning, then give verdict
      - structured_reasoning  : rate on specific criteria, then give verdict

    Agreement with gold label tells us how much reasoning helps.
    """
    conditions = conditions or REASONING_CONDITIONS
    calls = []
    for cond in conditions:
        for pair in pairs:
            sys_p, usr_p = build_prompt(cond["template"], pair.prompt,
                                         pair.response_a, pair.response_b, criterion)
            calls.append(dict(
                system_prompt=sys_p, user_prompt=usr_p,
                prompt_id=pair.prompt_id, experiment_id="reasoning_depth",
                condition=cond["label"],
            ))

    logger.info("[B3] Dispatching %d calls (%d conditions × %d pairs)",
                len(calls), len(conditions), len(pairs))
    results = await client.batch_judge(calls)
    save_jsonl(results, output_dir / f"criterion_{criterion}.jsonl")
    return results
# ---------------------------------------------------------------------------
# Experiment B4: Input / Wording Sensitivity TODO
# ---------------------------------------------------------------------------
INPUT_SENSITIVITY_VARIANTS = [
    "expert_rater",
    "expert_rater_alt1",
    "expert_rater_alt2",
    "expert_rater_alt3",
]
async def run_input_sensitivity(
    client: SwissAIClient,
    pairs: list[PairRecord],
    template_ids: Optional[list[str]] = None,
    criteria: Optional[list[str]] = None,
    output_dir: Path = Path("results/input_sensitivity"),
) -> list[JudgeResponse]:
    """
    Minor paraphrasing of the same expert-rater template.
    Also sweeps across multiple criteria to check criterion-label drift.
    """
    template_ids = template_ids or INPUT_SENSITIVITY_VARIANTS
    criteria     = criteria     or list(CRITERIA.keys())

    calls = []
    for tmpl in template_ids:
        for crit in criteria:
            for pair in pairs:
                sys_p, usr_p = build_prompt(tmpl, pair.prompt,
                                             pair.response_a, pair.response_b, crit)
                calls.append(dict(
                    system_prompt=sys_p, user_prompt=usr_p,
                    prompt_id=pair.prompt_id,
                    experiment_id="input_sensitivity",
                    condition=f"{tmpl}_{crit}",
                ))

    logger.info("[B4] Dispatching %d calls (%d templates × %d criteria × %d pairs)",
                len(calls), len(template_ids), len(criteria), len(pairs))
    results = await client.batch_judge(calls)
    save_jsonl(results, output_dir / "all.jsonl")
    return results


# ---------------------------------------------------------------------------
# Experiment B5: Repetition Stability
# For each pair, send the SAME AB-order judge call n_repetitions times.
# Each repetition is tagged with condition=f"rep_{i}" so results remain
# distinguishable in the JSONL without changing the JudgeResponse schema.
# Temperature is controlled via the client's InferenceConfig.
# ---------------------------------------------------------------------------
async def run_repetition_stability(
    client: SwissAIClient,
    pairs: list[PairRecord],
    n_repetitions: int = 30,
    template_id: str = "expert_rater",
    criterion: str = "overall",
    output_dir: Path = Path("results/repetition_stability"),
    run_name: Optional[str] = None,
    meta: Optional[dict] = None,
) -> list[JudgeResponse]:
    """
    Experiment B5: Repetition Stability.

    For each pair, submit the same (AB-order) judge call `n_repetitions` times.
    With temperature=0 this measures residual server non-determinism; with
    temperature>0 it measures sampling variance of the judge.

    If `run_name` is given, it is appended to the output filename so multiple
    runs at the same temperature don't overwrite each other. If `meta` is given,
    it is saved as a sidecar `<basename>.meta.json` alongside the JSONL.
    """
    calls = []
    for pair in pairs:
        sys_p, usr_p = build_prompt(
            template_id, pair.prompt, pair.response_a, pair.response_b, criterion
        )
        for i in range(n_repetitions):
            calls.append(dict(
                system_prompt=sys_p,
                user_prompt=usr_p,
                prompt_id=pair.prompt_id,
                experiment_id="repetition_stability",
                condition=f"rep_{i}",
            ))

    logger.info("[B5] Dispatching %d calls (%d pairs × %d repetitions, temperature=%.2f)",
                len(calls), len(pairs), n_repetitions, client.config.temperature)
    results = await client.batch_judge(calls)

    suffix = f"_{run_name}" if run_name else ""
    base_name = f"{template_id}_{criterion}_t{client.config.temperature}{suffix}"
    save_jsonl(results, output_dir / f"{base_name}.jsonl")
    if meta is not None:
        meta_path = output_dir / f"{base_name}.meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        logger.info("Saved metadata → %s", meta_path)
    return results


# ---------------------------------------------------------------------------
# Experiment: Evaluate Accuracy (AB-only, enriched output)
# One judge call per pair in AB order. Each saved record is a JudgeResponse
# dict enriched with: gold_label, score_gap, score_a, score_b, is_correct.
# This lets downstream analysis bucket errors by score_gap ("difficulty").
# ---------------------------------------------------------------------------
async def run_evaluate_accuracy(
    client: SwissAIClient,
    pairs: list[PairRecord],
    template_id: str = "expert_rater",
    criterion: str = "overall",
    output_dir: Path = Path("results/evaluate_accuracy"),
    run_name: Optional[str] = None,
    meta: Optional[dict] = None,
) -> list[dict]:
    """
    One AB-order judge call per pair. Returns enriched dicts (not JudgeResponse)
    carrying gold_label, score_gap, score_a, score_b, is_correct alongside the
    standard JudgeResponse fields.
    """
    calls = []
    for pair in pairs:
        sys_p, usr_p = build_prompt(
            template_id, pair.prompt, pair.response_a, pair.response_b, criterion
        )
        calls.append(dict(
            system_prompt=sys_p,
            user_prompt=usr_p,
            prompt_id=pair.prompt_id,
            experiment_id="evaluate_accuracy",
            condition="AB",
        ))

    logger.info("[Acc] Dispatching %d calls (accuracy evaluation, template=%s, criterion=%s)",
                len(calls), template_id, criterion)
    results = await client.batch_judge(calls)

    enriched: list[dict] = []
    for pair, resp in zip(pairs, results):
        d = resp.to_dict()
        d["gold_label"] = pair.gold_label
        d["difficulty"] = pair.difficulty
        d["score_gap"] = pair.extras.get("score_gap")
        d["score_a"] = pair.extras.get("score_a")
        d["score_b"] = pair.extras.get("score_b")
        d["is_correct"] = (d["label"] == pair.gold_label) if d["label"] is not None else None
        enriched.append(d)

    suffix = f"_{run_name}" if run_name else ""
    base_name = f"{template_id}_{criterion}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{base_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in enriched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d enriched results → %s", len(enriched), jsonl_path)

    if meta is not None:
        meta_path = output_dir / f"{base_name}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        logger.info("Saved metadata → %s", meta_path)

    return enriched
