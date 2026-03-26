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

All experiments share the same data (list[PairRecord]) and client.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from inference_client import SwissAIClient, InferenceConfig, JudgeResponse
from templates import build_prompt, TEMPLATES, CRITERIA
from dataset import PairRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: save results
# ---------------------------------------------------------------------------

def save_jsonl(results: list[JudgeResponse], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    logger.info("Saved %d results → %s", len(results), path)


# ---------------------------------------------------------------------------
# Experiment B1: Position Bias
# ---------------------------------------------------------------------------

async def run_position_bias(
    client: SwissAIClient,
    pairs: list[PairRecord],
    template_id: str = "expert_rater",
    criterion: str = "helpful",
    output_dir: Path = Path("results/position_bias"),
) -> list[JudgeResponse]:
    """
    For each pair, judge both orders:
      - condition "AB": A=chosen, B=rejected
      - condition "BA": A=rejected, B=chosen  (flipped)

    Consistency = fraction where the model prefers the same response
                  regardless of order (accounting for label flip).
    """
    calls = []
    for pair in pairs:
        # Original order
        sys_p, usr_p = build_prompt(template_id, pair.prompt,
                                     pair.response_a, pair.response_b, criterion)
        calls.append(dict(
            system_prompt=sys_p, user_prompt=usr_p,
            prompt_id=pair.prompt_id, experiment_id="position_bias",
            condition="AB",
        ))

        # Flipped order
        flipped = pair.flipped()
        sys_p2, usr_p2 = build_prompt(template_id, flipped.prompt,
                                       flipped.response_a, flipped.response_b, criterion)
        calls.append(dict(
            system_prompt=sys_p2, user_prompt=usr_p2,
            prompt_id=pair.prompt_id, experiment_id="position_bias",
            condition="BA",
        ))

    logger.info("[B1] Dispatching %d calls (position bias, template=%s)", len(calls), template_id)
    results = await client.batch_judge(calls)
    save_jsonl(results, output_dir / f"{template_id}_{criterion}.jsonl")
    return results


# ---------------------------------------------------------------------------
# Experiment B2: Template Sensitivity
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
    criterion: str = "helpful",
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
# Experiment B3: Reasoning Depth
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
    criterion: str = "helpful",
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
# Experiment B4: Input / Wording Sensitivity
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
