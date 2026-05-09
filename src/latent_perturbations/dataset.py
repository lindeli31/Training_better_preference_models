"""
dataset.py
----------
Build swap-pair datasets for the gradient-based positional bias experiments.

For each HelpSteer2 pair we produce two formatted judge prompts:
  prompt_ab  — resp_a in slot_1, resp_b in slot_2
  prompt_ba  — resp_b in slot_1, resp_a in slot_2

Bias labeling (slot vs response distinction):
  verdict labels refer to SLOTS ("A"=slot_1, "B"=slot_2), not responses.
  Same verdict on both orderings → model locked onto a slot regardless of content → BIASED.
  Different verdict on both orderings → model's preferred response stayed the same → UNBIASED.

  verdict_AB="A", verdict_BA="B" → resp_a won both times          → unbiased
  verdict_AB="B", verdict_BA="A" → resp_b won both times          → unbiased
  verdict_AB="A", verdict_BA="A" → slot_1 won both times          → biased (prefers first)
  verdict_AB="B", verdict_BA="B" → slot_2 won both times          → biased (prefers second)

filter_biased=True loads a larger pool, runs verdicts on both orderings,
and keeps only the biased pairs — giving the experiment maximum signal.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from src.datasets.dataset import load_dataset_pairs
from src.core.templates import build_prompt


@dataclass
class SwapPair:
    pair_id: int
    prompt_ab: str            # judge prompt: resp_a in slot_1, resp_b in slot_2
    prompt_ba: str            # judge prompt: resp_b in slot_1, resp_a in slot_2
    second_token_ab: str = "B"   # slot_2 token for (A,B) ordering
    second_token_ba: str = "A"   # slot_2 token for (B,A) ordering
    question: str = ""
    response_a: str = ""
    response_b: str = ""
    human_pref: Optional[str] = None
    # Filled when filter_biased=True
    verdict_ab: Optional[str] = None
    verdict_ba: Optional[str] = None
    biased: Optional[bool] = None   # True = same slot chosen both times


def _chat_format(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _build_swap_pair(i: int, pair, tokenizer, template_id: str, criterion: str) -> SwapPair:
    sys_ab, usr_ab = build_prompt(
        template_id, pair.prompt, pair.response_a, pair.response_b, criterion
    )
    sys_ba, usr_ba = build_prompt(
        template_id, pair.prompt, pair.response_b, pair.response_a, criterion
    )
    return SwapPair(
        pair_id=i,
        prompt_ab=_chat_format(tokenizer, sys_ab, usr_ab),
        prompt_ba=_chat_format(tokenizer, sys_ba, usr_ba),
        question=pair.prompt,
        response_a=pair.response_a,
        response_b=pair.response_b,
        human_pref=getattr(pair, "label", None),
    )


def load_swap_pairs(
    tokenizer,
    n: int = 200,
    split: str = "validation",
    seed: int = 42,
    template_id: str = "expert_rater",
    criterion: str = "overall",
    filter_biased: bool = False,
    model=None,
    verdict_tokens: Optional[list[str]] = None,
    pool_multiplier: int = 4,
    bias_cache_path: Optional[str] = None,
    bias_ratio: float = 0.5,
) -> list[SwapPair]:
    """
    Load HelpSteer2 pairs and build both prompt orderings for each.

    filter_biased=True:
      Loads n * pool_multiplier pairs, runs the judge on both orderings,
      keeps only positionally biased pairs (same slot chosen regardless of order),
      then returns the first n. Requires model to be passed.

      Bias labels are cached to bias_cache_path (JSONL) so subsequent runs
      with the same pool skip the verdict-running step.
    """
    if filter_biased and model is None:
        raise ValueError("filter_biased=True requires model to be passed")

    if verdict_tokens is None:
        verdict_tokens = ["A", "B", "C"]

    if filter_biased:
        return _load_mixed_pairs(
            tokenizer, model, n, split, seed, template_id, criterion,
            verdict_tokens, pool_multiplier, bias_cache_path, bias_ratio,
        )

    raw_pairs = load_dataset_pairs(split=split, n=n, seed=seed)
    return [_build_swap_pair(i, p, tokenizer, template_id, criterion)
            for i, p in enumerate(raw_pairs)]


def _load_mixed_pairs(
    tokenizer, model, n, split, seed, template_id, criterion,
    verdict_tokens, pool_multiplier, bias_cache_path,
    bias_ratio: float = 0.5,
) -> list[SwapPair]:
    """
    Load a pool, run verdicts on both orderings, then return n pairs sampled
    so that ~bias_ratio fraction are positionally biased and the rest are unbiased.

    This gives Experiment 1/2/4 high signal (mostly biased pairs) while
    preserving enough unbiased pairs for the Experiment 3 ROC contrast.
    """
    import torch

    pool_n = n * pool_multiplier
    device = next(model.parameters()).device

    # --- Load or compute bias labels ---
    cached: dict[int, dict] = {}
    if bias_cache_path and Path(bias_cache_path).exists():
        with open(bias_cache_path) as f:
            for line in f:
                rec = json.loads(line)
                cached[rec["pair_id"]] = rec
        print(f"  Loaded {len(cached)} bias labels from cache: {bias_cache_path}")

    raw_pairs = load_dataset_pairs(split=split, n=pool_n, seed=seed)
    swap_pairs_pool = [
        _build_swap_pair(i, p, tokenizer, template_id, criterion)
        for i, p in enumerate(raw_pairs)
    ]

    token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in verdict_tokens]
    uncached = [sp for sp in swap_pairs_pool if sp.pair_id not in cached]

    if uncached:
        print(f"  Screening {len(uncached)} pairs for positional bias...")
        cache_file = open(bias_cache_path, "a") if bias_cache_path else None
        try:
            for sp in tqdm(uncached, desc="bias screening"):
                v_ab = _get_verdict(model, tokenizer, sp.prompt_ab, verdict_tokens, token_ids, device)
                v_ba = _get_verdict(model, tokenizer, sp.prompt_ba, verdict_tokens, token_ids, device)
                rec = {"pair_id": sp.pair_id, "verdict_ab": v_ab,
                       "verdict_ba": v_ba, "biased": v_ab == v_ba}
                cached[sp.pair_id] = rec
                if cache_file:
                    cache_file.write(json.dumps(rec) + "\n")
                    cache_file.flush()
        finally:
            if cache_file:
                cache_file.close()

    # Attach labels
    for sp in swap_pairs_pool:
        rec = cached.get(sp.pair_id, {})
        sp.verdict_ab = rec.get("verdict_ab")
        sp.verdict_ba = rec.get("verdict_ba")
        sp.biased     = rec.get("biased")

    biased   = [sp for sp in swap_pairs_pool if sp.biased]
    unbiased = [sp for sp in swap_pairs_pool if sp.biased is False]

    n_biased_target   = min(int(n * bias_ratio), len(biased))
    n_unbiased_target = min(n - n_biased_target, len(unbiased))

    selected = biased[:n_biased_target] + unbiased[:n_unbiased_target]

    print(f"  Pool: {len(biased)} biased, {len(unbiased)} unbiased")
    print(f"  Selected: {n_biased_target} biased + {n_unbiased_target} unbiased = {len(selected)} pairs")

    return selected


def _get_verdict(model, tokenizer, prompt, verdict_tokens, token_ids, device) -> str:
    import torch
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :]
    restricted = {t: logits[tid].item() for t, tid in zip(verdict_tokens, token_ids)}
    return max(restricted, key=restricted.get)
