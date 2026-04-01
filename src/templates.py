"""
templates.py
------------
All judge prompt templates used across experiments.
Each template is a pair (system_prompt, user_prompt_fn).

The user_prompt_fn takes:
    prompt       : str  - the original dataset prompt
    response_a   : str  - the response to show as "A"
    response_b   : str  - the response to show as "B"
    criterion    : str  - e.g. "more helpful", "better quality"

and returns a formatted string.

Templates are registered in TEMPLATES dict, keyed by a short ID.
"""

from typing import Callable

# ---------------------------------------------------------------------------
# Shared suffix (output instruction) — factored out so it stays consistent
# ---------------------------------------------------------------------------

_OUTPUT_INSTRUCTION = (
    "Output only a single letter: A if the first response is better, "
    "B if the second response is better, or C if they are equally good."
)

_OUTPUT_INSTRUCTION_VERBOSE = (
    "After your evaluation, provide your final verdict by strictly "
    "following this format:\n"
    "Verdict: <A|B|C>\n"
    "where A means the first response is better, B means the second "
    "response is better, and C means they are equally good."
)

# ---------------------------------------------------------------------------
# User-prompt builder (shared across system-prompt variants)
# ---------------------------------------------------------------------------

def _build_user_prompt(prompt: str, response_a: str, response_b: str,
                        criterion: str = "better overall") -> str:
    return (
        f"[User Prompt]\n{prompt}\n\n"
        f"[Response A]\n{response_a}\n\n"
        f"[Response B]\n{response_b}\n\n"
        f"Which response is {criterion}?"
    )


# ---------------------------------------------------------------------------
# System-prompt variants  (Experiment B2 – template sensitivity)
# ---------------------------------------------------------------------------

SYSTEM_EXPERT_RATER = (
    "You are an expert rater. You will be provided with a user prompt and two "
    "assistant responses. Your task is to determine which response is better.\n\n"
    + _OUTPUT_INSTRUCTION
)

SYSTEM_LLM_JUDGE = (
    "You are an LLM used to judge and compare assistant completions. You will be "
    "given a user prompt and two assistant responses labeled A and B.\n\n"
    + _OUTPUT_INSTRUCTION
)

SYSTEM_NEUTRAL = (
    "Compare the following two responses to the given prompt and decide which one "
    "is better.\n\n"
    + _OUTPUT_INSTRUCTION
)

SYSTEM_ACADEMIC = (
    "You are a quality evaluator assessing AI-generated responses. Using rigorous "
    "evaluation criteria, compare the two provided responses with respect to the "
    "given user prompt.\n\n"
    + _OUTPUT_INSTRUCTION
)

SYSTEM_MINIMAL = (
    "Pick the better response: A, B, or C (tie)."
)

SYSTEM_OPRO = (
    "You are an impartial evaluator tasked with comparing the quality of two responses "
    "to a given user query. To ensure a completely unbiased assessment, consider the "
    "responses, labeled A and B, as two separate entities that exist independently of "
    "each other, and evaluate them based on their inherent qualities, such as accuracy, "
    "completeness, relevance, clarity, and usefulness in addressing the user's query.\n\n"
    "Imagine you are evaluating each response in isolation, without knowing which one was "
    "presented first or having any prior knowledge of the other response. Then, consider "
    "both responses together, weighing their strengths and weaknesses in a neutral and "
    "parallel manner, as if they were two separate solutions to the same problem.\n\n"
    "To further minimize any potential bias, evaluate the responses based on a standardized "
    "set of criteria, considering multiple aspects of each, such as how well they address "
    "the query, their clarity, and their usefulness. Ask yourself: If the responses were "
    "swapped, would my evaluation change? If the responses were presented in a different "
    "order, would my judgment be different?\n\n"
    "Additionally, consider the following questions for each response: Does it fully address "
    "the user's query? Is it clear, concise, and easy to understand? Does it provide relevant "
    "and accurate information? Does it offer a unique perspective or solution?\n\n"
    "Ensure that your judgment is based solely on the intrinsic merits of each response, "
    "without being influenced by the order in which they are presented or any extraneous "
    "factors. Carefully consider multiple aspects of each response, evaluating them based on "
    "their individual merits, and then output a single letter: A if one response is superior, "
    "B if the other response is superior, or C if both responses are of equal quality, "
    "ensuring that your decision reflects a balanced and impartial comparison of the "
    "responses' content.\n\n"
    "To finalize your evaluation, take a moment to reflect on your judgment and ask yourself: "
    "Is my decision based on a thorough and unbiased comparison of the responses? Have I "
    "considered all relevant aspects of each response? Is my output consistent with the "
    "principles of impartial evaluation?"
)

# ---------------------------------------------------------------------------
# Criterion variants  (what we ask the model to optimise for)
# ---------------------------------------------------------------------------

CRITERIA = {
    "helpful":    "more helpful to the user",
    "quality":    "of higher overall quality",
    "accurate":   "more accurate and factually correct",
    "harmless":   "safer and more appropriate",
    "concise":    "more concise while still fully answering the question",
}

# ---------------------------------------------------------------------------
# Reasoning variants  (Experiment B3 – reasoning depth)
# ---------------------------------------------------------------------------

SYSTEM_REASON_THEN_JUDGE = (
    "You are an expert rater. You will be provided with a user prompt and two "
    "assistant responses. Explain your reasoning about the strengths and weaknesses "
    "of each response, then provide your final verdict.\n\n"
    + _OUTPUT_INSTRUCTION_VERBOSE
)

SYSTEM_STRUCTURED_REASONING = (
    "You are an expert rater. You will be provided with a user prompt and two "
    "assistant responses. Rate each response on the following criteria:\n"
    "1. Helpfulness: How well does the response address the user's needs?\n"
    "2. Accuracy: Is the information factually correct?\n"
    "3. Coherence: Is the response well-organized and easy to follow?\n\n"
    "After rating each response on these criteria, provide your final verdict.\n\n"
    + _OUTPUT_INSTRUCTION_VERBOSE
)

# ---------------------------------------------------------------------------
# Minor-wording variants  (Experiment B4 – input sensitivity)
# ---------------------------------------------------------------------------

SYSTEM_EXPERT_RATER_ALT1 = SYSTEM_EXPERT_RATER.replace(
    "determine which response is better",
    "identify the superior response",
)

SYSTEM_EXPERT_RATER_ALT2 = SYSTEM_EXPERT_RATER.replace(
    "two assistant responses",
    "two AI-generated answers",
).replace(
    "which response is better",
    "which answer is preferable",
)

SYSTEM_EXPERT_RATER_ALT3 = SYSTEM_EXPERT_RATER.replace(
    "Your task is to determine which response is better.",
    "Evaluate both responses carefully and select the one that better serves the user.",
)

# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TemplateEntry = dict  # {system: str, user_fn: Callable, description: str}

TEMPLATES: dict[str, TemplateEntry] = {
    # --- Primary template (baseline) ---
    "expert_rater": {
        "system": SYSTEM_EXPERT_RATER,
        "user_fn": _build_user_prompt,
        "description": "Baseline expert-rater template",
    },

    # --- Template sensitivity variants (B2) ---
    "llm_judge": {
        "system": SYSTEM_LLM_JUDGE,
        "user_fn": _build_user_prompt,
        "description": "Template framing model as an LLM judge",
    },
    "neutral": {
        "system": SYSTEM_NEUTRAL,
        "user_fn": _build_user_prompt,
        "description": "Neutral imperative template",
    },
    "academic": {
        "system": SYSTEM_ACADEMIC,
        "user_fn": _build_user_prompt,
        "description": "Academic/formal evaluator framing",
    },
    "minimal": {
        "system": SYSTEM_MINIMAL,
        "user_fn": _build_user_prompt,
        "description": "Minimal one-liner template",
    },
    "opro": {
        "system": SYSTEM_OPRO,
        "user_fn": _build_user_prompt,
        "description": "OPRO-optimised prompt (train: 0.749, val: 0.769 position consistency)",
    },

    # --- Reasoning variants (B3) ---
    "reason_then_judge": {
        "system": SYSTEM_REASON_THEN_JUDGE,
        "user_fn": _build_user_prompt,
        "description": "Explain reasoning about strengths/weaknesses, then verdict",
    },
    "structured_reasoning": {
        "system": SYSTEM_STRUCTURED_REASONING,
        "user_fn": _build_user_prompt,
        "description": "Rate on helpfulness/accuracy/coherence criteria, then verdict",
    },

    # --- Minor-wording sensitivity variants (B4) ---
    "expert_rater_alt1": {
        "system": SYSTEM_EXPERT_RATER_ALT1,
        "user_fn": _build_user_prompt,
        "description": "Expert-rater: 'superior response' wording",
    },
    "expert_rater_alt2": {
        "system": SYSTEM_EXPERT_RATER_ALT2,
        "user_fn": _build_user_prompt,
        "description": "Expert-rater: 'AI-generated answers / preferable' wording",
    },
    "expert_rater_alt3": {
        "system": SYSTEM_EXPERT_RATER_ALT3,
        "user_fn": _build_user_prompt,
        "description": "Expert-rater: 'serves the user' wording",
    },
}


def build_prompt(
    template_id: str,
    prompt: str,
    response_a: str,
    response_b: str,
    criterion: str = "helpful",
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for the given template.
    criterion is resolved via CRITERIA dict.
    """
    if template_id not in TEMPLATES:
        raise ValueError(f"Unknown template_id '{template_id}'. "
                         f"Available: {list(TEMPLATES)}")
    entry = TEMPLATES[template_id]
    criterion_str = CRITERIA.get(criterion, criterion)
    system = entry["system"]
    user = entry["user_fn"](prompt, response_a, response_b, criterion_str)
    return system, user
