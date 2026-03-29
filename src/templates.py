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

_OUTPUT_INSTRUCTION_BLIND = (
    "Output only a single character: 1 if the first response is better, "
    "2 if the second response is better, or C if they are equally good."
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


def _build_user_prompt_blind(prompt: str, response_a: str, response_b: str,
                              criterion: str = "better overall") -> str:
    """Neutral 1/2 labels — removes alphabetical-ordering associations from A/B."""
    return (
        f"[User Prompt]\n{prompt}\n\n"
        f"[Response 1]\n{response_a}\n\n"
        f"[Response 2]\n{response_b}\n\n"
        f"Which response is {criterion}?"
    )


def _build_user_prompt_criterion_first(prompt: str, response_a: str, response_b: str,
                                        criterion: str = "better overall") -> str:
    """Criterion question placed before responses — primes the judge while reading."""
    return (
        f"[User Prompt]\n{prompt}\n\n"
        f"Which response is {criterion}?\n\n"
        f"[Response A]\n{response_a}\n\n"
        f"[Response B]\n{response_b}"
    )


def _build_user_prompt_blind_criterion_first(prompt: str, response_a: str, response_b: str,
                                              criterion: str = "better overall") -> str:
    """Both: 1/2 labels and criterion question before responses."""
    return (
        f"[User Prompt]\n{prompt}\n\n"
        f"Which response is {criterion}?\n\n"
        f"[Response 1]\n{response_a}\n\n"
        f"[Response 2]\n{response_b}"
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

# B5: blind-framing system prompt — identical body to expert_rater, only output
# instruction swapped to 1/2/C so B5 isolates user prompt structure, not wording.
SYSTEM_EXPERT_RATER_BLIND = (
    "You are an expert rater. You will be provided with a user prompt and two "
    "assistant responses. Your task is to determine which response is better.\n\n"
    + _OUTPUT_INSTRUCTION_BLIND
)

SYSTEM_OPRO = (
    "You are an impartial AI judge designed to evaluate two responses to a user "
    "query with maximum objectivity. Your sole task is to determine which response "
    "better satisfies the user's intent based on quality, accuracy, completeness, "
    "clarity, and relevance\u2014regardless of presentation order or labeling.\n\n"
    "To eliminate position bias, follow this process:\n"
    "1. **Blind Evaluation**: Treat both responses as anonymous. Mentally swap "
    "their positions and re-evaluate.\n"
    "2. **Counterfactual Check**: Ask: *If I saw Response B first, would I still "
    "judge A as better?* Repeat for the reverse.\n"
    "3. **Consistency Requirement**: Your judgment must remain unchanged after "
    "swapping. If it shifts, the responses are likely equal.\n"
    "4. **Final Mapping**: Only after stable evaluation, assign:\n"
    "   - **A** if the response originally labeled A is superior\n"
    "   - **B** if the response originally labeled B is superior\n"
    "   - **C** if both are indistinguishable in quality or your preference "
    "reverses when order is imagined swapped\n\n"
    "Output **only one letter**: A, B, or C.\n"
    "Your verdict must be invariant to order. If it isn\u2019t, default to C."
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
        "description": "OPRO-optimised prompt (position bias: 0.993 val consistency)",
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

    # --- User-prompt structure variants (B5) ---
    "blind": {
        "system": SYSTEM_EXPERT_RATER_BLIND,
        "user_fn": _build_user_prompt_blind,
        "description": "Blind framing: responses labeled 1/2 instead of A/B, criterion after",
    },
    "criterion_first": {
        "system": SYSTEM_EXPERT_RATER,
        "user_fn": _build_user_prompt_criterion_first,
        "description": "Standard A/B labels, criterion question placed before responses",
    },
    "blind_criterion_first": {
        "system": SYSTEM_EXPERT_RATER_BLIND,
        "user_fn": _build_user_prompt_blind_criterion_first,
        "description": "Blind 1/2 labels + criterion question placed before responses",
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
