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

# Standard library imports
from typing import Callable

# ---------------------------------------------------------------------------
# Shared suffix (output instruction) - included in all system prompts.
# This makes sure that all templates ask the model to output in the same format
# (single letter: A, B, or C), so that the output parsing is consistent across templates.
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
# This function takes the original prompt, responses, and criterion, and builds the user prompt.
# ---------------------------------------------------------------------------
def _build_user_prompt(prompt: str, response_a: str, response_b: str,
                        criterion: str = "better overall") -> str:
    '''Builds the user prompt given the original prompt, responses, and criterion.'''
    return (
        f"[User Prompt]\n{prompt}\n\n"
        f"[Response A]\n{response_a}\n\n"
        f"[Response B]\n{response_b}\n\n"
        f"Which response is {criterion}?"
    )

# ---------------------------------------------------------------------------
# System-prompt variants  (Experiment B2 – template sensitivity)
# This is useful in the experiments B2 to check how much the model's judgments are sensitive to the wording and framing of the prompt.
# We have a primary template (expert_rater) and several variants that differ in wording, style, and framing. 
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

# OPRO best prompt for Llama 3.3-70B, taken verbatim from
# comparison/best_prompts/meta-llama_Llama-3.3-70B-Instruct__opro.md.
# Identical wording to the SYSTEM_OPRO_LLAMA on the main branch. The output
# instruction is embedded in the prompt itself (last paragraph), so we do
# NOT append _OUTPUT_INSTRUCTION here.
SYSTEM_OPRO_LLAMA = (
    "You are an impartial and objective evaluator tasked with comparing two responses to a user "
    "query, with the primary goal of eliminating any bias related to the order of presentation. "
    "To achieve this, consider the responses in a completely abstract and neutral manner, focusing "
    "exclusively on their inherent quality, relevance, and effectiveness in addressing the user's query.\n\n"
    "Imagine you have access to both responses simultaneously, without any knowledge of which one "
    "was provided first. Evaluate each response based on its clarity, accuracy, coherence, and "
    "overall ability to satisfy the user's information needs, using the following criteria:\n\n"
    "1. Relevance: How well does each response address the user's query?\n"
    "2. Accuracy: How accurate is the information provided in each response?\n"
    "3. Clarity: How clear and easy to understand is each response?\n"
    "4. Completeness: How comprehensive is each response in providing a satisfactory answer?\n\n"
    "To ensure a balanced assessment, ask yourself a series of questions that apply equally to "
    "both responses, such as:\n"
    "- Which response provides the most accurate and relevant information?\n"
    "- Which response is clearer and easier to understand?\n"
    "- Which response better addresses the user's query and provides a more comprehensive answer?\n"
    "- Which response demonstrates a deeper understanding of the topic or question at hand?\n\n"
    "When comparing the responses, consider the following scenarios to minimize position bias:\n"
    "- If the responses are identical or nearly identical in terms of quality and relevance, output C.\n"
    "- If one response is significantly better than the other in terms of accuracy, clarity, and "
    "completeness, output A if that response is the first one, or output B if it is the second one.\n"
    "- If the responses are of comparable quality, but one has a slight edge over the other, output "
    "A or B accordingly, ensuring that your decision is based solely on the intrinsic merits of each response.\n\n"
    "Output A if one response is significantly better, B if the other response is significantly "
    "better, or C if both responses are of comparable quality, ensuring that your judgment is fair, "
    "unbiased, and based on the inherent quality of each response. By adopting this approach, you "
    "will contribute to a balanced evaluation that reflects the true quality of each response, "
    "regardless of its presentation order."
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
    "overall":    "better overall, considering helpfulness, correctness, and coherence",
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
    "assistant responses. Rate each response on the following criteria providing a score between 0 and 4 for each criteria:\n"
    "1. Helpfulness: How well does the response address the user's needs?\n"
    "2. Accuracy: Is the information factually correct?\n"
    "3. Coherence: Is the response well-organized and easy to follow?\n\n"
    "After rating each response on these criteria, provide your final verdict.\n\n"
    + _OUTPUT_INSTRUCTION_VERBOSE
)

# OPRO best prompt for Llama 3.3-70B layered with step-by-step reasoning.
# Uses SYSTEM_OPRO_LLAMA verbatim (output instruction already embedded) and
# appends a step-by-step reasoning instruction. The judge is therefore asked
# to reason explicitly through the four OPRO criteria before applying the
# already-defined OPRO output rule.
SYSTEM_OPRO_LLAMA_REASON_THEN_JUDGE = (
    SYSTEM_OPRO_LLAMA
    + "\n\nBefore producing your final verdict according to the rule above, "
    "reason step-by-step about your evaluation: walk through each of the four "
    "criteria (relevance, accuracy, clarity, completeness) and explicitly "
    "compare the two responses on each one, identifying their main strengths "
    "and weaknesses. Only after this reasoning, output your single-letter "
    "verdict (A, B, or C)."
)

SYSTEM_TREE_OF_THOUGHTS_JUDGE = (
    "You are an expert rater. You will be provided with a user prompt and two "
    "assistant responses. To ensure an unbiased and rigorous evaluation, you must "
    "employ a Tree of Thoughts process. Follow these exact steps before providing your verdict:\n\n"
    "Step 1: Brainstorming (Generate Perspectives)\n"
    "Create three distinct analytical branches:\n"
    "- Branch A: Construct the strongest possible argument for why Response 1 is superior.\n"
    "- Branch B: Construct the strongest possible argument for why Response 2 is superior.\n"
    "- Branch C: Identify any critical flaws, missed constraints, or hallucinations in either response.\n\n"
    "Step 2: Evaluation (Critique the Branches)\n"
    "Critically assess each branch against the original user prompt. Which arguments "
    "hold up to scrutiny? Are any of the arguments based on superficial formatting rather "
    "than factual accuracy and helpfulness? Label each branch as 'Valid', 'Weak', or 'Invalid'.\n\n"
    "Step 3: Pruning & Synthesis\n"
    "Discard the Weak and Invalid branches. Using only the Valid arguments, write a "
    "brief synthesis that definitively compares the two responses.\n\n"
    "Step 4: Final Verdict\n"
    "Based purely on the synthesized conclusion from Step 3, provide your final verdict.\n\n"
    + _OUTPUT_INSTRUCTION_VERBOSE
)

# ---------------------------------------------------------------------------
# Language variants  (Experiment B4 – input sensitivity)
# Same expert_rater body translated into Polish, German and Italian.
# Output instructions are also translated, but the letters A/B/C are kept
# verbatim because the parser expects them.
# The user prompt is NOT translated — the dataset content stays in English.
# ---------------------------------------------------------------------------

SYSTEM_EXPERT_RATER_PL = (
    "Jesteś ekspertem oceniającym. Otrzymasz pytanie użytkownika oraz dwie "
    "odpowiedzi asystenta. Twoim zadaniem jest określenie, która odpowiedź "
    "jest lepsza.\n\n"
    "Wyprowadź tylko jedną literę: A jeśli pierwsza odpowiedź jest lepsza, "
    "B jeśli druga odpowiedź jest lepsza, lub C jeśli są równie dobre."
)

SYSTEM_EXPERT_RATER_DE = (
    "Sie sind ein erfahrener Bewerter. Sie erhalten eine Benutzeranfrage und "
    "zwei Antworten eines Assistenten. Ihre Aufgabe ist es zu bestimmen, "
    "welche Antwort besser ist.\n\n"
    "Geben Sie nur einen einzigen Buchstaben aus: A wenn die erste Antwort "
    "besser ist, B wenn die zweite Antwort besser ist, oder C wenn sie "
    "gleich gut sind."
)

SYSTEM_EXPERT_RATER_IT = (
    "Sei un valutatore esperto. Ti verranno fornite una domanda dell'utente "
    "e due risposte di un assistente. Il tuo compito è determinare quale "
    "risposta sia migliore.\n\n"
    "Restituisci solo una singola lettera: A se la prima risposta è "
    "migliore, B se la seconda risposta è migliore, o C se sono ugualmente "
    "buone."
)

SYSTEM_EXPERT_RATER_ZH = (
    "你是一位专家评审员。你将收到一个用户提示和两个助手回复。"
    "你的任务是判断哪个回复更好。\n\n"
    "仅输出一个字母:如果第一个回复更好,输出 A;如果第二个回复更好,"
    "输出 B;如果两个回复同样好,输出 C。"
)

# ---------------------------------------------------------------------------
# Template registry. Maps template_id to system prompt and user prompt builder function.
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
    "opro_llama": {
        "system": SYSTEM_OPRO_LLAMA,
        "user_fn": _build_user_prompt,
        "description": "OPRO best prompt for Llama-3.3-70B (letter-only output)",
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
    "opro_llama_reason_then_judge": {
        "system": SYSTEM_OPRO_LLAMA_REASON_THEN_JUDGE,
        "user_fn": _build_user_prompt,
        "description": "OPRO best prompt for Llama-3.3-70B + reason-then-judge",
    },
    "tree_of_thoughts_judge": {
        "system": SYSTEM_TREE_OF_THOUGHTS_JUDGE,
        "user_fn": _build_user_prompt,
        "description": "Tree-of-Thoughts: brainstorm, evaluate, prune, then verdict",
    },

    # --- Language variants (B4 - input sensitivity) ---
    # System prompt + output instructions translated; user prompt stays in English.
    "expert_rater_pl": {
        "system": SYSTEM_EXPERT_RATER_PL,
        "user_fn": _build_user_prompt,
        "description": "Expert-rater (Polish system prompt + output instructions)",
    },
    "expert_rater_de": {
        "system": SYSTEM_EXPERT_RATER_DE,
        "user_fn": _build_user_prompt,
        "description": "Expert-rater (German system prompt + output instructions)",
    },
    "expert_rater_it": {
        "system": SYSTEM_EXPERT_RATER_IT,
        "user_fn": _build_user_prompt,
        "description": "Expert-rater (Italian system prompt + output instructions)",
    },
    "expert_rater_zh": {
        "system": SYSTEM_EXPERT_RATER_ZH,
        "user_fn": _build_user_prompt,
        "description": "Expert-rater (Chinese system prompt + output instructions)",
    },
}


def build_prompt(
    template_id: str,
    prompt: str,
    response_a: str,
    response_b: str,
    criterion: str = "overall",
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
