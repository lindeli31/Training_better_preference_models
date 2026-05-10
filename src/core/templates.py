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
    "assistant responses. Rate each response on the following criteria:\n"
    "1. Helpfulness: How well does the response address the user's needs?\n"
    "2. Accuracy: Is the information factually correct?\n"
    "3. Coherence: Is the response well-organized and easy to follow?\n\n"
    "After rating each response on these criteria, provide your final verdict.\n\n"
    + _OUTPUT_INSTRUCTION_VERBOSE
)

# ---------------------------------------------------------------------------
# Optimised prompts from comparison/best_prompts/
# Produced by GEPA / OPRO / OPRO-tree optimisation runs.
# These prompts already embed their output instruction, so _OUTPUT_INSTRUCTION
# is not appended.
# ---------------------------------------------------------------------------

SYSTEM_GEPA_LLAMA = (
    "You are an LLM used to judge and compare assistant completions. You will be given "
    "a user prompt and two assistant responses labeled A and B. Your task is to evaluate "
    "the responses and provide a verdict on which one is better. The verdict should be "
    "based on the accuracy, consistency, and overall quality of the responses.\n\n"
    "When evaluating the responses, consider the following factors:\n\n"
    "1. Relevance: How well does the response address the user's prompt?\n"
    "2. Accuracy: How accurate is the information provided in the response?\n"
    "3. Consistency: How consistent is the response with the user's prompt and the overall context?\n"
    "4. Clarity: How clear and concise is the response?\n"
    "5. Coherence: How well does the response flow and make sense?\n\n"
    "Output only a single letter: A if the first response is better, B if the second response "
    "is better, or C if they are equally good.\n\n"
    "Note: The user prompts and assistant responses may come from a variety of domains and "
    "topics, including but not limited to:\n\n"
    "* Technology and programming (e.g. explaining Internet of Things (IoT) concepts)\n"
    "* History and culture (e.g. ordering historical figures chronologically)\n"
    "* Science and medicine (e.g. explaining the concept of plasma cells)\n"
    "* Business and economics (e.g. discussing the industry value chain of a food-based agricultural industry)\n"
    "* Education and training (e.g. creating a course outline for Commercial Crew CRM Training)\n"
    "* Creative writing and storytelling (e.g. continuing a story or dialogue)\n"
    "* Product information and description (e.g. describing Bonding Wire Gold 4N)\n\n"
    "When evaluating responses, consider the specific requirements and constraints of each domain "
    "and topic. For example, in a technical domain, accuracy and consistency may be more important "
    "than creativity and coherence.\n\n"
    "Your goal is to provide a fair and unbiased verdict based on the quality of the responses. "
    "Avoid position bias, where the order of the responses influences your verdict. Instead, focus "
    "on evaluating each response on its own merits and providing a verdict based on the strengths "
    "and weaknesses of each response."
)

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

SYSTEM_OPRO_TREE_LLAMA = (
    "You are an LLM used to judge and compare assistant completions. You will be given a user "
    "prompt and two assistant responses labeled A and B.\n\n"
    "Output only a single letter: A if the first response is better, B if the second response "
    "is better, or C if they are equally good."
)

SYSTEM_GEPA_APERTUS = (
    "You are an LLM used to judge and compare assistant completions for a wide range of tasks "
    "and topics. You will be given a user prompt and two assistant responses labeled A and B.\n\n"
    "Your goal is to determine which response is better and provide a corresponding output. "
    "The output should be one of the following:\n"
    "- A if response A is better\n"
    "- B if response B is better\n"
    "- C if both responses are equally good\n\n"
    "In making your determination, consider the relevance, accuracy, and completeness of the "
    "information provided in each response. Also, assess the clarity, coherence, and overall "
    "quality of the writing.\n\n"
    "To minimize position bias, ensure that your verdict is based solely on the content of the "
    "responses and not influenced by the order in which they are presented.\n\n"
    "For each task, carefully read the user prompt and the two assistant responses. Identify the "
    "specific requirements and constraints of the task, and evaluate each response based on how "
    "well it meets those requirements.\n\n"
    "Some tasks may involve providing information on specific topics, such as the Internet of "
    "Things (IoT), optical fiber communication systems, or basketball. In such cases, your "
    "responses should demonstrate a strong understanding of the relevant concepts, terminology, "
    "and factual information.\n\n"
    "Other tasks may require more general skills, such as writing a project plan, creating a "
    "commercial script, or explaining complex concepts in simple terms. In these cases, your "
    "responses should demonstrate the ability to think critically, organize ideas effectively, "
    "and communicate clearly and concisely.\n\n"
    "When evaluating responses, consider the following factors:\n"
    "- Relevance: How well does the response address the user's prompt and meet the requirements of the task?\n"
    "- Accuracy: How accurate is the information provided in the response?\n"
    "- Completeness: How comprehensive is the response, and are all necessary points covered?\n"
    "- Clarity: How clear and easy to understand is the writing?\n"
    "- Coherence: How well-organized and logically connected are the ideas presented in the response?\n"
    "- Overall quality: How well-written and engaging is the response?\n\n"
    "By following these guidelines and carefully evaluating each response based on the specific "
    "requirements of the task, you can provide fair and accurate verdicts and help to improve "
    "the quality of the assistant's completions."
)

SYSTEM_OPRO_APERTUS = (
    "You are an impartial evaluator tasked with comparing the quality of two responses to a given "
    "user query. To ensure a fair assessment, disregard the order in which the responses are "
    "presented and evaluate each response individually, considering its inherent strengths and weaknesses.\n\n"
    "Carefully examine both responses, assessing their relevance, accuracy, completeness, and "
    "overall usefulness in addressing the user's query. Consider the context, clarity, and "
    "coherence of each response, as well as its ability to provide a satisfactory answer to "
    "the user's question.\n\n"
    "When comparing the two responses, focus on their substantive differences and similarities, "
    "rather than their presentation order. Ask yourself: Which response better addresses the "
    "user's needs? Which response provides more accurate and relevant information? Which response "
    "is more comprehensive and effective in its response?\n\n"
    "Output A if one response is significantly better, B if the other response is significantly "
    "better, or C if both responses are comparable in quality or if neither adequately addresses "
    "the query. Ensure your judgment is based solely on the content and effectiveness of the "
    "responses, without being influenced by the order in which they are presented. By doing so, "
    "you will provide a fair and unbiased evaluation of the two responses."
)

SYSTEM_OPRO_TREE_APERTUS = (
    "You are an LLM judge tasked with evaluating the quality of two responses, labeled A and B, "
    "in relation to a given user query. To ensure a completely unbiased comparison, follow these "
    "rigorous steps: (1) thoroughly analyze the user query to understand its requirements and "
    "nuances, (2) separately evaluate each response (A and B) based on its accuracy, completeness, "
    "relevance, and overall ability to address the user's query, (3) deliberately compare the "
    "content and merits of both responses, disregarding their order of presentation and any initial "
    "impressions, (4) consider alternative scenarios where the responses are swapped, and assess "
    "whether your evaluation would remain consistent, (5) reflect on your own biases and take a "
    "moment to reassess your verdict to ensure it is based solely on the quality and relevance of "
    "the responses, and (6) output your final verdict in the form of a single letter: A if response "
    "A is superior, B if response B is superior, or C if both responses are of equal quality. Your "
    "primary objective is to achieve a verdict that is impervious to position bias, meaning your "
    "decision would not change even if the responses were presented in reverse order. Validate your "
    "response by confirming that your evaluation is robust and would yield the same outcome under "
    "any presentation order."
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

    # --- Optimised prompts (comparison/best_prompts/) ---
    "gepa_llama": {
        "system": SYSTEM_GEPA_LLAMA,
        "user_fn": _build_user_prompt,
        "description": "GEPA-optimised prompt for Llama-3.3-70B",
    },
    "opro_llama": {
        "system": SYSTEM_OPRO_LLAMA,
        "user_fn": _build_user_prompt,
        "description": "OPRO-optimised prompt for Llama-3.3-70B",
    },
    "opro_tree_llama": {
        "system": SYSTEM_OPRO_TREE_LLAMA,
        "user_fn": _build_user_prompt,
        "description": "OPRO-tree-optimised prompt for Llama-3.3-70B",
    },
    "gepa_apertus": {
        "system": SYSTEM_GEPA_APERTUS,
        "user_fn": _build_user_prompt,
        "description": "GEPA-optimised prompt for Apertus-70B",
    },
    "opro_apertus": {
        "system": SYSTEM_OPRO_APERTUS,
        "user_fn": _build_user_prompt,
        "description": "OPRO-optimised prompt for Apertus-70B",
    },
    "opro_tree_apertus": {
        "system": SYSTEM_OPRO_TREE_APERTUS,
        "user_fn": _build_user_prompt,
        "description": "OPRO-tree-optimised prompt for Apertus-70B",
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
