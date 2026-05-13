from typing import List, Optional

from core.Response import Response
from src.core.LLM import LLM
from src.core.PermPair import PermPair
from src.core.Prompt import Prompt
from src.core.Template import Template
from src.core.inference_client import extract_label

import re


class EvalProcessor:
    def __init__(
            self,
            template: Template,
            generation_llm: LLM,
            reasoning: bool,
    ):
        self.template = template
        self.generation_llm = generation_llm
        self.reasoning = reasoning

    def __call__(self, example: dict) -> dict:
        self._generate_responses(example)
        self._evaluate_responses(example)
        return example

    def _generate_responses(self, example: dict):
        if example.get("answers"):
            return

        perm_pair: PermPair = PermPair(
            context=example["context"],
            answer_candidate_a=example["answer_candidate_a"],
            answer_candidate_b=example["answer_candidate_b"],
            ground_truth=example["ground_truth"],
            metadata=example["metadata"],
        )
        prompts: List[Prompt] = self.template.render(perm_pair)

        answers = []
        for prompt in prompts:
            response = self.generation_llm.generate(prompt, self.reasoning)
            answer = response.message
            reasoning = response.reasoning # TODO: do we need the reasoning?
            answers.append(answer)

        example["answers"] = answers

    def _evaluate_responses(self, example: dict):
        answers = example["answers"]

        # Extract labels from responses
        extracted_labels = []
        parse_failures = 0
        for answer in answers:
            label, parse_ok = _get_score(answer)
            extracted_labels.append(label)
            if not parse_ok:
                parse_failures += 1

        example["extracted_labels"] = extracted_labels
        example["parse_failures"] = parse_failures
        if parse_failures > 0:
            # Set default values for metrics when parsing failed
            example["ab_difference"] = None
            example["ba_difference"] = None
            example["shift"] = None
            example["bias_toward_first_position"] = None
            example["bias_toward_second_position"] = None
            example["tie_inconsistency"] = None
            return

        # sanity check
        if len(extracted_labels) != 2:
            raise Exception(f"extracted labels is of size {len(extracted_labels)} but not 2. It contains [{', '.join(extracted_labels)}].")

        # Compute accuracy metrics
        ground_truth = example["ground_truth"]
        gold_ab_label = ground_truth
        gold_ba_label = ground_truth * -1
        ab_difference = extracted_labels[0] - gold_ab_label
        ba_difference = extracted_labels[1] - gold_ba_label

        example["ab_difference"] = ab_difference
        example["ba_difference"] = ba_difference

        # Compute consistency metrics
        shift = extracted_labels[0] + extracted_labels[1] # AB - (-BA) = AB + BA
        bias_toward_first_position = extracted_labels[0] < 0 and extracted_labels[1] < 0
        bias_toward_second_position = extracted_labels[0] > 0 and extracted_labels[1] > 0
        tie_inconsistency = (extracted_labels[0] == 0 or extracted_labels[1] == 0) and extracted_labels[0] != extracted_labels[1]

        example["shift"] = shift
        example["bias_toward_first_position"] = bias_toward_first_position
        example["bias_toward_second_position"] = bias_toward_second_position
        example["tie_inconsistency"] = tie_inconsistency

def _get_score(text: str) -> tuple[Optional[int], bool]:
    _NUMBER_RE = r"[+-]?\d+"
    _NUMBER_PATTERNS = [
        # explicit verdict lines  e.g. "Score: -1"  "Answer: 3"
        rf"(?i)(?:verdict|answer|response|output|score|rating)\s*:\s*({_NUMBER_RE})(?!\d)",
        # bare number at the very end of the text
        rf"({_NUMBER_RE})\s*$",
        # number followed by punctuation at end
        rf"({_NUMBER_RE})[.\s]*\Z",
    ]
    # Strip CoT thinking block if present (common Qwen3 format)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for pattern in _NUMBER_PATTERNS:
        m = re.search(pattern, clean)
        if m:
            try:
                return int(m.group(1)), True
            except ValueError:
                continue
    # Fallback: find the last number in the text
    numbers = re.findall(_NUMBER_RE, clean)
    if numbers:
        try:
            return int(numbers[-1]), True
        except ValueError:
            pass
    return None, False
