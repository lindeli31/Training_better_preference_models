from typing import List

from core.LLM import LLM
from core.PermPair import PermPair
from core.Prompt import Prompt
from core.Template import Template
from core.inference_client import extract_label


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
            answer = self.generation_llm.generate(prompt, self.reasoning)
            answers.append(answer)

        example["answers"] = answers

    def _evaluate_responses(self, example: dict):
        answers = example["answers"]
        ground_truth = example["ground_truth"]

        # Map ground_truth to label: 0 -> "A", 1 -> "B"
        gold_label = "A" if ground_truth == 0 else "B"

        # Extract labels from responses
        extracted_labels = []
        parse_failures = 0
        for answer in answers:
            label, parse_ok = extract_label(answer.message)
            extracted_labels.append(label)
            if not parse_ok:
                parse_failures += 1

        # Compute accuracy metrics
        total_responses = len(extracted_labels)
        correct_predictions = sum(1 for label in extracted_labels if label == gold_label)
        accuracy = correct_predictions / total_responses if total_responses > 0 else 0.0
        parse_success_rate = (total_responses - parse_failures) / total_responses if total_responses > 0 else 0.0

        # Add evaluation metrics to example
        example["extracted_labels"] = extracted_labels
        example["gold_label"] = gold_label
        example["accuracy"] = round(accuracy, 4)
        example["correct_predictions"] = correct_predictions
        example["total_responses"] = total_responses
        example["parse_failures"] = parse_failures
        example["parse_success_rate"] = round(parse_success_rate, 4)
