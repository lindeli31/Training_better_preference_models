from typing import List

from src.core.LLM import LLM
from src.core.PermPair import PermPair
from src.core.Prompt import Prompt
from src.core.Template import Template
from src.core.inference_client import extract_label


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
            answer = self.generation_llm.generate(prompt, self.reasoning)
            answers.append(answer)

        example["answers"] = answers

    def _evaluate_responses(self, example: dict):
        answers = example["answers"]

        # Extract labels from responses
        extracted_labels = []
        parse_failures = 0
        for answer in answers:
            label, parse_ok = extract_label(answer.message)
            extracted_labels.append(label)
            if not parse_ok:
                parse_failures += 1

        # sanity check
        if len(extracted_labels) != 2:
            raise Exception(f"extracted labels is of size {len(extracted_labels)} but not 2. It contains [{', '.join(extracted_labels)}].")

        # Compute accuracy metrics
        ground_truth = example["ground_truth"]
        # Map ground_truth to label: 0 -> "A", 1 -> "B", 2 -> "C"
        if ground_truth == 0:
            gold_label = "A"
        elif ground_truth == 1:
            gold_label = "B"
        else:
            gold_label = "C"
        flip_map = {"A": "B", "B": "A", "C": "C"}
        gold_ab_label = gold_label
        gold_ba_label = flip_map[gold_label]
        ab_correct = extracted_labels[0] == gold_ab_label
        ba_correct = extracted_labels[1] == gold_ba_label

        example["extracted_labels"] = extracted_labels
        example["parse_failures"] = parse_failures
        example["ab_correct"] = ab_correct
        example["ba_correct"] = ba_correct

        # Compute consistency metrics
        consistency = extracted_labels[0] == flip_map[extracted_labels[1]]
        bias_toward_first_position = extracted_labels[0] == "A" and extracted_labels[1] == "A"
        bias_toward_second_position = extracted_labels[0] == "B" and extracted_labels[1] == "B"
        tie_inconsistency = (extracted_labels[0] == "C" or extracted_labels[1] == "C") and extracted_labels[0] != extracted_labels[1]

        example["consistency"] = consistency
        example["bias_toward_first_position"] = bias_toward_first_position
        example["bias_toward_second_position"] = bias_toward_second_position
        example["tie_inconsistency"] = tie_inconsistency
