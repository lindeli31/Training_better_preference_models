from typing import List

import dspy
from datasets import Dataset


def _convert_context(conversation_history):
    context = ""
    for message in conversation_history:
        context += f"Message from: {message['role']}\n"
        context += message['content']
        context += f"\n\n"
    return context

def adapt_dataset_for_dspy(dataset: Dataset) -> List[dspy.Example]:
    adapted_dataset = [
        dspy.Example(
            answer_candidate_a=row['answer_candidate_a'],
            answer_candidate_b=row['answer_candidate_b'],
            conversation_history=_convert_context(row['context']),
            ground_truth=row['ground_truth'],
        ).with_inputs('answer_candidate_a', 'answer_candidate_b', 'conversation_history')
        for row in dataset
    ]
    return adapted_dataset

def get_train_test_split(dataset: List[dspy.Example], train_ratio: float = 0.8) -> tuple[List[dspy.Example], List[dspy.Example]]:
    # no shuffle the training data for the sake of reproducibility. Assuming that data order is random.
    train_size: int = int(len(dataset) * train_ratio)
    train_split = dataset[:train_size]
    test_split = dataset[train_size:]
    return train_split, test_split