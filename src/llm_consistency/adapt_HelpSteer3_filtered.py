from typing import List

from datasets import Dataset, load_dataset


def _filter_record(example: dict) -> bool:
    context_str = "".join(turn["content"] for turn in example["context"])
    total_length = len(context_str) + len(example["response1"]) + len(example["response2"])

    if total_length > 1000: return False

    return True


def _process_record(example: dict):
    context = ""
    for message in example["context"]:
        context += f"Message from: {message['role']}\n"
        context += message["content"]
        context += f"\n\n"
    shards: List[str] = [example["response1"], example["response2"]]
    score: int = example["overall_preference"]

    match score:
        case 1 | 2 | 3:
            overall_preference = 1
        case 0:
            overall_preference = -1
        case -1 | -2 | -3:
            overall_preference = 0
        case _:
            raise ValueError(f"Unexpected ground truth value: {score}")

    return {
        "answer_candidates": shards,
        "conversation_history": context,
        "ground_truth": overall_preference,
        # "choices": ["First response is better", "They are equal", "Second response is better"],
    }


def adapt_HelpSteer3_filtered() -> Dataset:
    dataset = load_dataset("nvidia/HelpSteer3", "preference", split="train")
    dataset = dataset.filter(_filter_record)
    dataset = dataset.map(_process_record, remove_columns=dataset.column_names)
    return dataset