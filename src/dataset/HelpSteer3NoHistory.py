from datasets import Dataset, load_dataset

from src.dataset.util import balance_groundtruth


def _get_context_length(example: dict):
    context_str = "".join(turn['content'] for turn in example['context'])
    total_length = len(context_str) + len(example['response1']) + len(example['response2'])
    return total_length

def _filter_record(example: dict) -> bool:
    total_length = _get_context_length(example)

    if total_length > 5000: return False

    return True

def _process_record(example: dict, id: int):
    score: int = example['overall_preference']

    match score:
        case -2 | -3:  # answer candidate "a" is much better
            ground_truth = -2
        case -1:  # answer candidate "a" is better
            ground_truth = -1
        case 0:  # both answers are equally good
            ground_truth = 0
        case 1:  # answer candidate "b" is better
            ground_truth = 1
        case 2 | 3:  # answer candidate "b" is much better
            ground_truth = 2
        case _:
            raise ValueError(f"Unexpected ground truth value: {score}")

    return {
        'id': id,
        'context': [],  # we remove the context for this dataset variant
        'answer_candidate_a': example['response1'],
        'answer_candidate_b': example['response2'],
        'ground_truth': ground_truth,
        'metadata': {
            'context_length': _get_context_length(example),
            'history_length': len(example['context'])
        },
    }


def get_help_steer_3_no_history() -> Dataset:
    dataset = load_dataset('nvidia/HelpSteer3', 'preference', split='train')
    dataset = dataset.filter(_filter_record)
    dataset = dataset.map(_process_record, with_indices=True, remove_columns=dataset.column_names)
    dataset = balance_groundtruth(dataset)
    return dataset
