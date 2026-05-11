from datasets import Dataset, load_dataset


def _get_context_length(example: dict):
    context_str = "".join(turn['content'] for turn in example['context'])
    total_length = len(context_str) + len(example['response1']) + len(example['response2'])
    return total_length

def _process_record(example: dict):
    score: int = example['overall_preference']

    match score:
        case -1 | -2 | -3:  # answer candidate "a" is better
            ground_truth = 0
        case 1 | 2 | 3:  # answer candidate "b" is better
            ground_truth = 1
        case 0:  # both answers are equally good
            ground_truth = 2
        case _:
            raise ValueError(f"Unexpected ground truth value: {score}")

    return {
        'context': example['context'],
        'answer_candidate_a': example['response1'],
        'answer_candidate_b': example['response2'],
        'ground_truth': ground_truth,
        'metadata': {
            'context_length': _get_context_length(example),
        },
    }


def get_help_steer_3() -> Dataset:
    dataset = load_dataset('nvidia/HelpSteer3', 'preference', split='train')
    dataset = dataset.map(_process_record, remove_columns=dataset.column_names)
    return dataset
