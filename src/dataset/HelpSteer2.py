from datasets import Dataset, load_dataset


def get_help_steer_2() -> Dataset:
    dataset = load_dataset('nvidia/HelpSteer2', split='train')

    processed_data = {
        "id": [],
        "context": [],
        "answer_candidate_a": [],
        "answer_candidate_b": [],
        "ground_truth": [],
        "metadata": []
    }

    # Iterate through the dataset in steps of 2 to pair the adjacent responses
    for i in range(0, len(dataset), 2):
        row_a = dataset[i]
        row_b = dataset[i + 1]

        # Safety check: ensure we are comparing responses for the exact same prompt
        if row_a['prompt'] != row_b['prompt']:
            raise ValueError(f"Prompt mismatch at index {i} and {i + 1}")

        # 1. Format Context
        context = [{"role": "user", "content": row_a['prompt']}]

        # 2. Extract Answers
        ans_a = row_a['response']
        ans_b = row_b['response']

        # 3. Determine Ground Truth using the 'helpfulness' score
        # TODO: calculate the scores for a and b the right way!
        score_a = row_a['helpfulness']
        score_b = row_b['helpfulness']

        if score_a > score_b:
            ground_truth = 0
        elif score_a < score_b:
            ground_truth = 1
        else:
            ground_truth = 2

        # 4. Calculate Metadata (Context Length)
        context_len = len(row_a['prompt']) + len(ans_a) + len(ans_b)

        # Append to our unified dictionary
        processed_data["id"].append(i/2)
        processed_data["context"].append(context)
        processed_data["answer_candidate_a"].append(ans_a)
        processed_data["answer_candidate_b"].append(ans_b)
        processed_data["ground_truth"].append(ground_truth)
        processed_data["metadata"].append({"context_length": context_len})

    # Convert the processed dictionary back into a HuggingFace Dataset
    return Dataset.from_dict(processed_data)