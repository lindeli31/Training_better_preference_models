from datasets import load_dataset, Dataset


def analyze_run(
        repo_id: str,
        dataset_name: str,
        split_name: str,
) -> str:
    dataset = load_dataset(path=repo_id, name=dataset_name, split=split_name)

    consistency = _get_consistency(dataset)
    bias_toward_first_position = _get_bias_toward_first_position(dataset)
    bias_toward_second_position = _get_bias_toward_second_position(dataset)
    tie_inconsistency_rate = _get_tie_inconsistency_rate(dataset)

    hard_accuracy = _get_hard_accuracy(dataset)
    soft_accuracy = _get_soft_accuracy(dataset)
    ab_accuracy = _get_ab_accuracy(dataset)
    ba_accuracy = _get_ba_accuracy(dataset)
    accuracy_gap = ab_accuracy - ba_accuracy

    return f"""
    ==================================================
    Run Analysis: {repo_id} | Split: {split_name}
    ==================================================

    [ Accuracy Metrics ]
    --------------------------------------------------
    Hard Accuracy:            {hard_accuracy:.2f}
    Soft Accuracy:            {soft_accuracy:.2f}
    AB Accuracy:              {ab_accuracy:.2f}
    BA Accuracy:              {ba_accuracy:.2f}
    Accuracy Gap:             {accuracy_gap:.2f}

    [ Bias & Consistency ]
    --------------------------------------------------
    Consistency:              {consistency:.2f}
    Bias (First Position):    {bias_toward_first_position:.2f}
    Bias (Second Position):   {bias_toward_second_position:.2f}
    Tie Inconsistency Rate:   {tie_inconsistency_rate:.2f}
    ==================================================
    """.strip()

def _get_consistency(dataset: Dataset) -> float:
    filtered_dataset = dataset.filter(
        lambda x: x["consistency"] == 1.0
    )
    return len(filtered_dataset) / len(dataset)

def _get_bias_toward_first_position(dataset: Dataset) -> float:
    filtered_dataset = dataset.filter(
        lambda x: x["bias_toward_first_position"] == True
    )
    return len(filtered_dataset) / len(dataset)

def _get_bias_toward_second_position(dataset: Dataset) -> float:
    filtered_dataset = dataset.filter(
        lambda x: x["bias_toward_second_position"] == True
    )
    return len(filtered_dataset) / len(dataset)

def _get_tie_inconsistency_rate(dataset: Dataset) -> float:
    filtered_dataset = dataset.filter(
        lambda x: x["tie_inconsistency"] == True
    )
    return len(filtered_dataset) / len(dataset)

def _get_hard_accuracy(dataset: Dataset) -> float:
    filtered_dataset = dataset.filter(
        lambda x: x["ab_correct"] == True and x["ba_correct"] == True
    )
    return len(filtered_dataset) / len(dataset)

def _get_soft_accuracy(dataset: Dataset) -> float:
    filtered_dataset = dataset.filter(
        lambda x: x["ab_correct"] == True
    )
    return len(filtered_dataset) / len(dataset)

def _get_ab_accuracy(dataset: Dataset) -> float:
    filtered_correct_dataset = dataset.filter(
        lambda x: (x["ab_correct"] == True and x["ground_truth"] == 0) # 0 -> "A"
               or (x["ba_correct"] == True and x["ground_truth"] == 1) # 1 -> "B"
    )
    filtered_total_dataset = dataset.filter(
        lambda x: x["ground_truth"] == 0 or  x["ground_truth"] == 1
    )
    return len(filtered_correct_dataset) / len(filtered_total_dataset)

def _get_ba_accuracy(dataset: Dataset) -> float:
    filtered_correct_dataset = dataset.filter(
        lambda x: (x["ba_correct"] == True and x["ground_truth"] == 0) # 0 -> "A"
               or (x["ab_correct"] == True and x["ground_truth"] == 1) # 1 -> "B"
    )
    filtered_total_dataset = dataset.filter(
        lambda x: x["ground_truth"] == 0 or  x["ground_truth"] == 1
    )
    return len(filtered_correct_dataset) / len(filtered_total_dataset)
