import os

from datasets import get_dataset_config_names, get_dataset_split_names, disable_caching, Dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import login, get_token


def dataset_exists(repo_id, config_name, split_name):
    try:
        configs = get_dataset_config_names(repo_id)
        if config_name not in configs:
            return False
        splits = get_dataset_split_names(repo_id, config_name)
        return split_name in splits
    except Exception:
        return False

def get_split_name_from_config(
        model_name: str,
        template_name: str,
        reasoning: bool,
        number_of_records: int | None = None,
) -> str:
    split_name = model_name.replace('/', '_').replace('-', '_') + "_" + template_name

    if reasoning:
        split_name += "_reasoning"
    else:
        split_name += "_NOreasoning"

    if number_of_records is not None:
        split_name += f"_{number_of_records}"

    return split_name

def setup_HF():
    if get_token() is not None:
        return

    load_dotenv()
    HF_TOKEN = os.environ['HF_TOKEN']
    login(token=HF_TOKEN)
    disable_caching()

def balance_groundtruth(dataset: Dataset):
    choices = [0, 1, 2]

    # Split by choice
    splits = {
        choice: dataset.filter(lambda x: x['ground_truth'] == choice)
        for choice in choices
    }

    # Downsample each split
    min_count = min(len(splits[choice]) for choice in choices)
    balanced_splits = [
        splits[label].select(range(min_count))
        for label in choices
    ]

    # Concatenate
    result_dataset = concatenate_datasets(balanced_splits).shuffle(seed=42)
    return result_dataset