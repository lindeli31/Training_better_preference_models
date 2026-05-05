from datasets import get_dataset_config_names, get_dataset_split_names


def dataset_exists(repo_id, config_name, split_name):
    try:
        configs = get_dataset_config_names(repo_id)
        if config_name not in configs:
            return False
        splits = get_dataset_split_names(repo_id, config_name)
        return split_name in splits
    except Exception:
        return False
