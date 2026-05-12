from typing import List

import typer
from tqdm.contrib import itertools

from analysis.analyze_run import analyze_run
from dataset.util import setup_HF, get_split_name_from_config
from scripts.evaluate_model import evaluate_model


def evaluate_config(
        repo_id: str,
        dataset_names: List[str],
        model_names: List[str],
        template_names: List[str],
        reasoning: bool,
        number_of_records: int | None = None,
):
    setup_HF()

    combinations = itertools.product(dataset_names, model_names, template_names)
    for dataset_name, model_name, template_name in combinations:
        evaluate_model(
            repo_id=repo_id,
            dataset_name=dataset_name,
            model_name=model_name,
            template_name=template_name,
            reasoning=reasoning,
            number_of_records=number_of_records,
        )
        split_name = get_split_name_from_config(model_name, template_name, reasoning, number_of_records)
        analysis = analyze_run(repo_id=repo_id, dataset_name=dataset_name, split_name=split_name)
        print(analysis)


if __name__ == "__main__":
    typer.run(evaluate_config)
