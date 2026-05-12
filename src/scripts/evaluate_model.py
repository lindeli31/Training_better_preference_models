import os

import typer

from dataset.dataset_catalog import dataset_catalog
from dataset.util import dataset_exists, get_split_name_from_config, setup_HF
from src.core.LLM import LLM
from src.core.Template import Template
from src.eval.EvalProcessor import EvalProcessor
from src.models.SwissAiLLM import SwissAiLLM
from templates.templates_catalog import templates_catalog


def evaluate_model(
        repo_id: str,
        dataset_name: str,
        model_name: str,
        template_name: str,
        reasoning: bool,
        number_of_records: int | None = None,
):
    setup_HF()
    split_name = get_split_name_from_config(model_name, template_name, reasoning, number_of_records)

    if dataset_exists(repo_id, dataset_name, split_name):
        print(f"The dataset {dataset_name}  has already been processed with config {split_name} in {repo_id}. Skipping.")
        return

    dataset = dataset_catalog[dataset_name]()
    if number_of_records is not None:
        dataset = dataset.select(range(number_of_records))

    generation_llm: LLM = SwissAiLLM(model_name)
    generation_template: Template = templates_catalog[template_name]
    processor = EvalProcessor(
        template=generation_template,
        generation_llm=generation_llm,
        reasoning=reasoning,
    )
    print(f"Using EvalProcessor with:\n- model: {model_name}\n- template: {generation_template.__class__.__name__}\n")
    processed_dataset = dataset.map(processor, num_proc=8)

    processed_dataset.push_to_hub(
        repo_id=repo_id,
        config_name=dataset_name,
        split=split_name,
        token=os.environ['HF_TOKEN'],
    )


if __name__ == "__main__":
    typer.run(evaluate_model)