import os

import typer
from datasets import disable_caching
from dotenv import load_dotenv
from huggingface_hub import login

from core.LLM import LLM
from core.Template import Template
from eval.EvalProcessor import EvalProcessor
from models.SwissAiLLM import SwissAiLLM
from src.datasets.HelpSteer3 import get_help_steer_3
from scripts.util import dataset_exists
from templates.HumanTemplate import HumanTemplate


def evaluate_model(
        repo_id: str,
        dataset_name: str,
        model_name: str,
        generation_template_name: str,
        reasoning: bool,
        number_of_records: int | None = None,
):
    load_dotenv()
    HF_TOKEN = os.environ['HF_TOKEN']
    CSCS_SERVING_API = os.environ['CSCS_SERVING_API']
    login(token=HF_TOKEN)
    disable_caching()

    split_name = model_name + "___" + generation_template_name
    if number_of_records is not None:
        split_name += f"_{number_of_records}"

    if dataset_exists(repo_id, dataset_name, split_name):
        print(f"Dataset {dataset_name} with split {split_name} already exists in {repo_id}. Skipping.")
        return

    dataset = get_help_steer_3() #TODO: make it flexible, based on dataset_name
    if number_of_records is not None:
        dataset = dataset.select(range(number_of_records))

    generation_llm: LLM = SwissAiLLM()
    generation_template: Template = HumanTemplate()
    processor = EvalProcessor(
        template=generation_template,
        generation_llm=generation_llm,
        reasoning=reasoning,
    )
    print(f"Using EvalProcessor with:\n- generation model: {model_name}\n- generation template: {generation_template.__class__.__name__}\n")

    if isinstance(generation_llm, SwissAiLLM):
        num_proc = 8
    else:
        num_proc = 1
    print(f"Processing dataset with {num_proc} processes.")
    processed_dataset = dataset.map(processor, num_proc=num_proc)

    processed_dataset.push_to_hub(
        repo_id=repo_id,
        config_name=dataset_name,
        split=split_name,
        token=HF_TOKEN,
    )


if __name__ == "__main__":
    typer.run(evaluate_model)