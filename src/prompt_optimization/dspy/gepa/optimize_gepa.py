import os

import dspy
from dataset import disable_caching
from dotenv import load_dotenv

from src.dataset.HelpSteer2 import get_help_steer_2
from src.prompt_optimization.dspy.adapt_dataset_for_dspy import adapt_dataset_for_dspy, get_train_test_split
from src.prompt_optimization.dspy.JudgeModule import JudgeModule
from src.prompt_optimization.dspy.PermbatchModule import PermbatchModule
from prompt_optimization.dspy.gepa.metrics_gepa import eval_metric, metric_with_feedback, accuracy_score_metric, consistency_score_metric

disable_caching()
load_dotenv()

# wandb.login(os.environ['WANDB_API_KEY'])
# weave.init(project_name="DSPy")

CSCS_SERVING_API = os.environ['CSCS_SERVING_API']
base_url = "https://api.swissai.svc.cscs.ch/v1"
task_lm = dspy.LM(
    "openai/swiss-ai/Apertus-8B-Instruct-2509",
    temperature=1.0,
    api_key=CSCS_SERVING_API,
    api_base=base_url
)
reflection_lm = dspy.LM( # Strong model required for GEPA's reflection
    "openai/meta-llama/Llama-3.3-70B-Instruct",
    temperature=1.0,
    api_key=CSCS_SERVING_API,
    api_base=base_url
)

dspy.configure(lm=task_lm)

dataset = get_help_steer_2()
dataset = adapt_dataset_for_dspy(dataset)[:100]
train_dataset, test_dataset = get_train_test_split(dataset)

student_model = JudgeModule(reason=False)
wrapper_model = PermbatchModule(student_model)

combine_metric_evaluate = dspy.Evaluate(devset=test_dataset, metric=eval_metric, num_threads=32)
accuracy_metric_evaluate = dspy.Evaluate(devset=test_dataset, metric=accuracy_score_metric, num_threads=32)
consistency_metric_evaluate = dspy.Evaluate(devset=test_dataset, metric=consistency_score_metric, num_threads=32)

print(f"Combined metric: {combine_metric_evaluate(wrapper_model).score}")
print(f"\n\nAccuracy metric: {accuracy_metric_evaluate(wrapper_model).score}")
print(f"\n\nConsistency metric: {consistency_metric_evaluate(wrapper_model).score}")

optimizer = dspy.GEPA(
    metric=metric_with_feedback,
    reflection_lm=reflection_lm,
    auto="light",
)
optimized_wrapper = optimizer.compile(
    student=wrapper_model,
    trainset=train_dataset,
    valset=test_dataset,
)

print(f"Combined metric: {combine_metric_evaluate(optimized_wrapper).score}")
print(f"\n\nAccuracy metric: {accuracy_metric_evaluate(optimized_wrapper).score}")
print(f"\n\nConsistency metric: {consistency_metric_evaluate(optimized_wrapper).score}")
