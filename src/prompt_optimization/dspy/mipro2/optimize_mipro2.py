import os

import dspy
import wandb
import weave
from datasets import disable_caching
from dotenv import load_dotenv

from src.dataset.HelpSteer2 import get_help_steer_2
from src.prompt_optimization.dspy.adapt_dataset_for_dspy import adapt_dataset_for_dspy, get_train_test_split
from src.dataset.HelpSteer3 import get_help_steer_3
from src.prompt_optimization.dspy.JudgeModule import JudgeModule
from src.prompt_optimization.dspy.PermbatchModule import PermbatchModule
from src.prompt_optimization.dspy.mipro2.metrics_mipro2 import eval_metric, metric_with_feedback, accuracy_score_metric, consistency_score_metric

disable_caching()
load_dotenv()

# wandb.login(os.environ['WANDB_API_KEY'])
# weave.init(project_name="DSPy")

CSCS_SERVING_API = os.environ['CSCS_SERVING_API']
base_url = "https://api.swissai.svc.cscs.ch/v1"
task_lm = dspy.LM(
    "openai/google/gemma-4-31B-it-TEqd",
    temperature=0.0,
    api_key=CSCS_SERVING_API,
    api_base=base_url
)

dspy.configure(lm=task_lm)

dataset = get_help_steer_3().shuffle(seed=42)
dataset = adapt_dataset_for_dspy(dataset)[:1000]
train_dataset, test_dataset = get_train_test_split(dataset)

student_model = JudgeModule(reason=False)
wrapper_model = PermbatchModule(student_model)

combine_metric_evaluate = dspy.Evaluate(devset=test_dataset, metric=eval_metric, num_threads=32)
accuracy_metric_evaluate = dspy.Evaluate(devset=test_dataset, metric=accuracy_score_metric, num_threads=32)
consistency_metric_evaluate = dspy.Evaluate(devset=test_dataset, metric=consistency_score_metric, num_threads=32)

print(f"Combined metric: {combine_metric_evaluate(wrapper_model).score}")
print(f"\n\nAccuracy metric: {accuracy_metric_evaluate(wrapper_model).score}")
print(f"\n\nConsistency metric: {consistency_metric_evaluate(wrapper_model).score}")

optimizer = dspy.MIPROv2(
    metric=metric_with_feedback,
    auto="medium",
)
optimized_wrapper = optimizer.compile(
    student=wrapper_model,
    trainset=train_dataset,
    valset=test_dataset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)

print(f"Combined metric: {combine_metric_evaluate(optimized_wrapper).score}")
print(f"\n\nAccuracy metric: {accuracy_metric_evaluate(optimized_wrapper).score}")
print(f"\n\nConsistency metric: {consistency_metric_evaluate(optimized_wrapper).score}")

print("Best instructions:")
print(optimized_wrapper.evaluation_module.prog.signature.instructions)

optimized_wrapper.save("optimized_wrapper.json")

# Inspect ALL predictors MIPROv2 saw
print("=" * 80)
print("Predictors discovered by DSPy:")
for name, predictor in optimized_wrapper.named_predictors():
    print(f"\n--- {name} ---")
    print(f"Instructions:\n{predictor.signature.instructions}")
    print(f"\nNumber of demos: {len(predictor.demos)}")
    for i, demo in enumerate(predictor.demos):
        print(f"\nDemo {i}:")
        print(demo)