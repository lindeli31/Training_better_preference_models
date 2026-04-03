import os

import dspy
import wandb
import weave
from datasets import disable_caching
from dotenv import load_dotenv

from llm_consistency.BatchPermutationWrapper import BatchPermutationWrapper
from llm_consistency.ConsistentQA import ConsistentQA
from llm_consistency.adapt_HelpSteer3_filtered import adapt_HelpSteer3_filtered
from llm_consistency.metrics import eval_metric, metric_with_feedback, accuracy_score_metric, consistency_score_metric

disable_caching()
load_dotenv()

WANDB_API_KEY = os.environ['WANDB_API_KEY']
wandb.login(WANDB_API_KEY)
weave.init(project_name="DSPy")

CSCS_SERVING_API = os.environ['CSCS_SERVING_API']
base_url = "https://api.swissai.svc.cscs.ch/v1"
task_lm = dspy.LM(
    "openai/meta-llama/Llama-3.3-70B-Instruct",
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

trainset = [
    dspy.Example(
        answer_candidates=[
            "To bake a chocolate cake, first preheat your oven to 350°F. Mix flour, sugar, cocoa powder, baking soda, and salt in a bowl. In another bowl, combine eggs, milk, oil, and vanilla. Gradually add the wet ingredients to the dry, then pour into a greased pan and bake for 30-35 minutes.",
            "Chocolate cake is delicious. You can buy one at the store instead of baking."
        ],
        conversation_history="How do I bake a chocolate cake?",
        ground_truth=0 # 0->first 1->second -1->equal
    ).with_inputs('answer_candidates', 'conversation_history')
]

dataset = adapt_HelpSteer3_filtered()
list_dataset = dataset.to_list()
adapted_dataset = [
    dspy.Example(
        answer_candidates=row['answer_candidates'],
        conversation_history=row['conversation_history'],
        ground_truth=row['ground_truth'] # 0->first 1->second -1->equal
    ).with_inputs('answer_candidates', 'conversation_history')
    for row in list_dataset
]
train_dataset = adapted_dataset[:50] # 500
validation_dataset = adapted_dataset[50:75]

student_model = ConsistentQA()
wrapper_model = BatchPermutationWrapper(student_model)

############3

response = wrapper_model(
    # answer_candidates=[
    #     "To bake a chocolate cake, first preheat your oven to 350°F. Mix flour, sugar, cocoa powder, baking soda, and salt in a bowl. In another bowl, combine eggs, milk, oil, and vanilla. Gradually add the wet ingredients to the dry, then pour into a greased pan and bake for 30-35 minutes.",
    #     "Chocolate cake is delicious. You can buy one at the store instead of baking."
    # ],
    # conversation_history="How do I bake a chocolate cake?",
    answer_candidates=list_dataset[0]['answer_candidates'],
    conversation_history=list_dataset[0]['conversation_history'],
)
print("answers:", response.answers)
print("history: ", task_lm.inspect_history(n=1))

############3

combine_metric_evaluate = dspy.Evaluate(
    devset=validation_dataset,
    metric=eval_metric,#batch_consistency_metric,
    num_threads=32,
    display_table=True,
    display_progress=True,
)

accuracy_metric_evaluate = dspy.Evaluate(
    devset=validation_dataset,
    metric=accuracy_score_metric,
    num_threads=32,
    display_table=True,
    display_progress=True,
)

consistency_metric_evaluate = dspy.Evaluate(
    devset=validation_dataset,
    metric=consistency_score_metric,
    num_threads=32,
    display_table=True,
    display_progress=True,
)

print("Combined metric: ")
combine_metric_evaluate(wrapper_model)

print("\n\nAccuracy metric: ")
accuracy_metric_evaluate(wrapper_model)

print("\n\nConsistency metric: ")
consistency_metric_evaluate(wrapper_model)


############3 optimisation

optimizer = dspy.GEPA(
    metric=metric_with_feedback,#batch_consistency_metric,
    reflection_lm=reflection_lm,
    auto="light"
)

optimized_wrapper = optimizer.compile(
    student=wrapper_model,
    trainset=train_dataset,
    valset=validation_dataset,
)

production_model = optimized_wrapper.evaluation_module

print("Combined metric: ")
combine_metric_evaluate(optimized_wrapper)

print("\n\nAccuracy metric: ")
accuracy_metric_evaluate(optimized_wrapper)

print("\n\nConsistency metric: ")
consistency_metric_evaluate(optimized_wrapper)




# optimized_wrapper.evaluation_module.prog.predict.signature