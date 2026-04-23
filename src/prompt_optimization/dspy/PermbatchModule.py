import itertools

import dspy


class PermbatchModule(dspy.Module):
    def __init__(self, evaluation_module):
        super().__init__()
        # We pass the core module in so DSPy tracks it
        self.evaluation_module = evaluation_module

    def forward(self, answer_candidate_a, answer_candidate_b, conversation_history):
        answers = []

        # Generate all possible (A, B) permutations
        perms = [
            (answer_candidate_a, answer_candidate_b),
            (answer_candidate_b, answer_candidate_a)
        ]

        # Run the model to get an answer for each permutation
        for p in perms:
            answer = self.evaluation_module(answer_candidate_a=p[0], answer_candidate_b=p[1], conversation_history=conversation_history)
            answers.append(answer.preference)

        return dspy.Prediction(answers=answers)