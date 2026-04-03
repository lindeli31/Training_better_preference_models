import itertools

import dspy


class BatchPermutationWrapper(dspy.Module):
    def __init__(self, evaluation_module):
        super().__init__()
        # We pass the core module in so DSPy tracks it
        self.evaluation_module = evaluation_module

    def forward(self, answer_candidates, conversation_history):
        # facts is a list: ["The server crashed at 3 PM.", "John restarted it at 4 PM."]
        answers = []

        # Generate all possible (A, B) permutations
        perms = list(itertools.permutations(answer_candidates, len(answer_candidates)))

        for p in perms:
            # Run the core module for each permutation
            result = self.evaluation_module(answer_candidate_a=p[0], answer_candidate_b=p[1], conversation_history=conversation_history)
            answers.append(result.preference)

        # Return all answers generated in this batch
        return dspy.Prediction(answers=answers)