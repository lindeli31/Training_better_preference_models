import dspy

from src.prompt_optimization.dspy.JudgeSignature import JudgeSignature


class JudgeModule(dspy.Module):
    def __init__(self, reason: bool):
        super().__init__()
        if reason:
            self.prog = dspy.ChainOfThought(JudgeSignature)
        else:
            self.prog = dspy.Predict(JudgeSignature)

    def forward(self, answer_candidate_a, answer_candidate_b, conversation_history):
        return self.prog(
            answer_candidate_a=answer_candidate_a,
            answer_candidate_b=answer_candidate_b,
            conversation_history=conversation_history,
        )
