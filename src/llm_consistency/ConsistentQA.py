import dspy

from llm_consistency.ConsistencySignature import ConsistencySignature


class ConsistentQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought gives the model room to reason about the order before answering
        # self.prog = dspy.ChainOfThought(ConsistencySignature)
        self.prog = dspy.Predict(ConsistencySignature)

    def forward(self, answer_candidate_a, answer_candidate_b, conversation_history):
        return self.prog(answer_candidate_a=answer_candidate_a, answer_candidate_b=answer_candidate_b, conversation_history=conversation_history)