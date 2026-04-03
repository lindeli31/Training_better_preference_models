import dspy


class ConsistencySignature(dspy.Signature):
    answer_candidate_a = dspy.InputField(desc="First possible answer")
    answer_candidate_b = dspy.InputField(desc="Second possible answer")
    conversation_history = dspy.InputField(desc="Messages in conversation history")
    preference = dspy.OutputField(desc="the better answer - a or b, or 'equal'")