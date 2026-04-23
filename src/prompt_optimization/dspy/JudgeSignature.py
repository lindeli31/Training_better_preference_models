import dspy


class JudgeSignature(dspy.Signature):
    """Evaluate which of two responses better answers the conversation. You must choose 'A' for the first candidate, 'B' for the second, or 'C' if they are equal. Your judgment must be accurate and consistent regardless of the order the responses are presented."""

    conversation_history = dspy.InputField(desc="Messages in conversation history")
    answer_candidate_a = dspy.InputField(desc="First possible answer")
    answer_candidate_b = dspy.InputField(desc="Second possible answer")

    preference = dspy.OutputField(desc="The final decision. You MUST output exactly one letter: 'A', 'B', or 'C'.")