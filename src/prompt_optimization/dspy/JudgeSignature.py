import dspy


class JudgeSignature(dspy.Signature):
    """Evaluate which of two responses better answers the conversation. The evaluation scale is from -2 (the first answer is much better) to +2 (the second answer is much better). Return -1 if the first answer candidate is better, and +1 if the second answer candidate is better. Return 0 if they are equal. Your judgment must be accurate and consistent regardless of the order the responses are presented. Give only one number as the answer."""

    conversation_history = dspy.InputField(desc="Messages in conversation history")
    answer_candidate_a = dspy.InputField(desc="First possible answer")
    answer_candidate_b = dspy.InputField(desc="Second possible answer")

    preference = dspy.OutputField(desc="The final decision. You MUST output exactly one number between -2 and +2.")