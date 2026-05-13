from typing import List

from src.core.Message import Message
from src.core.PermPair import PermPair
from src.core.Prompt import Prompt
from src.core.Template import Template


class HumanTemplate(Template):
    generation_instructions: str = (
        "Evaluate which of two responses better answers the conversation. "
        "The evaluation scale is from -2 (the first answer is much better) to +2 (the second answer is much better). "
        "Return -1 if the first answer candidate is better, and +1 if the second answer candidate is better. "
        "Return 0 if they are equal. "
        "Your judgment must be accurate and consistent regardless of the order the responses are presented. "
        "Give only one number as the answer."
    )
    
    def render(self, perm_pair: PermPair) -> List[Prompt]:
        return [
            self._from_perm_to_prompt(perm_pair.context, perm_pair.answer_candidate_a, perm_pair.answer_candidate_b, perm_pair.metadata),
            self._from_perm_to_prompt(perm_pair.context, perm_pair.answer_candidate_b, perm_pair.answer_candidate_a, perm_pair.metadata),
        ]

    def _from_perm_to_prompt(
            self,
            context,
            answer_candidate_a,
            answer_candidate_b,
            metadata,
    ) -> Prompt:
        system_message = (
            Message(role="system")
            .add_text(self.generation_instructions)
        )

        user_message = (
            Message(role="user")
            .add_context(context)
            .add_text("RESPONSES:\n")
            .add_text(f"answer candidate A: {answer_candidate_a}\n")
            .add_text(f"answer candidate B: {answer_candidate_b}\n")
            .add_text("ANSWER:") # should we keep it?
        )

        return Prompt(messages=[system_message, user_message])
