from typing import List

from core.Message import Message
from core.PermPair import PermPair
from core.Prompt import Prompt
from core.Template import Template


class HumanTemplate(Template):
    generation_instructions: str = (
        "Given the context and two responses, indicate your preference by choosing the corresponding letter: 'A' if the first response is better, '0' if they are equal, '+' if the second response is better. "
        "Keep your answer grounded in the quality and relevance of the responses to the context. "
        "Return only the letter, without any additional text or punctuation."
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
            .add_text(f"answer candidate A: {answer_candidate_b}\n")
            .add_text("ANSWER:") # should we keep it?
        )

        return Prompt(messages=[system_message, user_message])
