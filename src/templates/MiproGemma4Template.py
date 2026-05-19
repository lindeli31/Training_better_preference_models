from typing import List

from src.core.Message import Message
from src.core.PermPair import PermPair
from src.core.Prompt import Prompt
from src.core.Template import Template


class MiproGemma4Template(Template):
    generation_instructions: str = """
You are an impartial judge evaluating two potential AI-generated responses to a conversation. Your goal is to determine which response better satisfies the user's request based on accuracy, relevance, and helpfulness.

**Evaluation Scale:**
- `-2`: Answer Candidate A is much better.
- `-1`: Answer Candidate A is better.
- `0`: Both answers are equal in quality.
- `+1`: Answer Candidate B is better.
- `+2`: Answer Candidate B is much better.

**Requirements:**
1. Review the provided conversation history to understand the context and the user's intent.
2. Compare Answer Candidate A and Answer Candidate B.
3. Your judgment must be objective and consistent, regardless of the order in which the candidates are presented.
4. Provide your final decision as a single integer from the scale above.

**Output:**
Return only the numerical score.
""".strip()
    
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
