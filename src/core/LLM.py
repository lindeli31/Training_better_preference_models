from abc import abstractmethod, ABC

from pydantic import BaseModel

from core import Response
from core.Prompt import Prompt


class LLM(BaseModel, ABC):
    @abstractmethod
    def generate(self, prompt: Prompt, reasoning: bool) -> Response:
        raise NotImplementedError("Method generate() must be implemented in the subclass.")
