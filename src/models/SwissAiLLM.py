import os
from typing import cast

import instructor
from litellm import completion
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from src.core.LLM import LLM
from src.core.Prompt import Prompt
from src.core.Response import Response


class SwissAiLLM(LLM):
    api_key: str = ""
    base_url: str = "https://api.swissai.cscs.ch/v1"
    model_name: str = ""

    def __init__(self, model_name: str):
        super().__init__()
        self.api_key: str = os.environ['SWISSAI_API_KEY']
        self.model_name = model_name

    def generate(self, prompt: Prompt, reasoning: bool) -> Response:
        if reasoning:
            return self._generate_with_reasoning(prompt)
        else:
            return self._generate_no_reasoning(prompt)

    def _generate_no_reasoning(self, prompt: Prompt) -> Response:
        answer = completion(
            model=f"openai/{self.model_name}",
            messages=[{"role": m.role, "content": m.content} for m in prompt.messages],
            base_url=self.base_url,
            api_key=self.api_key,
            n=1,
            temperature=0.0
        )
        message: str = answer.choices[0].message.content
        return Response(message=message)

    def _generate_with_reasoning(self, prompt: Prompt) -> Response:
        # client = instructor.from_litellm(completion)
        client = instructor.from_openai(OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        ))

        answer: ReasonedAnswer = client.chat.completions.create(
            model=self.model_name,
            response_model=ReasonedAnswer,
            messages=[{"role": m.role, "content": m.content} for m in prompt.messages],
            temperature=0.0,
        )
        return Response(message=answer.answer, reasoning=answer.reasoning)


class ReasonedAnswer(BaseModel):
    reasoning: str = Field(description="Step-by-step logic to solve the problem")
    answer: str = Field(description="The final concise answer")
