import os
import time
from typing import Callable, TypeVar

import instructor
from litellm import completion
from openai import OpenAI
from pydantic import BaseModel, Field

from src.core.LLM import LLM
from src.core.Prompt import Prompt
from src.core.Response import Response

T = TypeVar('T')


def _call_with_retry(func: Callable[[], T], max_timeout: int = 60) -> T | None:
    """
    Execute a function with retry logic and timeout handling.

    Retries on timeout/connection errors with delays of 4, 8, and 16 seconds.
    Returns None if all retries are exhausted.

    Args:
        func: Callable that performs the API call
        max_timeout: Timeout for each attempt in seconds

    Returns:
        Result from func, or None if all retries fail
    """
    retry_delays = [4, 8, 16]  # seconds
    attempts = retry_delays + [None]  # None represents final attempt

    for attempt_idx, delay in enumerate(attempts):
        try:
            return func()
        except (TimeoutError, OSError, ConnectionError, KeyboardInterrupt) as e:
            if delay is not None and delay > 0:
                print(f"Attempt {attempt_idx + 1} failed with {type(e).__name__}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                # All retries exhausted
                print(f"All retry attempts failed for API call.")
                return None
        except Exception as e:
            # Unexpected error, log and return None
            print(f"Unexpected error during API call: {type(e).__name__}: {e}")
            return None


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
        """Generate response without reasoning using retry-wrapped API call."""
        def api_call():
            answer = completion(
                model=f"openai/{self.model_name}",
                messages=[{"role": m.role, "content": m.content} for m in prompt.messages],
                base_url=self.base_url,
                api_key=self.api_key,
                n=1,
                temperature=0.0,
                timeout=60
            )
            message = answer.choices[0].message.content
            return message if message is not None else ""

        result = _call_with_retry(api_call)
        return Response(message=result if result is not None else "")

    def _generate_with_reasoning(self, prompt: Prompt) -> Response:
        """Generate response with reasoning using retry-wrapped API call."""
        def api_call():
            client = instructor.from_openai(OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=60,
            ))
            answer = client.chat.completions.create(
                model=self.model_name,
                response_model=ReasonedAnswer,
                messages=[{"role": m.role, "content": m.content} for m in prompt.messages],
                temperature=0.0,
            )
            return answer

        answer = _call_with_retry(api_call)
        if answer is None:
            return Response(message="", reasoning="")
        return Response(message=answer.answer, reasoning=answer.reasoning)


class ReasonedAnswer(BaseModel):
    reasoning: str = Field(description="Step-by-step logic to solve the problem")
    answer: str = Field(description="The final concise answer")
