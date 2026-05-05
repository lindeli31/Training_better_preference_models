import asyncio
import os

from core.LLM import LLM
from core.Prompt import Prompt
from core.Response import Response
from core.inference_client import InferenceConfig, SwissAIClient, extract_label, extract_thinking


class SwissAiLLM(LLM):
    def __init__(self):
        """Initialize SwissAiLLM with configuration from environment variables."""
        super().__init__()
        self.config = InferenceConfig(
            api_key=os.environ.get("SWISSAI_API_KEY", "SWISSAI_API_KEY"),
            base_url=os.environ.get("SWISSAI_BASE_URL", "https://api.swissai.cscs.ch/v1"),
        )

    def generate(self, prompt: Prompt, reasoning: bool) -> Response:
        """
        Generate a response using the Swiss AI Stack API.

        Args:
            prompt: A Prompt object containing a list of Message objects
            reasoning: Whether to include reasoning in the prompt (affects prompt construction)

        Returns:
            A Response object with the generated message and optional reasoning
        """
        # Convert prompt messages to API format
        system_prompt = ""
        user_prompt = ""

        for message in prompt.messages:
            if message.role == "developer" or message.role == "system":
                system_prompt += message.content
            else:  # "user" or default role
                user_prompt += message.content

        # Run the async API call synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to use a different approach
                # This shouldn't happen in the typical workflow, but we handle it gracefully
                import nest_asyncio
                nest_asyncio.apply()
                response_text = loop.run_until_complete(
                    self._call_api(system_prompt, user_prompt)
                )
            else:
                response_text = loop.run_until_complete(
                    self._call_api(system_prompt, user_prompt)
                )
        except RuntimeError:
            # No event loop in current thread, create a new one
            response_text = asyncio.run(self._call_api(system_prompt, user_prompt))

        # Extract label and thinking from the response
        label, parse_ok = extract_label(response_text)
        thinking = extract_thinking(response_text)

        # Return Response object with the full text as message
        # The label extraction happens in EvalProcessor, so we return the full text
        return Response(
            message=response_text,
            reasoning=thinking if reasoning else None,
        )

    async def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the Swiss AI Stack API asynchronously.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model

        Returns:
            The raw text response from the model
        """
        async with SwissAIClient(self.config) as client:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            url = f"{self.config.base_url}/chat/completions"
            payload = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }

            # Use the internal _post method from the client
            raw_response = await client._post(messages)

            # Extract the content from the response
            choice = raw_response["choices"][0]
            response_text = choice["message"]["content"]

            return response_text
