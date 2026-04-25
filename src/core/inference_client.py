"""
inference_client.py
-------------------
This module implements the SwissAIClient, an asynchronous client for calling
the Swiss AI Stack API as an LLM judge. It includes:
- InferenceConfig: a configuration class for setting API parameters and retry logic.
- JudgeResponse: a structured class to encapsulate the model's response, including the extracted label and reasoning.
- extract_label: a robust function to extract the A/B/C judgment from the model's output, handling various formats and edge cases.
- extract_thinking: a function to extract the CoT reasoning block if present in the model's output.
- SwissAIClient: the main client

Remark: We canc hange the function to extract the answer also depending on the model.
"""

# Standard library imports
import asyncio
import json
import logging
import re
import time
from typing import Optional
import aiohttp
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass for the inference client. The following parameters can be set:
# The following parameters are the most relevant for controlling the inference:
# - base_url: the base URL of the API endpoint (default: "https://api.swissai.cscs.ch/v1")
# - api_key: the API key for authentication (default: "SWISSAI_API_KEY", should be set from environment variable or config file)
# - model: the model to use for inference (default: "meta-llama/Llama-3.3-70B-Instruct", can be changed to any model available on the stack)
# - max_tokens: the maximum number of tokens to generate (default: 2048)
# - temperature: the sampling temperature (default: 0.0 for deterministic output)
# These parameters are more related to the retry logic and concurrency, and can usually be left at their default values:
# - top_p: the nucleus sampling parameter (default: 1.0 for no nucleus sampling)
# - max_retries: the maximum number of retries for failed requests (default: 5
# - retry_base_delay: the base delay for retries (default: 2.0 seconds)
# - retry_max_delay: the maximum delay for retries (default: 60.0 seconds)
# - concurrent_requests: the maximum number of concurrent requests (default: 8)
# 
# ---------------------------------------------------------------------------

class InferenceConfig:
    def __init__(
        self,
        base_url: str = "https://api.swissai.cscs.ch/v1",
        api_key: str = "SWISSAI_API_KEY",
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
        concurrent_requests: int = 8,
        extra_body: dict | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.concurrent_requests = concurrent_requests
        self.extra_body = extra_body or {}

# ---------------------------------------------------------------------------
# Response dataclass
# The JudgeResponse class encapsulates the structured response from the judge call.
# As before the most important fields for analysis are:
# - prompt_id: the ID of the prompt (for tracking)
# - experiment_id: the ID of the experiment (for tracking)
# - condition: the experimental condition (e.g. "AB", "BA", "template_v1", ...)
# - raw_text: the full raw text output from the model (for logging and debugging)
# - label: the extracted label ("A", "B", "C")
# - thinking: the extracted CoT block (if present)
#
# The following fields are more related to debugging and analysis of the inference process, and can be logged but are not essential for the main experiments:
# - latency_s: the latency of the API call in seconds
# - total_tokens: the total number of tokens used in the API call (if available)
# - parse_ok: whether the label extraction was successful
# - error: any error message if the API call failed
# ---------------------------------------------------------------------------
class JudgeResponse:
    def __init__(
        self,
        prompt_id: str,
        experiment_id: str,
        condition: str,
        raw_text: str,
        label: Optional[str] = None,
        thinking: Optional[str] = None,
        latency_s: float = 0.0,
        total_tokens: int = 0,
        parse_ok: bool = True,
        error: Optional[str] = None,
    ):
        self.prompt_id = prompt_id
        self.experiment_id = experiment_id
        self.condition = condition               # e.g. "AB", "BA", "template_v1", ...
        self.raw_text = raw_text                 # full model output
        self.label = label                       # "A", "B", "C", or None if parse fails
        self.thinking = thinking                 # extracted CoT block if present
        self.latency_s = latency_s
        self.total_tokens = total_tokens
        self.parse_ok = parse_ok
        self.error = error

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "experiment_id": self.experiment_id,
            "condition": self.condition,
            "raw_text": self.raw_text,
            "label": self.label,
            "thinking": self.thinking,
            "latency_s": self.latency_s,
            "total_tokens": self.total_tokens,
            "parse_ok": self.parse_ok,
            "error": self.error,
        }

# ---------------------------------------------------------------------------
# Label extractor The problem of extracting a clear A/B/C label from the model's 
# output can be tricky (see documentation), especially if the model provides 
# reasoning before giving the final judgment. 
# The extract_label function implements a robust approach to handle this: 
# The extract_label function tries multiple regex patterns to find 
# a clear A/B/C label in the model's output. It also has a fallback to find the 
# last uppercase A/B/C in the text if the more structured patterns fail. 
# The extract_thinking function looks for a CoT block enclosed in 
# <tool_call>...<tool_call> tags, which is a common format for reasoning in Qwen3. 
# This allows us to separate the final judgment (label) from the reasoning process 
# (thinking) if both are present in the output.
# ---------------------------------------------------------------------------

# Matches the final judgment label (A / B / C) even when the model reasons first.
_LABEL_PATTERNS = [
    # explicit verdict lines  e.g. "Verdict: B"  "Answer: A"
    r"(?i)(?:verdict|answer|response|output|better response)\s*:\s*([ABC])\b",
    # bare label at the very end of the text
    r"(?i)\b([ABC])\s*$",
    # label followed by punctuation
    r"(?i)\b([ABC])[.\s]*\Z",
]
def extract_label(text: str) -> tuple[Optional[str], bool]:
    """Return (label, parse_ok). Tries multiple patterns in order."""
    # Strip CoT thinking block if present (common Qwen3 format)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    for pattern in _LABEL_PATTERNS:
        m = re.search(pattern, clean)
        if m:
            return m.group(1).upper(), True

    # Fallback: find last uppercase A/B/C in the text
    letters = re.findall(r"\b([ABC])\b", clean)
    if letters:
        return letters[-1].upper(), True

    return None, False


# ----------------------------------------------------------------------------
# Thinking extractor (optional) The extract_thinking function looks for a CoT 
# block enclosed in <think>...</think> tags, which is a common format for reasoning in Qwen3.
#  This allows us to separate the final judgment (label) from the reasoning process (thinking)
#  if both are present in the output.
# ----------------------------------------------------------------------------
def extract_thinking(text: str) -> Optional[str]:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else None

# ---------------------------------------------------------------------------
# Async Inference Client
#
# The SwissAIClient is an asynchronous HTTP client that communicates with an
# OpenAI-compatible LLM endpoint (Swiss AI Stack). It handles the full lifecycle
# of a judge call: sending prompts to the model, receiving raw text responses,
# and parsing them into structured JudgeResponse objects.
#
# Key features:
# - Concurrency control: a semaphore limits the number of simultaneous API
#   requests (default 8), preventing the server from being overwhelmed.
# - Retry with exponential back-off: if the server returns a 429 (rate limit)
#   or a transient error, the client automatically retries with increasing wait
#   times (2s, 4s, 8s, ...) up to a configurable maximum.
# - Label extraction: each raw model output is parsed to extract the final
#   A/B/C judgment label and, if present, the chain-of-thought reasoning block.
# - Batch processing: the batch_judge method dispatches all calls in parallel
#   using asyncio.gather, making it efficient to evaluate hundreds of prompt
#   pairs across multiple experimental conditions.
# ---------------------------------------------------------------------------

class SwissAIClient:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.concurrent_requests)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=300),
        )
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Core request
    # This method implements the core logic for making a POST request 
    # to the API endpoint, with built-in retry logic that handles rate 
    # limiting (HTTP 429) and other transient errors. It uses exponential 
    # back-off for retries, and logs warnings when retries are triggered. 
    # If all retries are exhausted, it raises an exception.
    # ------------------------------------------------------------------

    async def _post(self, messages: list[dict]) -> dict:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            **self.config.extra_body,
        }

        delay = self.config.retry_base_delay
        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(url, json=payload) as resp:
                    if resp.status == 429:          # rate limited
                        wait = min(delay * (2 ** attempt), self.config.retry_max_delay)
                        logger.warning("Rate limited. Retrying in %.1fs (attempt %d)", wait, attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                wait = min(delay * (2 ** attempt), self.config.retry_max_delay)
                logger.warning("Request error: %s. Retrying in %.1fs", e, wait)
                await asyncio.sleep(wait)

        # All retries exhausted (e.g. repeated 429s)
        raise RuntimeError(f"All {self.config.max_retries} retries exhausted for {url}")

    # ------------------------------------------------------------------
    # Public judge call
    # The judge method is the main public interface for calling the model as an LLM judge.
    # It takes the system prompt, user prompt, and metadata (prompt_id, experiment_id
    # condition) as input, constructs the messages for the API call, and then calls the _post method.
    # It also measures the latency of the API call, and uses the extract_label and extract_thinking 
    # functions to parse the model's output. The result is returned as a JudgeResponse object,
    # ------------------------------------------------------------------

    async def judge(
        self,
        system_prompt: str,
        user_prompt: str,
        prompt_id: str,
        experiment_id: str,
        condition: str,
    ) -> JudgeResponse:
        """
        Call the model as an LLM judge. Returns a JudgeResponse.
        Reasoning depth is controlled entirely by the prompt template.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        async with self._semaphore:
            t0 = time.perf_counter()
            try:
                raw = await self._post(messages)
                latency = time.perf_counter() - t0

                choice = raw["choices"][0]
                full_text = choice["message"]["content"]
                total_tokens = raw.get("usage", {}).get("total_tokens", 0)

                label, parse_ok = extract_label(full_text)
                thinking = extract_thinking(full_text)

                return JudgeResponse(
                    prompt_id=prompt_id,
                    experiment_id=experiment_id,
                    condition=condition,
                    raw_text=full_text,
                    label=label,
                    thinking=thinking,
                    latency_s=round(latency, 3),
                    total_tokens=total_tokens,
                    parse_ok=parse_ok,
                )

            except Exception as exc:
                latency = time.perf_counter() - t0
                logger.error("Judge call failed for prompt %s: %s", prompt_id, exc)
                return JudgeResponse(
                    prompt_id=prompt_id,
                    experiment_id=experiment_id,
                    condition=condition,
                    raw_text="",
                    parse_ok=False,
                    error=str(exc),
                    latency_s=round(latency, 3),
                )

    # ------------------------------------------------------------------
    # Batch helper
    # The batch_judge method is a helper that allows calling the judge method on a batch of inputs concurrently.
    # It takes a list of dictionaries (each containing the kwargs for a judge call), and returns a list of JudgeResponse objects in the same order. It uses asyncio.gather to run the
    # ------------------------------------------------------------------

    async def batch_judge(self, calls: list[dict]) -> list[JudgeResponse]:
        """
        calls: list of kwargs dicts for .judge()
        Returns responses in the same order.
        """
        tasks = [self.judge(**kw) for kw in calls]
        return await asyncio.gather(*tasks)
