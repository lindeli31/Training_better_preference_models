"""
inference_client.py
-------------------
Async inference client for the Swiss AI Stack (OpenAI-compatible endpoint).
Wraps Qwen3-30B-A3B-Instruct-2507 (or any model on the stack) with:
  - Rate-limit-aware retry logic with exponential back-off
  - Structured response parsing (A / B / C label extraction)
  - Full raw-response logging for reproducibility
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    base_url: str = "https://api.swissai.cscs.ch/v1"
    api_key: str = "SWISSAI_API_KEY"          # replace or load from env
    model: str = "meta-llama/Llama-3.3-70B-Instruct"

    # Generation params
    max_tokens: int = 2048                     # enough room for CoT + label
    temperature: float = 0.0                   # deterministic for reproducibility
    top_p: float = 1.0

    # Retry / rate-limit
    max_retries: int = 5
    retry_base_delay: float = 2.0             # seconds
    retry_max_delay: float = 60.0
    concurrent_requests: int = 8              # semaphore limit


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class JudgeResponse:
    prompt_id: str
    experiment_id: str
    condition: str                             # e.g. "AB", "BA", "template_v1", ...
    raw_text: str                              # full model output
    label: Optional[str] = None               # "A", "B", "C", or None if parse fails
    thinking: Optional[str] = None            # extracted CoT block if present
    latency_s: float = 0.0
    total_tokens: int = 0
    parse_ok: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Label extractor
# ---------------------------------------------------------------------------

# Matches the final judgment label (A / B / C, or 1 / 2 for blind templates).
# 1/2 are added to the end-anchored patterns only — safe because they are anchored
# to end-of-text, avoiding false positives from numbers in response text.
_LABEL_PATTERNS = [
    # explicit verdict lines  e.g. "Verdict: B"  "Answer: 1"
    r"(?i)(?:verdict|answer|response|output|better response)\s*:\s*([ABC12])\b",
    # bare label at the very end of the text
    r"(?i)\b([ABC12])\s*$",
    # label followed by punctuation
    r"(?i)\b([ABC12])[.\s]*\Z",
]

# Normalise blind-template outputs (1/2) back to A/B so all downstream metrics
# remain compatible without modification.
_BLIND_MAP: dict[str, str] = {"1": "A", "2": "B"}


def extract_label(text: str) -> tuple[Optional[str], bool]:
    """Return (label, parse_ok). Tries multiple patterns in order.

    Labels 1 and 2 (used by blind templates) are normalised to A and B
    respectively so that all downstream metrics are compatible.
    """
    # Strip CoT thinking block if present (common Qwen3 format)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    for pattern in _LABEL_PATTERNS:
        m = re.search(pattern, clean)
        if m:
            raw = m.group(1).upper()
            return _BLIND_MAP.get(raw, raw), True

    # Fallback: find last uppercase A/B/C in the text (1/2 excluded here to
    # avoid false positives from numbers appearing in response content)
    letters = re.findall(r"\b([ABC])\b", clean)
    if letters:
        return letters[-1].upper(), True

    return None, False


def extract_thinking(text: str) -> Optional[str]:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Async inference client
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
    # ------------------------------------------------------------------

    async def _post(self, messages: list[dict]) -> dict:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
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
    # ------------------------------------------------------------------

    async def batch_judge(self, calls: list[dict]) -> list[JudgeResponse]:
        """
        calls: list of kwargs dicts for .judge()
        Returns responses in the same order.
        """
        tasks = [self.judge(**kw) for kw in calls]
        return await asyncio.gather(*tasks)
