"""Unified LLM client. All agents use this instead of their own _llm_call().

Supports both OpenAI-compatible (/chat/completions) and Anthropic Messages (/v1/messages)
API formats. Format is auto-detected from the base URL.
"""
import json
import time
import logging
from typing import Any

import requests

from backend.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Shared LLM API client with retry logic. Auto-detects API format."""

    def __init__(self):
        self.settings = get_settings()
        self._base = self.settings.ds_base_url.rstrip("/")
        # Auto-detect API format from base URL
        self._anthropic_api = "anthropic" in self._base.lower()

    @property
    def _headers(self) -> dict:
        h = {"Authorization": f"Bearer {self.settings.ds_api_key}"}
        if self._anthropic_api:
            h["anthropic-version"] = "2023-06-01"
        return h

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> str | None:
        """Standard chat completion. Returns content string or None on failure."""
        if not self.settings.ds_api_key:
            return None

        if self._anthropic_api:
            return self._chat_anthropic(prompt, system=system, temperature=temperature,
                                        max_tokens=max_tokens, response_format=response_format,
                                        model=model)
        else:
            return self._chat_openai(prompt, system=system, temperature=temperature,
                                     max_tokens=max_tokens, response_format=response_format,
                                     model=model)

    def _chat_openai(self, prompt, *, system, temperature, max_tokens, response_format, model):
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": model or "deepseek-v4-flash",
            "messages": msgs,
            "max_tokens": max_tokens or 4000,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if response_format == "json_object":
            body["response_format"] = {"type": "json_object"}

        for attempt in range(self.settings.llm_max_retries):
            try:
                r = requests.post(
                    f"{self._base}/chat/completions",
                    headers=self._headers,
                    json=body,
                    timeout=(30, self.settings.llm_timeout),  # (connect, read)
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                if r.status_code == 429:
                    wait = self.settings.retry_backoff ** attempt
                    logger.warning(f"LLM rate limited, retry in {wait}s")
                    time.sleep(wait)
                    continue
            except Exception:
                if attempt < self.settings.llm_max_retries - 1:
                    wait = self.settings.retry_backoff ** attempt
                    time.sleep(wait)
        return None

    def _chat_anthropic(self, prompt, *, system, temperature, max_tokens, response_format, model):
        """Anthropic Messages API format."""
        msgs: list[dict] = [{"role": "user", "content": prompt}]

        # thinking model consumes tokens for internal reasoning — double the budget
        effective_max = max_tokens or 4000
        body: dict[str, Any] = {
            "model": model or "deepseek-v4-flash",
            "messages": msgs,
            "max_tokens": effective_max * 2,  # headroom for thinking blocks
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature

        # Note: Anthropic API doesn't have native json_object response_format.
        # The prompt itself should request JSON output.

        for attempt in range(self.settings.llm_max_retries):
            try:
                r = requests.post(
                    f"{self._base}/v1/messages",
                    headers=self._headers,
                    json=body,
                    timeout=(30, self.settings.llm_timeout),  # (connect, read)
                )
                if r.status_code == 200:
                    data = r.json()
                    return self._extract_text(data)
                if r.status_code == 429:
                    wait = self.settings.retry_backoff ** attempt
                    logger.warning(f"LLM rate limited (429), retry in {wait}s")
                    time.sleep(wait)
                    continue
                # Log non-200, non-429 errors
                logger.warning(f"LLM HTTP {r.status_code}: {r.text[:200]}")
            except requests.exceptions.Timeout:
                logger.warning(f"LLM request timeout after {self.settings.llm_timeout}s")
            except Exception as e:
                logger.warning(f"LLM request error: {e}")
            if attempt < self.settings.llm_max_retries - 1:
                wait = self.settings.retry_backoff ** attempt
                time.sleep(wait)
        return None

    def _extract_text(self, data: dict) -> str:
        """Extract text from Anthropic Messages response content blocks.

        Handles text blocks, thinking blocks, and tool_use blocks.
        """
        content = data.get("content", [])
        texts = []
        block_types = set()
        for block in content:
            if isinstance(block, dict):
                block_types.add(block.get("type", "unknown"))
                if block.get("type") in ("text", "thinking"):
                    texts.append(block.get("text", ""))
        if not texts and block_types:
            logger.warning(f"LLM response has no text/thinking blocks. Types: {block_types}")
        return "\n".join(texts) if texts else ""

    def reason(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> str | None:
        """Reasoning mode for A6 Risk Officer. Uses thinking-capable model."""
        system = ("ultrathink 加强逻辑推理能力。你是独立审查官。对投资决策做客观评估。发现风险点但不过度否决。"
                  if self._anthropic_api else
                  "ultrathink 加强逻辑推理能力。你是严格的风控官，对投资决策做魔鬼代言人式质疑。不容忍逻辑漏洞。")
        return self.chat(
            prompt,
            system=system,
            model=model,
            temperature=0.7,
            max_tokens=max_tokens or 8000,
            response_format=response_format,
        )

    def chat_json(self, prompt: str, model: str | None = None, **kwargs) -> dict | None:
        """Chat with JSON output, parsed to dict. Use model= to override default."""
        if not self._anthropic_api:
            kwargs.setdefault("response_format", "json_object")
        result = self.chat(prompt, model=model, **kwargs)
        if not result:
            return None
        return extract_json(result)


def extract_json(text: str):
    """Extract JSON object/array from LLM response text. Public utility for agents."""
    # 1) Try pure JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) Try markdown code block
    if "```" in text:
        blocks = text.split("```")
        for i, block in enumerate(blocks):
            if i % 2 == 1:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
    # 3) Try to extract JSON object/array from within text
    for brace_open, brace_close in [("{", "}"), ("[", "]")]:
        start = text.find(brace_open)
        end = text.rfind(brace_close)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
