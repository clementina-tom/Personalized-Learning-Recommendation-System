"""
plrs.llm.providers.openai
==========================
OpenAI provider for PLRS.

Install:
    pip install plrs[openai]
    # or: pip install openai

Usage:
    from plrs.llm.providers.openai import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4o-mini")
    response = provider.complete("What are the prerequisites for calculus?")

Environment variable:
    OPENAI_API_KEY — required
"""

from __future__ import annotations

import os
from typing import Iterator

from plrs.llm.base import LLMProvider

DEFAULT_MODEL      = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider (GPT-4o, GPT-4o-mini, GPT-3.5-turbo, etc.)

    Parameters
    ----------
    model : str
        Model identifier (default: gpt-4o-mini — cheapest capable model).
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    base_url : str, optional
        Custom base URL for OpenAI-compatible APIs (e.g. Together AI, Groq, Fireworks).
    max_tokens : int
        Default max tokens per completion.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model    = model
        self.api_key  = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.max_tokens = max_tokens
        self._client  = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Run: pip install plrs[openai]  or  pip install openai"
                )
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not found. "
                    "Set OPENAI_API_KEY environment variable or pass api_key= to OpenAIProvider."
                )
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _build_messages(self, prompt: str, system: str | None) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system),
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        client = self._get_client()
        stream = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system),
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return f"OpenAIProvider({self.model})"
