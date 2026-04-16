"""
plrs.llm.providers.claude
==========================
Anthropic Claude provider for PLRS.

Install:
    pip install plrs[claude]
    # or: pip install anthropic

Usage:
    from plrs.llm.providers.claude import ClaudeProvider

    provider = ClaudeProvider(model="claude-sonnet-4-20250514")
    response = provider.complete("What are the prerequisites for calculus?")

    # Streaming
    for chunk in provider.stream("Explain quadratic equations simply."):
        print(chunk, end="", flush=True)

Environment variable:
    ANTHROPIC_API_KEY — required
"""

from __future__ import annotations

import os
from typing import Iterator

from plrs.llm.base import LLMProvider

DEFAULT_MODEL      = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 1024


class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Parameters
    ----------
    model : str
        Model identifier (default: claude-sonnet-4-20250514).
    api_key : str, optional
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    max_tokens : int
        Default max tokens per completion.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model      = model
        self.api_key    = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._client    = None   # lazy init

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Run: pip install plrs[claude]  or  pip install anthropic"
                )
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not found. "
                    "Set ANTHROPIC_API_KEY environment variable or pass api_key= to ClaudeProvider."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        client = self._get_client()
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        if temperature > 0:
            kwargs["temperature"] = temperature

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return f"ClaudeProvider({self.model})"
