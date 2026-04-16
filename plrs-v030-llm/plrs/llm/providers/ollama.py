"""
plrs.llm.providers.ollama
==========================
Ollama provider for PLRS — run open-source models locally.

Popular models that work well for curriculum tasks:
    llama3.2        — Meta Llama 3.2 (recommended, fast)
    mistral         — Mistral 7B (good instruction following)
    phi3            — Microsoft Phi-3 (lightweight, capable)
    gemma2          — Google Gemma 2
    qwen2           — Alibaba Qwen 2

Install Ollama:
    https://ollama.com/download
    ollama pull llama3.2

Install Python client:
    pip install plrs[ollama]
    # or: pip install ollama

Usage:
    from plrs.llm.providers.ollama import OllamaProvider

    provider = OllamaProvider(model="llama3.2")
    response = provider.complete("What are prerequisites for calculus?")

    # Streaming
    for chunk in provider.stream("Explain quadratic equations."):
        print(chunk, end="", flush=True)

No API key needed — runs fully locally.
"""

from __future__ import annotations

import os
from typing import Iterator

from plrs.llm.base import LLMProvider

DEFAULT_MODEL   = "llama3.2"
DEFAULT_HOST    = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """
    Ollama provider — local open-source model inference.

    No API key, no usage cost, fully private.

    Parameters
    ----------
    model : str
        Ollama model name (default: llama3.2).
        Run `ollama list` to see installed models.
    host : str
        Ollama server URL (default: http://localhost:11434).
        Override with OLLAMA_HOST env var for remote Ollama servers.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
    ) -> None:
        self.model = model
        self.host  = host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError:
                raise ImportError(
                    "ollama package not installed. "
                    "Run: pip install plrs[ollama]  or  pip install ollama\n"
                    "Also ensure Ollama is running: https://ollama.com/download"
                )
            self._client = ollama.Client(host=self.host)
        return self._client

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat(
            model=self.model,
            messages=messages,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return response.message.content or ""

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for chunk in client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        ):
            content = chunk.message.content
            if content:
                yield content

    def list_models(self) -> list[str]:
        """Return names of locally available Ollama models."""
        client = self._get_client()
        return [m.model for m in client.list().models]

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return f"OllamaProvider({self.model} @ {self.host})"
