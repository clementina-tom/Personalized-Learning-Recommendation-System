"""
plrs.llm.providers.huggingface
===============================
HuggingFace Inference API provider for PLRS.

Supports both:
  - Serverless Inference API (free tier, rate-limited)
  - Dedicated Inference Endpoints (paid, production)

Good open-source models for curriculum tasks:
    mistralai/Mistral-7B-Instruct-v0.3
    meta-llama/Llama-3.2-3B-Instruct
    google/gemma-2-2b-it
    microsoft/Phi-3-mini-4k-instruct
    Qwen/Qwen2.5-7B-Instruct

Install:
    pip install plrs[huggingface]
    # or: pip install huggingface_hub

Usage:
    from plrs.llm.providers.huggingface import HuggingFaceProvider

    # Serverless (free, rate-limited)
    provider = HuggingFaceProvider(model="mistralai/Mistral-7B-Instruct-v0.3")

    # Dedicated endpoint
    provider = HuggingFaceProvider(
        endpoint_url="https://your-endpoint.huggingface.cloud"
    )

    response = provider.complete("What are prerequisites for calculus?")

Environment variable:
    HF_TOKEN — required (get from huggingface.co/settings/tokens)
"""

from __future__ import annotations

import os
from typing import Iterator

from plrs.llm.base import LLMProvider

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


class HuggingFaceProvider(LLMProvider):
    """
    HuggingFace Inference API provider.

    Parameters
    ----------
    model : str
        HuggingFace model ID (default: Mistral-7B-Instruct-v0.3).
    api_key : str, optional
        HuggingFace token. Falls back to HF_TOKEN env var.
    endpoint_url : str, optional
        Dedicated Inference Endpoint URL. If set, model is ignored.
    max_tokens : int
        Default max new tokens per completion.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        endpoint_url: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model        = model
        self.api_key      = api_key or os.getenv("HF_TOKEN")
        self.endpoint_url = endpoint_url
        self.max_tokens   = max_tokens
        self._client      = None

    def _get_client(self):
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
            except ImportError:
                raise ImportError(
                    "huggingface_hub not installed. "
                    "Run: pip install plrs[huggingface]  or  pip install huggingface_hub"
                )
            if not self.api_key:
                raise ValueError(
                    "HuggingFace token not found. "
                    "Set HF_TOKEN environment variable or pass api_key= to HuggingFaceProvider."
                )
            if self.endpoint_url:
                self._client = InferenceClient(
                    base_url=self.endpoint_url,
                    token=self.api_key,
                )
            else:
                self._client = InferenceClient(
                    model=self.model,
                    token=self.api_key,
                )
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
        response = client.chat_completion(
            messages=self._build_messages(prompt, system),
            max_tokens=max_tokens or self.max_tokens,
            temperature=max(temperature, 0.01),  # HF requires > 0
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
        for chunk in client.chat_completion(
            messages=self._build_messages(prompt, system),
            max_tokens=max_tokens or self.max_tokens,
            temperature=max(temperature, 0.01),
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        if self.endpoint_url:
            return f"HuggingFaceProvider(endpoint={self.endpoint_url})"
        return f"HuggingFaceProvider({self.model})"
