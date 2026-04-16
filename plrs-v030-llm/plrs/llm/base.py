"""
plrs.llm.base
=============
Abstract base class for LLM providers in PLRS.

All LLM functionality in PLRS (curriculum building, explainability)
routes through this interface. Developers can use a built-in adapter
or implement their own by subclassing LLMProvider.

Built-in adapters:
    ClaudeProvider      — Anthropic API  (pip install plrs[claude])
    OpenAIProvider      — OpenAI API     (pip install plrs[openai])
    OllamaProvider      — Local models   (pip install plrs[ollama])
    HuggingFaceProvider — HF Inference   (pip install plrs[huggingface])

Bring your own:
    class MyAdapter(LLMProvider):
        def complete(self, prompt, system=None):
            return my_llm.generate(prompt)

    pipeline = PLRSPipeline(curriculum, llm=MyAdapter())

Provider resolution (string shorthand):
    PLRSPipeline(curriculum, llm="claude")       → ClaudeProvider()
    PLRSPipeline(curriculum, llm="openai")       → OpenAIProvider()
    PLRSPipeline(curriculum, llm="ollama")       → OllamaProvider()
    PLRSPipeline(curriculum, llm="huggingface")  → HuggingFaceProvider()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this to add support for any LLM backend.
    Only `complete()` is required — `stream()` defaults to a
    non-streaming wrapper around `complete()`.
    """

    # ------------------------------------------------------------------ #
    # Required                                                             #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Parameters
        ----------
        prompt : str
            The user message / prompt.
        system : str, optional
            System prompt / instruction (not all providers support this).
        max_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature. 0.0 = deterministic.

        Returns
        -------
        str — the model's response text.
        """
        ...

    # ------------------------------------------------------------------ #
    # Optional — defaults to non-streaming wrapper                        #
    # ------------------------------------------------------------------ #

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """
        Stream a completion token by token.

        Default implementation calls complete() and yields the full response
        as a single chunk. Override for true streaming support.

        Yields
        ------
        str — text chunks (tokens or larger segments).
        """
        yield self.complete(prompt, system=system, max_tokens=max_tokens, temperature=temperature)

    # ------------------------------------------------------------------ #
    # Metadata                                                            #
    # ------------------------------------------------------------------ #

    @property
    def provider_name(self) -> str:
        """Human-readable provider name."""
        return self.__class__.__name__

    @property
    def supports_streaming(self) -> bool:
        """True if this provider has a real streaming implementation."""
        return False

    def __repr__(self) -> str:
        return f"{self.provider_name}()"


class MockProvider(LLMProvider):
    """
    Deterministic mock provider for testing.

    Returns a fixed response or echoes the prompt.
    No external dependencies or API calls.

    Parameters
    ----------
    response : str, optional
        Fixed response to return for every call.
        If None, echoes "[MOCK] <first 80 chars of prompt>".
    """

    def __init__(self, response: str | None = None) -> None:
        self._response = response

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        if self._response is not None:
            return self._response
        return f"[MOCK] {prompt[:80]}"

    @property
    def provider_name(self) -> str:
        return "MockProvider"


def resolve_provider(llm: "str | LLMProvider | None") -> "LLMProvider | None":
    """
    Resolve a provider from a string shorthand or return as-is.

    Parameters
    ----------
    llm : str, LLMProvider, or None
        "claude"      → ClaudeProvider()
        "openai"      → OpenAIProvider()
        "ollama"      → OllamaProvider()
        "huggingface" → HuggingFaceProvider()
        LLMProvider   → returned as-is
        None          → None (LLM features disabled)

    Returns
    -------
    LLMProvider or None
    """
    if llm is None:
        return None

    if isinstance(llm, LLMProvider):
        return llm

    if not isinstance(llm, str):
        raise TypeError(
            f"llm must be a string, LLMProvider instance, or None. Got: {type(llm)}"
        )

    key = llm.lower().strip()

    if key == "claude":
        from plrs.llm.providers.claude import ClaudeProvider
        return ClaudeProvider()

    if key == "openai":
        from plrs.llm.providers.openai import OpenAIProvider
        return OpenAIProvider()

    if key == "ollama":
        from plrs.llm.providers.ollama import OllamaProvider
        return OllamaProvider()

    if key in ("huggingface", "hf"):
        from plrs.llm.providers.huggingface import HuggingFaceProvider
        return HuggingFaceProvider()

    if key == "mock":
        return MockProvider()

    raise ValueError(
        f"Unknown LLM provider: '{llm}'. "
        f"Valid options: 'claude', 'openai', 'ollama', 'huggingface', 'mock'. "
        f"Or pass a LLMProvider instance directly."
    )
