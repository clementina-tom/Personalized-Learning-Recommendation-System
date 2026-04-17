"""
plrs.llm
=========
Pluggable LLM provider layer for PLRS.

Use a built-in adapter:
    from plrs.llm import ClaudeProvider, OpenAIProvider, OllamaProvider

Or implement your own:
    from plrs.llm import LLMProvider

    class MyProvider(LLMProvider):
        def complete(self, prompt, system=None, max_tokens=1024, temperature=0.0):
            return my_model.generate(prompt)

String resolution (shorthand):
    resolve_provider("claude")      → ClaudeProvider()
    resolve_provider("openai")      → OpenAIProvider()
    resolve_provider("ollama")      → OllamaProvider()
    resolve_provider("huggingface") → HuggingFaceProvider()
    resolve_provider("mock")        → MockProvider()  ← for testing
"""

from plrs.llm.base import LLMProvider, MockProvider, resolve_provider
from plrs.llm.providers.claude import ClaudeProvider
from plrs.llm.providers.openai import OpenAIProvider
from plrs.llm.providers.ollama import OllamaProvider
from plrs.llm.providers.huggingface import HuggingFaceProvider

__all__ = [
    "LLMProvider",
    "MockProvider",
    "resolve_provider",
    "ClaudeProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
]
