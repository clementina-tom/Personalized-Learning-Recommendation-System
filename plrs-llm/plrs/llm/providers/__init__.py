"""
plrs.llm.providers
==================
Built-in LLM provider adapters.

Each provider has optional dependencies — install only what you need:

    pip install plrs[claude]        # anthropic
    pip install plrs[openai]        # openai
    pip install plrs[ollama]        # ollama
    pip install plrs[huggingface]   # huggingface_hub
    pip install plrs[llm]           # all of the above
"""

# Providers are imported lazily inside resolve_provider()
# to avoid ImportError when optional deps aren't installed.
# Only import here for type-checking / explicit use.

__all__ = [
    "ClaudeProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
]
