"""
tests/test_llm_providers.py
============================
Tests for the pluggable LLM provider abstraction layer.

All tests use MockProvider — no real API calls, no keys needed.
Provider-specific tests (Claude, OpenAI, Ollama, HF) use monkeypatching.
"""

from __future__ import annotations

import pytest


# ── Base interface tests ──────────────────────────────────────────────────────

class TestLLMProviderBase:
    def test_mock_provider_returns_fixed_response(self):
        from plrs.llm.base import MockProvider
        p = MockProvider(response="hello world")
        assert p.complete("anything") == "hello world"

    def test_mock_provider_echoes_prompt_when_no_response(self):
        from plrs.llm.base import MockProvider
        p = MockProvider()
        result = p.complete("what is algebra?")
        assert "[MOCK]" in result
        assert "what is algebra" in result

    def test_mock_provider_stream_yields_complete_response(self):
        from plrs.llm.base import MockProvider
        p = MockProvider(response="streamed response")
        chunks = list(p.stream("prompt"))
        assert "".join(chunks) == "streamed response"

    def test_mock_provider_name(self):
        from plrs.llm.base import MockProvider
        p = MockProvider()
        assert p.provider_name == "MockProvider"

    def test_mock_provider_repr(self):
        from plrs.llm.base import MockProvider
        p = MockProvider()
        assert "MockProvider" in repr(p)

    def test_base_class_is_abstract(self):
        from plrs.llm.base import LLMProvider
        with pytest.raises(TypeError):
            LLMProvider()  # cannot instantiate abstract class

    def test_custom_provider_works(self):
        from plrs.llm.base import LLMProvider

        class EchoProvider(LLMProvider):
            def complete(self, prompt, system=None, max_tokens=1024, temperature=0.0):
                return f"ECHO: {prompt}"

        p = EchoProvider()
        assert p.complete("hello") == "ECHO: hello"
        # Default stream wraps complete
        chunks = list(p.stream("hello"))
        assert "".join(chunks) == "ECHO: hello"

    def test_custom_provider_with_system_prompt(self):
        from plrs.llm.base import LLMProvider

        class SystemAwareProvider(LLMProvider):
            def complete(self, prompt, system=None, max_tokens=1024, temperature=0.0):
                prefix = f"[SYS: {system}] " if system else ""
                return f"{prefix}{prompt}"

        p = SystemAwareProvider()
        result = p.complete("hello", system="You are helpful")
        assert "You are helpful" in result
        assert "hello" in result


# ── resolve_provider tests ────────────────────────────────────────────────────

class TestResolveProvider:
    def test_resolve_mock(self):
        from plrs.llm.base import MockProvider, resolve_provider
        p = resolve_provider("mock")
        assert isinstance(p, MockProvider)

    def test_resolve_none_returns_none(self):
        from plrs.llm.base import resolve_provider
        assert resolve_provider(None) is None

    def test_resolve_instance_returns_as_is(self):
        from plrs.llm.base import MockProvider, resolve_provider
        mock = MockProvider(response="test")
        result = resolve_provider(mock)
        assert result is mock

    def test_resolve_unknown_raises(self):
        from plrs.llm.base import resolve_provider
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            resolve_provider("grok")

    def test_resolve_wrong_type_raises(self):
        from plrs.llm.base import resolve_provider
        with pytest.raises(TypeError):
            resolve_provider(42)

    def test_resolve_case_insensitive(self):
        from plrs.llm.base import MockProvider, resolve_provider
        p = resolve_provider("MOCK")
        assert isinstance(p, MockProvider)

    def test_resolve_claude_string(self):
        """resolve_provider("claude") should return ClaudeProvider — lazy import."""
        from plrs.llm.base import resolve_provider
        from plrs.llm.providers.claude import ClaudeProvider
        p = resolve_provider("claude")
        assert isinstance(p, ClaudeProvider)

    def test_resolve_openai_string(self):
        from plrs.llm.base import resolve_provider
        from plrs.llm.providers.openai import OpenAIProvider
        p = resolve_provider("openai")
        assert isinstance(p, OpenAIProvider)

    def test_resolve_ollama_string(self):
        from plrs.llm.base import resolve_provider
        from plrs.llm.providers.ollama import OllamaProvider
        p = resolve_provider("ollama")
        assert isinstance(p, OllamaProvider)

    def test_resolve_huggingface_string(self):
        from plrs.llm.base import resolve_provider
        from plrs.llm.providers.huggingface import HuggingFaceProvider
        p = resolve_provider("huggingface")
        assert isinstance(p, HuggingFaceProvider)

    def test_resolve_hf_alias(self):
        from plrs.llm.base import resolve_provider
        from plrs.llm.providers.huggingface import HuggingFaceProvider
        p = resolve_provider("hf")
        assert isinstance(p, HuggingFaceProvider)


# ── Provider instantiation tests (no API calls) ───────────────────────────────

class TestClaudeProviderInit:
    def test_default_model(self):
        from plrs.llm.providers.claude import ClaudeProvider, DEFAULT_MODEL
        p = ClaudeProvider()
        assert p.model == DEFAULT_MODEL

    def test_custom_model(self):
        from plrs.llm.providers.claude import ClaudeProvider
        p = ClaudeProvider(model="claude-haiku-4-5-20251001")
        assert p.model == "claude-haiku-4-5-20251001"

    def test_api_key_from_param(self):
        from plrs.llm.providers.claude import ClaudeProvider
        p = ClaudeProvider(api_key="test-key")
        assert p.api_key == "test-key"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        from plrs.llm.providers.claude import ClaudeProvider
        p = ClaudeProvider()
        assert p.api_key == "env-key"

    def test_supports_streaming(self):
        from plrs.llm.providers.claude import ClaudeProvider
        assert ClaudeProvider().supports_streaming is True

    def test_provider_name_includes_model(self):
        from plrs.llm.providers.claude import ClaudeProvider
        p = ClaudeProvider(model="claude-test")
        assert "claude-test" in p.provider_name

    def test_missing_api_key_raises_on_call(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from plrs.llm.providers.claude import ClaudeProvider
        p = ClaudeProvider(api_key=None)
        with pytest.raises((ValueError, ImportError)):
            p.complete("test")

    def test_complete_with_mock_client(self, monkeypatch):
        """Test complete() logic without real API call."""
        from plrs.llm.providers.claude import ClaudeProvider

        class FakeMessage:
            text = "mocked response"

        class FakeResponse:
            content = [FakeMessage()]

        class FakeClient:
            def messages(self): pass
            class messages:
                @staticmethod
                def create(**kwargs):
                    return FakeResponse()

        p = ClaudeProvider(api_key="fake")
        p._client = FakeClient()
        result = p.complete("hello")
        assert result == "mocked response"


class TestOpenAIProviderInit:
    def test_default_model(self):
        from plrs.llm.providers.openai import OpenAIProvider, DEFAULT_MODEL
        p = OpenAIProvider()
        assert p.model == DEFAULT_MODEL

    def test_custom_base_url(self):
        """base_url enables OpenAI-compatible APIs (Groq, Together, Fireworks)."""
        from plrs.llm.providers.openai import OpenAIProvider
        p = OpenAIProvider(base_url="https://api.groq.com/openai/v1")
        assert p.base_url == "https://api.groq.com/openai/v1"

    def test_supports_streaming(self):
        from plrs.llm.providers.openai import OpenAIProvider
        assert OpenAIProvider().supports_streaming is True

    def test_message_building_with_system(self):
        from plrs.llm.providers.openai import OpenAIProvider
        p = OpenAIProvider(api_key="fake")
        messages = p._build_messages("hello", "be helpful")
        assert messages[0] == {"role": "system", "content": "be helpful"}
        assert messages[1] == {"role": "user", "content": "hello"}

    def test_message_building_without_system(self):
        from plrs.llm.providers.openai import OpenAIProvider
        p = OpenAIProvider(api_key="fake")
        messages = p._build_messages("hello", None)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestOllamaProviderInit:
    def test_default_model(self):
        from plrs.llm.providers.ollama import OllamaProvider, DEFAULT_MODEL
        p = OllamaProvider()
        assert p.model == DEFAULT_MODEL

    def test_custom_host(self):
        from plrs.llm.providers.ollama import OllamaProvider
        p = OllamaProvider(host="http://192.168.1.100:11434")
        assert "192.168.1.100" in p.host

    def test_host_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://remote:11434")
        from plrs.llm.providers.ollama import OllamaProvider
        p = OllamaProvider()
        assert "remote" in p.host

    def test_supports_streaming(self):
        from plrs.llm.providers.ollama import OllamaProvider
        assert OllamaProvider().supports_streaming is True

    def test_provider_name_includes_host(self):
        from plrs.llm.providers.ollama import OllamaProvider
        p = OllamaProvider(model="llama3.2", host="http://localhost:11434")
        assert "llama3.2" in p.provider_name
        assert "localhost" in p.provider_name

    def test_missing_package_raises_on_call(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "ollama", None)
        from plrs.llm.providers.ollama import OllamaProvider
        p = OllamaProvider()
        p._client = None  # force re-init
        with pytest.raises((ImportError, TypeError)):
            p._get_client()


class TestHuggingFaceProviderInit:
    def test_default_model(self):
        from plrs.llm.providers.huggingface import HuggingFaceProvider, DEFAULT_MODEL
        p = HuggingFaceProvider()
        assert p.model == DEFAULT_MODEL

    def test_endpoint_url_overrides_model(self):
        from plrs.llm.providers.huggingface import HuggingFaceProvider
        p = HuggingFaceProvider(endpoint_url="https://my-endpoint.huggingface.cloud")
        assert p.endpoint_url == "https://my-endpoint.huggingface.cloud"
        assert "endpoint" in p.provider_name

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf-test-token")
        from plrs.llm.providers.huggingface import HuggingFaceProvider
        p = HuggingFaceProvider()
        assert p.api_key == "hf-test-token"

    def test_supports_streaming(self):
        from plrs.llm.providers.huggingface import HuggingFaceProvider
        assert HuggingFaceProvider().supports_streaming is True


# ── Module-level import test ──────────────────────────────────────────────────

class TestModuleImports:
    def test_llm_module_exports(self):
        from plrs.llm import (
            LLMProvider, MockProvider, resolve_provider,
            ClaudeProvider, OpenAIProvider,
            OllamaProvider, HuggingFaceProvider,
        )
        assert all([
            LLMProvider, MockProvider, resolve_provider,
            ClaudeProvider, OpenAIProvider,
            OllamaProvider, HuggingFaceProvider,
        ])

    def test_all_providers_are_subclasses(self):
        from plrs.llm import (
            LLMProvider, ClaudeProvider, OpenAIProvider,
            OllamaProvider, HuggingFaceProvider, MockProvider,
        )
        for cls in [ClaudeProvider, OpenAIProvider, OllamaProvider,
                    HuggingFaceProvider, MockProvider]:
            assert issubclass(cls, LLMProvider), f"{cls} is not a subclass of LLMProvider"

    def test_mock_provider_complete_is_fast(self):
        """Mock should never make network calls — instant response."""
        import time
        from plrs.llm.base import MockProvider
        p = MockProvider(response="fast")
        t0 = time.time()
        for _ in range(100):
            p.complete("test")
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"MockProvider too slow: {elapsed:.2f}s for 100 calls"
