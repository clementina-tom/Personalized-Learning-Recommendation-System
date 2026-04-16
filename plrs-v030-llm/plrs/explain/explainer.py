"""
plrs.explain.explainer
=======================
Natural language explanations for PLRS recommendations.

Turns structured recommendation data into human-readable explanations
using any pluggable LLM provider.

Usage:
    from plrs.explain import Explainer
    from plrs.llm import ClaudeProvider, OllamaProvider

    # With Claude
    explainer = Explainer(provider=ClaudeProvider())

    # With a local model (free, no API key)
    explainer = Explainer(provider=OllamaProvider(model="llama3.2"))

    # Explain a single recommendation
    explanation = explainer.explain_recommendation(
        topic_label="Algebraic Factorization",
        mastery=0.42,
        status="approved",
        reasoning="All 2 prerequisite(s) met.",
        prerequisites=["Algebraic Expressions", "Basic Arithmetic"],
        unmet_prerequisites=[],
        downstream_count=4,
        score_breakdown={"gap": 0.23, "readiness": 0.40, "downstream": 0.14, "spaced_rep": 0.12},
        student_mastery={"algebraic_expressions": 0.85, "basic_arithmetic": 0.90},
    )
    print(explanation.text)

    # Explain a full recommendation set
    summary = explainer.explain_results(results, top_n=3)
    print(summary.text)

    # Streaming (for real-time UI)
    for chunk in explainer.stream_explanation(...):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from plrs.llm.base import LLMProvider, resolve_provider


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful and encouraging educational advisor embedded in a learning system.
Your role is to explain learning recommendations to students in clear, motivating language.
Be concise (2-4 sentences max), specific, and positive. Avoid jargon. Write for a secondary school student."""

SINGLE_RECOMMENDATION_PROMPT = """A student learning system has recommended: "{topic_label}"

Context:
- Status: {status} (✅ approved = prerequisites met, ⚠️ challenging = partially met)
- Student's current mastery of this topic: {mastery_pct}%
- Prerequisites needed: {prerequisites}
- Prerequisites not yet mastered: {unmet}
- This topic unlocks {downstream_count} future topic(s) if mastered
- Why recommended: {reasoning}
- Scoring: gap={gap:.2f} (mastery gap), readiness={readiness:.2f} (prereq readiness), downstream={downstream:.2f} (future value), spaced_rep={spaced_rep:.2f} (review urgency)

Write a 2-3 sentence explanation for the student explaining:
1. Why this topic is recommended right now
2. What they need to do / be aware of
3. What they'll unlock by mastering it (if downstream > 0)

Be encouraging and specific. No bullet points. Plain sentences only."""

FULL_RESULTS_PROMPT = """A student's personalized learning recommendations:

Approved (ready to learn now):
{approved_list}

Challenging (can attempt but difficult):
{challenging_list}

Student mastery summary: {mastery_summary}

Write a 3-4 sentence overview for the student:
1. What they should focus on first and why
2. Any challenging topics they might attempt if motivated
3. An encouraging note about their progress

Be specific, warm, and actionable. No bullet points."""

WHAT_IF_PROMPT = """A student is asking: "What happens if I master {topic_label}?"

This topic:
- Directly unlocks: {direct_unlocks}
- Total topics eventually unlocked: {total_unlocked}
- Currently blocked by: {blocked_by}

Write 2-3 sentences explaining what mastering this topic opens up for the student.
Be specific about the unlocked topics. Be encouraging."""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExplanationResult:
    """A generated explanation."""
    text:       str
    topic_id:   str | None
    provider:   str
    cached:     bool = False
    token_hint: int  = 0   # approximate length

    def __str__(self) -> str:
        return self.text


# ── Explainer ─────────────────────────────────────────────────────────────────

class Explainer:
    """
    Generate natural language explanations for PLRS recommendations.

    Provider-agnostic — works with any LLMProvider.
    Includes an in-memory response cache to avoid duplicate API calls.

    Parameters
    ----------
    provider : str or LLMProvider
        LLM provider. String shorthand: "claude", "openai", "ollama", "huggingface".
    max_tokens : int
        Max tokens per explanation (default 200 — explanations should be short).
    temperature : float
        Sampling temperature. 0.3 gives natural variation without randomness.
    cache : bool
        Cache responses by content hash. Same inputs → same explanation without
        extra API calls. Default True.
    """

    def __init__(
        self,
        provider: str | LLMProvider = "claude",
        max_tokens: int = 200,
        temperature: float = 0.3,
        cache: bool = True,
    ) -> None:
        self.provider    = resolve_provider(provider)
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._cache: dict[str, str] = {} if cache else {}
        self._cache_enabled = cache

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def explain_recommendation(
        self,
        topic_label: str,
        mastery: float,
        status: str,
        reasoning: str,
        prerequisites: list[str],
        unmet_prerequisites: list[str],
        downstream_count: int,
        score_breakdown: dict[str, float],
        student_mastery: dict[str, float] | None = None,
        topic_id: str | None = None,
    ) -> ExplanationResult:
        """
        Explain a single recommendation in natural language.

        Parameters
        ----------
        topic_label : str
        mastery : float — current mastery [0, 1]
        status : str — "approved" | "challenging" | "vetoed"
        reasoning : str — constraint layer reasoning string
        prerequisites : list[str] — prerequisite topic labels
        unmet_prerequisites : list[str] — not yet mastered prereq labels
        downstream_count : int — topics this unlocks
        score_breakdown : dict — gap/readiness/downstream/spaced_rep scores
        student_mastery : dict, optional — broader mastery context
        topic_id : str, optional

        Returns
        -------
        ExplanationResult
        """
        prompt = SINGLE_RECOMMENDATION_PROMPT.format(
            topic_label=topic_label,
            status=status,
            mastery_pct=int(mastery * 100),
            prerequisites=", ".join(prerequisites) if prerequisites else "none",
            unmet=", ".join(unmet_prerequisites) if unmet_prerequisites else "none",
            downstream_count=downstream_count,
            reasoning=reasoning,
            gap=score_breakdown.get("gap", 0),
            readiness=score_breakdown.get("readiness", 0),
            downstream=score_breakdown.get("downstream", 0),
            spaced_rep=score_breakdown.get("spaced_rep", 0),
        )

        text = self._complete_cached(prompt, topic_id or topic_label)

        return ExplanationResult(
            text=text,
            topic_id=topic_id,
            provider=self.provider.provider_name,
            cached=self._was_cached,
            token_hint=len(text.split()),
        )

    def explain_results(
        self,
        results: dict[str, Any],
        top_n: int = 3,
    ) -> ExplanationResult:
        """
        Explain a full recommendation result set (from PLRSPipeline.recommend_from_mastery).

        Parameters
        ----------
        results : dict — output from PLRSPipeline.recommend_from_mastery()
        top_n : int — number of approved recommendations to summarise

        Returns
        -------
        ExplanationResult
        """
        approved    = results.get("approved", [])[:top_n]
        challenging = results.get("challenging", [])[:2]
        summary     = results.get("mastery_summary", {})

        approved_list = "\n".join(
            f"- {r['topic_label']} (mastery: {int(r['mastery']*100)}%, "
            f"unlocks {r['downstream_count']} topics)"
            for r in approved
        ) or "None"

        challenging_list = "\n".join(
            f"- {r['topic_label']} — {r['reasoning']}"
            for r in challenging
        ) or "None"

        mastery_str = (
            f"{summary.get('mastered', 0)}/{summary.get('total_topics', 0)} topics mastered "
            f"({int(summary.get('mastery_rate', 0) * 100)}%)"
        )

        prompt = FULL_RESULTS_PROMPT.format(
            approved_list=approved_list,
            challenging_list=challenging_list,
            mastery_summary=mastery_str,
        )

        text = self._complete_cached(prompt, cache_key="full_results")
        return ExplanationResult(
            text=text,
            topic_id=None,
            provider=self.provider.provider_name,
            cached=self._was_cached,
        )

    def explain_what_if(
        self,
        topic_label: str,
        direct_unlocks: list[str],
        total_unlocked: int,
        blocked_by: list[str],
        topic_id: str | None = None,
    ) -> ExplanationResult:
        """
        Explain what mastering a topic would unlock.

        Parameters
        ----------
        topic_label : str
        direct_unlocks : list[str] — labels of directly unlocked topics
        total_unlocked : int — total transitive unlocks
        blocked_by : list[str] — prerequisite labels blocking this topic
        topic_id : str, optional

        Returns
        -------
        ExplanationResult
        """
        prompt = WHAT_IF_PROMPT.format(
            topic_label=topic_label,
            direct_unlocks=", ".join(direct_unlocks[:5]) if direct_unlocks else "nothing directly",
            total_unlocked=total_unlocked,
            blocked_by=", ".join(blocked_by) if blocked_by else "nothing (root topic)",
        )

        text = self._complete_cached(prompt, topic_id or topic_label)
        return ExplanationResult(
            text=text,
            topic_id=topic_id,
            provider=self.provider.provider_name,
            cached=self._was_cached,
        )

    def stream_explanation(
        self,
        topic_label: str,
        mastery: float,
        status: str,
        reasoning: str,
        prerequisites: list[str],
        unmet_prerequisites: list[str],
        downstream_count: int,
        score_breakdown: dict[str, float],
    ) -> Iterator[str]:
        """
        Stream a recommendation explanation token by token.

        Useful for real-time UI updates (Streamlit, WebSocket).
        Falls back to single-chunk yield if provider doesn't support streaming.

        Yields
        ------
        str — text chunks
        """
        prompt = SINGLE_RECOMMENDATION_PROMPT.format(
            topic_label=topic_label,
            status=status,
            mastery_pct=int(mastery * 100),
            prerequisites=", ".join(prerequisites) if prerequisites else "none",
            unmet=", ".join(unmet_prerequisites) if unmet_prerequisites else "none",
            downstream_count=downstream_count,
            reasoning=reasoning,
            gap=score_breakdown.get("gap", 0),
            readiness=score_breakdown.get("readiness", 0),
            downstream=score_breakdown.get("downstream", 0),
            spaced_rep=score_breakdown.get("spaced_rep", 0),
        )

        yield from self.provider.stream(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    _was_cached: bool = False

    def _complete_cached(self, prompt: str, cache_key: str) -> str:
        """Complete with caching by content hash."""
        if self._cache_enabled:
            h = hashlib.sha256(
                (prompt + self.provider.provider_name).encode()
            ).hexdigest()[:16]

            if h in self._cache:
                self._was_cached = True
                return self._cache[h]
        else:
            h = ""

        self._was_cached = False
        text = self.provider.complete(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if self._cache_enabled and h:
            self._cache[h] = text

        return text
