"""
tests/test_llm_features.py
===========================
Tests for CurriculumBuilder and Explainer.
All tests use MockProvider — no real API calls or keys needed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from plrs.llm.base import MockProvider


# ── CurriculumBuilder tests ───────────────────────────────────────────────────

class TestCurriculumBuilder:

    def _builder(self, response: str | None = None):
        from plrs.curriculum.llm_builder import CurriculumBuilder
        return CurriculumBuilder(provider=MockProvider(response=response), validate=False)

    def _valid_dag_response(self, domain="Test Math") -> str:
        dag = {
            "domain": domain,
            "nodes": [
                {"id": "whole_numbers", "label": "Whole Numbers", "level": "JSS1"},
                {"id": "fractions", "label": "Fractions", "level": "JSS1"},
                {"id": "algebra", "label": "Algebra", "level": "JSS2"},
            ],
            "edges": [
                {"from": "whole_numbers", "to": "fractions"},
                {"from": "fractions", "to": "algebra"},
            ],
        }
        return json.dumps(dag)

    # from_topics
    def test_from_topics_returns_build_result(self):
        from plrs.curriculum.llm_builder import BuildResult
        b = self._builder(self._valid_dag_response())
        result = b.from_topics(["Whole Numbers", "Fractions", "Algebra"], domain="Test Math")
        assert isinstance(result, BuildResult)

    def test_from_topics_correct_node_count(self):
        b = self._builder(self._valid_dag_response())
        result = b.from_topics(["Whole Numbers", "Fractions", "Algebra"])
        assert result.num_nodes == 3

    def test_from_topics_correct_edge_count(self):
        b = self._builder(self._valid_dag_response())
        result = b.from_topics(["Whole Numbers", "Fractions", "Algebra"])
        assert result.num_edges == 2

    def test_from_topics_empty_raises(self):
        b = self._builder()
        with pytest.raises(ValueError, match="empty"):
            b.from_topics([])

    def test_from_topics_domain_preserved(self):
        b = self._builder(self._valid_dag_response("My Domain"))
        result = b.from_topics(["A", "B"], domain="My Domain")
        assert result.domain == "My Domain"

    def test_from_topics_provider_name_set(self):
        b = self._builder(self._valid_dag_response())
        result = b.from_topics(["A", "B"])
        assert "MockProvider" in result.provider

    # from_text
    def test_from_text_returns_build_result(self):
        b = self._builder(self._valid_dag_response())
        result = b.from_text("Unit 1: Whole Numbers. Unit 2: Fractions.")
        assert result.num_nodes == 3

    def test_from_text_empty_raises(self):
        b = self._builder()
        with pytest.raises(ValueError, match="empty"):
            b.from_text("   ")

    def test_from_text_long_text_truncated(self):
        """Builder should handle very long text without crashing."""
        b = self._builder(self._valid_dag_response())
        long_text = "Topic " * 5000  # ~30k chars
        result = b.from_text(long_text, domain="Test")
        assert result is not None

    # JSON parsing
    def test_handles_json_with_markdown_fences(self):
        """LLMs sometimes wrap JSON in ```json ... ```"""
        fenced = "```json\n" + self._valid_dag_response() + "\n```"
        b = self._builder(fenced)
        result = b.from_topics(["A", "B"])
        assert result.num_nodes == 3

    def test_handles_json_with_preamble(self):
        """LLMs sometimes add text before the JSON."""
        preambled = "Here is the curriculum:\n\n" + self._valid_dag_response()
        b = self._builder(preambled)
        result = b.from_topics(["A", "B"])
        assert result.num_nodes == 3

    def test_handles_invalid_json_gracefully(self):
        """Invalid JSON should return empty DAG with warnings."""
        b = self._builder("This is not JSON at all!")
        result = b.from_topics(["A", "B"])
        assert result.num_nodes == 0
        assert len(result.warnings) > 0

    def test_handles_missing_nodes_key(self):
        b = self._builder('{"edges": [], "domain": "Test"}')
        result = b.from_topics(["A"])
        assert result.num_nodes == 0
        assert any("nodes" in w for w in result.warnings)

    # ID sanitisation
    def test_sanitises_node_ids(self):
        """IDs with spaces/caps should be cleaned."""
        dag = {
            "domain": "Test",
            "nodes": [
                {"id": "Whole Numbers", "label": "Whole Numbers"},
                {"id": "Algebraic Expressions!", "label": "Algebraic Expressions"},
            ],
            "edges": [{"from": "Whole Numbers", "to": "Algebraic Expressions!"}],
        }
        b = self._builder(json.dumps(dag))
        result = b.from_topics(["A"])
        node_ids = {n["id"] for n in result.dag["nodes"]}
        assert all(" " not in nid and "!" not in nid for nid in node_ids)

    def test_removes_orphan_edges(self):
        """Edges referencing non-existent nodes should be removed."""
        dag = {
            "domain": "Test",
            "nodes": [{"id": "algebra", "label": "Algebra"}],
            "edges": [{"from": "nonexistent", "to": "algebra"}],
        }
        b = self._builder(json.dumps(dag))
        result = b.from_topics(["A"])
        assert result.num_edges == 0
        assert any("not in nodes" in w for w in result.warnings)

    def test_removes_self_loops(self):
        dag = {
            "domain": "Test",
            "nodes": [{"id": "algebra", "label": "Algebra"}],
            "edges": [{"from": "algebra", "to": "algebra"}],
        }
        b = self._builder(json.dumps(dag))
        result = b.from_topics(["A"])
        assert result.num_edges == 0

    def test_removes_cycles(self):
        """A→B→C→A cycle should be broken."""
        dag = {
            "domain": "Test",
            "nodes": [
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {"id": "c", "label": "C"},
            ],
            "edges": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"},
                {"from": "c", "to": "a"},  # creates cycle
            ],
        }
        b = self._builder(json.dumps(dag))
        result = b.from_topics(["A", "B", "C"])
        assert result.num_edges == 2  # one cycle-breaking edge removed
        assert any("Cycle" in w for w in result.warnings)

    # Save
    def test_save_writes_valid_json(self, tmp_path):
        b = self._builder(self._valid_dag_response())
        result = b.from_topics(["A", "B", "C"])
        path = tmp_path / "curriculum.json"
        b.save(result, path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert "nodes" in loaded
        assert "edges" in loaded

    def test_result_to_dict(self):
        b = self._builder(self._valid_dag_response())
        result = b.from_topics(["A", "B"])
        d = result.to_dict()
        assert "nodes" in d
        assert "edges" in d

    def test_sanitise_id_static(self):
        from plrs.curriculum.llm_builder import CurriculumBuilder
        assert CurriculumBuilder._sanitise_id("Whole Numbers") == "whole_numbers"
        assert CurriculumBuilder._sanitise_id("Algebra!!! 101") == "algebra_101"
        assert CurriculumBuilder._sanitise_id("") == "unknown_topic"
        assert CurriculumBuilder._sanitise_id("  spaces  ") == "spaces"


# ── Explainer tests ───────────────────────────────────────────────────────────

class TestExplainer:

    def _explainer(self, response="This topic is recommended because..."):
        from plrs.explain import Explainer
        return Explainer(provider=MockProvider(response=response), cache=True)

    def _sample_rec_kwargs(self, **overrides):
        defaults = dict(
            topic_label="Algebraic Factorization",
            mastery=0.42,
            status="approved",
            reasoning="All 2 prerequisite(s) met.",
            prerequisites=["Algebraic Expressions", "Basic Arithmetic"],
            unmet_prerequisites=[],
            downstream_count=4,
            score_breakdown={"gap": 0.23, "readiness": 0.40, "downstream": 0.14, "spaced_rep": 0.12},
        )
        defaults.update(overrides)
        return defaults

    def test_explain_recommendation_returns_result(self):
        from plrs.explain import ExplanationResult
        e = self._explainer()
        result = e.explain_recommendation(**self._sample_rec_kwargs())
        assert isinstance(result, ExplanationResult)

    def test_explain_recommendation_text_not_empty(self):
        e = self._explainer("Great explanation here.")
        result = e.explain_recommendation(**self._sample_rec_kwargs())
        assert result.text == "Great explanation here."

    def test_explain_recommendation_provider_name(self):
        e = self._explainer()
        result = e.explain_recommendation(**self._sample_rec_kwargs())
        assert "MockProvider" in result.provider

    def test_explain_recommendation_with_topic_id(self):
        e = self._explainer()
        result = e.explain_recommendation(
            **self._sample_rec_kwargs(), topic_id="algebraic_factorization"
        )
        assert result.topic_id == "algebraic_factorization"

    def test_caching_same_input_no_second_call(self):
        """Same inputs should hit cache, not call provider again."""
        call_count = [0]

        class CountingProvider(MockProvider):
            def complete(self, *args, **kwargs):
                call_count[0] += 1
                return "response"

        from plrs.explain import Explainer
        e = Explainer(provider=CountingProvider(), cache=True)

        kwargs = self._sample_rec_kwargs()
        e.explain_recommendation(**kwargs)
        e.explain_recommendation(**kwargs)

        assert call_count[0] == 1  # second call used cache

    def test_caching_disabled(self):
        """With cache=False, every call should hit provider."""
        call_count = [0]

        class CountingProvider(MockProvider):
            def complete(self, *args, **kwargs):
                call_count[0] += 1
                return "response"

        from plrs.explain import Explainer
        e = Explainer(provider=CountingProvider(), cache=False)

        kwargs = self._sample_rec_kwargs()
        e.explain_recommendation(**kwargs)
        e.explain_recommendation(**kwargs)

        assert call_count[0] == 2

    def test_clear_cache(self):
        e = self._explainer()
        e.explain_recommendation(**self._sample_rec_kwargs())
        assert e.cache_size > 0
        e.clear_cache()
        assert e.cache_size == 0

    def test_explain_results_returns_result(self):
        from plrs.explain import ExplanationResult
        e = self._explainer("Overview explanation.")
        results = {
            "approved": [
                {
                    "topic_id": "fractions", "topic_label": "Fractions",
                    "mastery": 0.3, "downstream_count": 3,
                    "reasoning": "Prerequisites met.", "score": 0.8,
                    "score_breakdown": {}, "prerequisites": [],
                    "unmet_prerequisites": [], "status": "approved",
                }
            ],
            "challenging": [],
            "vetoed": [],
            "mastery_summary": {"mastered": 5, "total_topics": 38, "mastery_rate": 0.13},
            "stats": {},
        }
        result = e.explain_results(results)
        assert isinstance(result, ExplanationResult)
        assert result.text == "Overview explanation."

    def test_explain_what_if_returns_result(self):
        e = self._explainer("Mastering this unlocks...")
        result = e.explain_what_if(
            topic_label="Algebra",
            direct_unlocks=["Quadratic Equations", "Sequences"],
            total_unlocked=5,
            blocked_by=[],
        )
        assert result.text == "Mastering this unlocks..."
        assert result.topic_id is None

    def test_stream_explanation_yields_chunks(self):
        e = self._explainer("chunk1 chunk2 chunk3")
        chunks = list(e.stream_explanation(**self._sample_rec_kwargs()))
        assert len(chunks) > 0
        assert "".join(chunks) == "chunk1 chunk2 chunk3"

    def test_explain_challenging_status(self):
        e = self._explainer("Challenging explanation.")
        result = e.explain_recommendation(
            **self._sample_rec_kwargs(
                status="challenging",
                unmet_prerequisites=["Trigonometry"],
            )
        )
        assert result.text == "Challenging explanation."

    def test_explain_no_prerequisites(self):
        e = self._explainer("Root topic explanation.")
        result = e.explain_recommendation(
            **self._sample_rec_kwargs(
                prerequisites=[],
                unmet_prerequisites=[],
            )
        )
        assert result is not None

    def test_explain_no_downstream(self):
        e = self._explainer("Leaf topic explanation.")
        result = e.explain_recommendation(
            **self._sample_rec_kwargs(downstream_count=0)
        )
        assert result is not None

    def test_str_conversion(self):
        e = self._explainer("Plain text output.")
        result = e.explain_recommendation(**self._sample_rec_kwargs())
        assert str(result) == "Plain text output."
