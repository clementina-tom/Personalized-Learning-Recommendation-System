"""
tests/test_llm_router.py
=========================
Tests for /explain and /curriculum/generate endpoints.
All tests use MockProvider — no real API calls.
"""

from __future__ import annotations

import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plrs.llm.base import MockProvider


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Minimal FastAPI app with just the LLM router mounted."""
    from plrs.api.llm_router import llm_router, register_default_llm_provider

    # Register a MockProvider as the default so no API keys are needed
    register_default_llm_provider(MockProvider(response="This is a test explanation."))

    app = FastAPI()
    app.include_router(llm_router)
    return TestClient(app)


@pytest.fixture
def sample_rec():
    return {
        "topic_label": "Algebraic Factorization",
        "mastery": 0.42,
        "status": "approved",
        "reasoning": "All 2 prerequisite(s) met.",
        "prerequisites": ["Algebraic Expressions", "Basic Arithmetic"],
        "unmet_prerequisites": [],
        "downstream_count": 4,
        "score_breakdown": {"gap": 0.23, "readiness": 0.40, "downstream": 0.14, "spaced_rep": 0.12},
        "topic_id": "algebraic_factorization",
    }


@pytest.fixture
def sample_results():
    return {
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
        "stats": {"approved_count": 1, "vetoed_count": 0},
    }


# ── /curriculum/providers ────────────────────────────────────────────────────

class TestProvidersEndpoint:
    def test_returns_200(self, client):
        r = client.get("/curriculum/providers")
        assert r.status_code == 200

    def test_lists_four_providers(self, client):
        r = client.get("/curriculum/providers")
        data = r.json()
        assert "providers" in data
        assert len(data["providers"]) == 4

    def test_provider_names(self, client):
        r = client.get("/curriculum/providers")
        names = {p["name"] for p in r.json()["providers"]}
        assert names == {"claude", "openai", "ollama", "huggingface"}

    def test_ollama_is_free(self, client):
        r = client.get("/curriculum/providers")
        ollama = next(p for p in r.json()["providers"] if p["name"] == "ollama")
        assert ollama["free"] is True

    def test_has_note(self, client):
        r = client.get("/curriculum/providers")
        assert "note" in r.json()


# ── /explain ─────────────────────────────────────────────────────────────────

class TestExplainEndpoint:
    def test_returns_200(self, client, sample_rec):
        r = client.post("/explain", json=sample_rec)
        assert r.status_code == 200

    def test_response_has_text(self, client, sample_rec):
        r = client.post("/explain", json=sample_rec)
        data = r.json()
        assert "text" in data
        assert len(data["text"]) > 0

    def test_response_has_provider(self, client, sample_rec):
        r = client.post("/explain", json=sample_rec)
        assert "provider" in r.json()

    def test_response_has_topic_id(self, client, sample_rec):
        r = client.post("/explain", json=sample_rec)
        assert r.json()["topic_id"] == "algebraic_factorization"

    def test_response_has_cached_field(self, client, sample_rec):
        r = client.post("/explain", json=sample_rec)
        assert "cached" in r.json()

    def test_mock_text_in_response(self, client, sample_rec):
        r = client.post("/explain", json=sample_rec)
        assert r.json()["text"] == "This is a test explanation."

    def test_without_topic_id(self, client, sample_rec):
        payload = {**sample_rec}
        del payload["topic_id"]
        r = client.post("/explain", json=payload)
        assert r.status_code == 200

    def test_mastery_validation(self, client, sample_rec):
        payload = {**sample_rec, "mastery": 1.5}  # > 1.0
        r = client.post("/explain", json=payload)
        assert r.status_code == 422  # Pydantic validation error

    def test_missing_required_field(self, client):
        r = client.post("/explain", json={"mastery": 0.5})
        assert r.status_code == 422

    def test_empty_prerequisites(self, client, sample_rec):
        payload = {**sample_rec, "prerequisites": [], "unmet_prerequisites": []}
        r = client.post("/explain", json=payload)
        assert r.status_code == 200

    def test_invalid_provider_returns_400(self, client, sample_rec):
        payload = {**sample_rec, "provider": "nonexistent_llm"}
        r = client.post("/explain", json=payload)
        assert r.status_code == 400
        assert "Invalid LLM provider" in r.json()["detail"]

    def test_mock_provider_override(self, client, sample_rec):
        """Explicit 'mock' provider should work."""
        payload = {**sample_rec, "provider": "mock"}
        r = client.post("/explain", json=payload)
        assert r.status_code == 200


# ── /explain/results ──────────────────────────────────────────────────────────

class TestExplainResultsEndpoint:
    def test_returns_200(self, client, sample_results):
        r = client.post("/explain/results", json={"results": sample_results})
        assert r.status_code == 200

    def test_response_has_text(self, client, sample_results):
        r = client.post("/explain/results", json={"results": sample_results})
        assert len(r.json()["text"]) > 0

    def test_topic_id_is_null(self, client, sample_results):
        r = client.post("/explain/results", json={"results": sample_results})
        assert r.json()["topic_id"] is None

    def test_top_n_parameter(self, client, sample_results):
        r = client.post("/explain/results", json={"results": sample_results, "top_n": 2})
        assert r.status_code == 200

    def test_top_n_too_large_returns_422(self, client, sample_results):
        r = client.post("/explain/results", json={"results": sample_results, "top_n": 100})
        assert r.status_code == 422


# ── /explain/what-if ─────────────────────────────────────────────────────────

class TestExplainWhatIfEndpoint:
    def _payload(self, **overrides):
        base = {
            "topic_label": "Algebra",
            "direct_unlocks": ["Quadratic Equations", "Sequences"],
            "total_unlocked": 5,
            "blocked_by": [],
        }
        base.update(overrides)
        return base

    def test_returns_200(self, client):
        r = client.post("/explain/what-if", json=self._payload())
        assert r.status_code == 200

    def test_response_text(self, client):
        r = client.post("/explain/what-if", json=self._payload())
        assert r.json()["text"] == "This is a test explanation."

    def test_with_topic_id(self, client):
        r = client.post("/explain/what-if", json=self._payload(topic_id="algebra"))
        assert r.json()["topic_id"] == "algebra"

    def test_no_unlocks(self, client):
        r = client.post("/explain/what-if", json=self._payload(
            direct_unlocks=[], total_unlocked=0
        ))
        assert r.status_code == 200

    def test_missing_topic_label_returns_422(self, client):
        r = client.post("/explain/what-if", json={"total_unlocked": 5})
        assert r.status_code == 422


# ── /curriculum/generate ─────────────────────────────────────────────────────

VALID_DAG_RESPONSE = json.dumps({
    "domain": "Test Physics",
    "nodes": [
        {"id": "mechanics", "label": "Mechanics", "level": "Year 1"},
        {"id": "kinematics", "label": "Kinematics", "level": "Year 1"},
        {"id": "dynamics", "label": "Dynamics", "level": "Year 2"},
    ],
    "edges": [
        {"from": "mechanics", "to": "kinematics"},
        {"from": "kinematics", "to": "dynamics"},
    ],
})


@pytest.fixture(scope="module")
def dag_client():
    """Client with MockProvider that returns a valid DAG JSON."""
    from plrs.api.llm_router import llm_router, register_default_llm_provider

    register_default_llm_provider(MockProvider(response=VALID_DAG_RESPONSE))

    app = FastAPI()
    app.include_router(llm_router)
    return TestClient(app)


class TestCurriculumGenerateEndpoint:
    def test_from_topics_returns_200(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test Physics",
            "topics": ["Mechanics", "Kinematics", "Dynamics"],
        })
        assert r.status_code == 200

    def test_from_topics_correct_structure(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test Physics",
            "topics": ["Mechanics", "Kinematics", "Dynamics"],
        })
        data = r.json()
        assert "dag" in data
        assert "num_nodes" in data
        assert "num_edges" in data
        assert "warnings" in data
        assert "provider" in data

    def test_from_topics_correct_counts(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test Physics",
            "topics": ["Mechanics", "Kinematics", "Dynamics"],
        })
        data = r.json()
        assert data["num_nodes"] == 3
        assert data["num_edges"] == 2

    def test_from_text_content(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test Physics",
            "text_content": "Unit 1: Mechanics. Unit 2: Kinematics builds on Unit 1.",
        })
        assert r.status_code == 200

    def test_both_topics_and_text_returns_400(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test",
            "topics": ["A", "B"],
            "text_content": "Some text",
        })
        assert r.status_code == 400

    def test_neither_topics_nor_text_returns_400(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test",
        })
        assert r.status_code == 400

    def test_dag_has_nodes_and_edges(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test Physics",
            "topics": ["Mechanics", "Kinematics"],
        })
        dag = r.json()["dag"]
        assert "nodes" in dag
        assert "edges" in dag

    def test_domain_in_response(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test Physics",
            "topics": ["A", "B"],
        })
        assert r.json()["domain"] == "Test Physics"

    def test_warnings_list_present(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test",
            "topics": ["A"],
        })
        assert isinstance(r.json()["warnings"], list)

    def test_invalid_provider_returns_400(self, dag_client):
        r = dag_client.post("/curriculum/generate", json={
            "domain": "Test",
            "topics": ["A"],
            "provider": "bad_provider",
        })
        assert r.status_code == 400
