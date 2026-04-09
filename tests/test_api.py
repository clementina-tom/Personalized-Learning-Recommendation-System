"""
tests/test_api.py
=================
Integration tests for the FastAPI endpoints.
Run with: pytest tests/test_api.py -v
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
MAPS = ROOT / "data" / "knowledge_maps"


@pytest.fixture(scope="module")
def client():
    import sys
    from plrs.api.app import app, register_pipeline
    from plrs.curriculum.loader import load_dag
    from plrs.pipeline import PLRSPipeline

    # Enable dev mode so existing tests pass without API keys
    app_module = sys.modules["plrs.api.app"]
    app_module.DEV_MODE = True

    for domain, fname in [("math", "math_dag.json"), ("cs", "cs_dag.json")]:
        path = MAPS / fname
        if path.exists():
            register_pipeline(domain, PLRSPipeline(load_dag(path)))

    return TestClient(app)


# ------------------------------------------------------------------ #
# Health & root                                                        #
# ------------------------------------------------------------------ #

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "math" in data["loaded_domains"]
        assert "cs" in data["loaded_domains"]

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "docs" in r.json()


# ------------------------------------------------------------------ #
# Curriculum endpoint                                                  #
# ------------------------------------------------------------------ #

class TestCurriculum:
    def test_math_curriculum(self, client):
        r = client.get("/curriculum/math")
        assert r.status_code == 200
        data = r.json()
        assert data["num_nodes"] == 38
        assert data["num_edges"] == 45
        assert len(data["nodes"]) == 38

    def test_cs_curriculum(self, client):
        r = client.get("/curriculum/cs")
        assert r.status_code == 200
        data = r.json()
        assert data["num_nodes"] == 31
        assert data["num_edges"] == 39

    def test_unknown_domain_404(self, client):
        r = client.get("/curriculum/physics")
        assert r.status_code == 404

    def test_node_has_required_fields(self, client):
        r = client.get("/curriculum/math")
        node = r.json()["nodes"][0]
        for field in ("id", "label", "level", "prerequisites", "successors"):
            assert field in node, f"Missing field: {field}"


# ------------------------------------------------------------------ #
# Recommend endpoint                                                   #
# ------------------------------------------------------------------ #

class TestRecommend:
    def test_basic_recommendation(self, client):
        r = client.post("/recommend", json={
            "domain": "math",
            "mastery_scores": {"whole_numbers": 0.90, "number_bases": 0.85},
            "top_n": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert "approved" in data
        assert "challenging" in data
        assert "vetoed" in data
        assert "stats" in data
        assert "mastery_summary" in data

    def test_approved_capped_at_top_n(self, client):
        r = client.post("/recommend", json={
            "domain": "math",
            "mastery_scores": {"whole_numbers": 0.95},
            "top_n": 3,
        })
        assert r.status_code == 200
        assert len(r.json()["approved"]) <= 3

    def test_recommendation_fields(self, client):
        r = client.post("/recommend", json={
            "domain": "cs",
            "mastery_scores": {"computer_basics": 0.90},
            "top_n": 5,
        })
        data = r.json()
        if data["approved"]:
            rec = data["approved"][0]
            for field in ("topic_id", "topic_label", "status", "mastery",
                          "score", "reasoning", "score_breakdown"):
                assert field in rec, f"Missing field: {field}"

    def test_empty_mastery_scores(self, client):
        r = client.post("/recommend", json={
            "domain": "math",
            "mastery_scores": {},
            "top_n": 5,
        })
        assert r.status_code == 200
        # With no mastery, root nodes (no prerequisites) should be approved
        data = r.json()
        assert data["stats"]["approved_count"] >= 1

    def test_fully_mastered_student(self, client):
        # Get all node IDs from curriculum
        curriculum_r = client.get("/curriculum/math")
        nodes = {n["id"]: 0.95 for n in curriculum_r.json()["nodes"]}
        r = client.post("/recommend", json={
            "domain": "math",
            "mastery_scores": nodes,
            "top_n": 5,
        })
        assert r.status_code == 200
        data = r.json()
        # All mastered → violation rate should be 0
        assert data["stats"]["prerequisite_violation_rate"] == 0.0

    def test_unknown_domain_404(self, client):
        r = client.post("/recommend", json={
            "domain": "biology",
            "mastery_scores": {},
        })
        assert r.status_code == 404

    def test_threshold_override(self, client):
        r = client.post("/recommend", json={
            "domain": "math",
            "mastery_scores": {"whole_numbers": 0.60},
            "threshold": 0.50,   # lower threshold → 0.60 counts as mastered
            "top_n": 5,
        })
        assert r.status_code == 200


# ------------------------------------------------------------------ #
# What-if endpoint                                                     #
# ------------------------------------------------------------------ #

class TestWhatIf:
    def test_basic_what_if(self, client):
        r = client.post("/what-if", json={
            "domain": "math",
            "topic_id": "whole_numbers",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["topic_id"] == "whole_numbers"
        assert "direct_unlocks" in data
        assert "all_unlocks" in data
        assert "blocked_by" in data
        assert isinstance(data["total_unlocked"], int)
        assert data["total_unlocked"] > 0

    def test_cs_what_if(self, client):
        r = client.post("/what-if", json={
            "domain": "cs",
            "topic_id": "computer_basics",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["total_unlocked"] > 0

    def test_unknown_topic_404(self, client):
        r = client.post("/what-if", json={
            "domain": "math",
            "topic_id": "nonexistent_topic",
        })
        assert r.status_code == 404

    def test_leaf_node_no_unlocks(self, client):
        # Get a node with no successors
        curriculum_r = client.get("/curriculum/math")
        leaf = next(
            n for n in curriculum_r.json()["nodes"]
            if not n["successors"]
        )
        r = client.post("/what-if", json={
            "domain": "math",
            "topic_id": leaf["id"],
        })
        assert r.status_code == 200
        assert r.json()["total_unlocked"] == 0
