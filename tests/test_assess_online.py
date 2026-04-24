"""
tests/test_assess_online.py
============================
Tests for cold start diagnostic assessment and online learning.
Uses a synthetic curriculum — no external dependencies.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ── Synthetic curriculum fixture ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_curriculum():
    """Minimal synthetic curriculum: A → B → C → D, plus E (root)."""
    nodes = ["topic_a", "topic_b", "topic_c", "topic_d", "topic_e"]
    edges = [
        ("topic_a", "topic_b"),
        ("topic_b", "topic_c"),
        ("topic_c", "topic_d"),
    ]
    labels = {
        "topic_a": "Topic A", "topic_b": "Topic B",
        "topic_c": "Topic C", "topic_d": "Topic D",
        "topic_e": "Topic E (root)",
    }
    levels = {n: "JSS1" for n in nodes}

    prereqs = {
        "topic_a": [], "topic_b": ["topic_a"],
        "topic_c": ["topic_b"], "topic_d": ["topic_c"],
        "topic_e": [],
    }
    successors = {
        "topic_a": ["topic_b"], "topic_b": ["topic_c"],
        "topic_c": ["topic_d"], "topic_d": [],
        "topic_e": [],
    }

    m = MagicMock()
    m.nodes = nodes
    m.label = lambda n: labels.get(n, n)
    m.level = lambda n: levels.get(n, "")
    m.prerequisites = lambda n: prereqs.get(n, [])
    m.successors = lambda n: successors.get(n, [])
    m.descendants = lambda n: []
    return m


@pytest.fixture(scope="module")
def mock_pipeline(mock_curriculum):
    """Mock pipeline wrapping the synthetic curriculum."""
    p = MagicMock()
    p.curriculum = mock_curriculum
    p._model = None
    p.threshold = 0.70
    p.soft_threshold = 0.50
    p.top_n = 5
    p.recommend_from_mastery = lambda mastery: {
        "approved": [], "challenging": [], "vetoed": [],
        "stats": {"approved_count": 0, "vetoed_count": 0, "prerequisite_violation_rate": 0.0,
                  "spaced_rep_enabled": False},
        "mastery_summary": {"mastered": 0, "total_topics": 5, "mastery_rate": 0.0},
    }
    return p


# ── DiagnosticEngine tests ────────────────────────────────────────────────────

class TestDiagnosticEngine:
    @pytest.fixture
    def engine(self, mock_curriculum):
        from plrs.assess.diagnostic import DiagnosticEngine
        return DiagnosticEngine(mock_curriculum)

    def test_question_bank_built(self, engine):
        assert len(engine.question_bank) == 5

    def test_each_topic_has_question(self, engine, mock_curriculum):
        for node in mock_curriculum.nodes:
            assert node in engine.question_bank

    def test_start_session_creates_session(self, engine):
        from plrs.assess.diagnostic import AssessmentSession
        session = engine.start_session(domain="test")
        assert isinstance(session, AssessmentSession)
        assert session.domain == "test"
        assert session.num_answered == 0

    def test_next_question_returns_question(self, engine):
        session = engine.start_session(domain="test")
        q = engine.next_question(session)
        assert q is not None

    def test_next_question_prefers_roots(self, engine):
        """First question should be a root topic (no prerequisites)."""
        session = engine.start_session(domain="test")
        q = engine.next_question(session)
        assert q.topic_id in ["topic_a", "topic_e"]

    def test_record_correct_increases_mastery(self, engine):
        session = engine.start_session(domain="test")
        q = engine.next_question(session)
        mastery_before = session.mastery_estimate[q.topic_id]
        engine.record_answer(session, q, chosen_idx=q.correct_idx)
        assert session.mastery_estimate[q.topic_id] > mastery_before

    def test_record_incorrect_decreases_mastery(self, engine):
        session = engine.start_session(domain="test")
        q = engine.next_question(session)
        mastery_before = session.mastery_estimate[q.topic_id]
        wrong_idx = (q.correct_idx + 1) % 4
        engine.record_answer(session, q, chosen_idx=wrong_idx)
        assert session.mastery_estimate[q.topic_id] < mastery_before

    def test_session_tracks_answered_count(self, engine):
        session = engine.start_session(domain="test", max_questions=5)
        q = engine.next_question(session)
        engine.record_answer(session, q, chosen_idx=0)
        assert session.num_answered == 1

    def test_session_complete_after_max_questions(self, engine):
        session = engine.start_session(domain="test", max_questions=3)
        for _ in range(3):
            q = engine.next_question(session)
            if q:
                engine.record_answer(session, q, chosen_idx=0)
        assert session.is_complete

    def test_finalise_returns_mastery_dict(self, engine):
        session = engine.start_session(domain="test", max_questions=3)
        for _ in range(3):
            q = engine.next_question(session)
            if q:
                engine.record_answer(session, q, chosen_idx=0)
        mastery = engine.finalise(session)
        assert isinstance(mastery, dict)
        assert len(mastery) == 5  # all topics covered

    def test_finalise_mastery_in_range(self, engine):
        session = engine.start_session(domain="test", max_questions=5)
        for _ in range(5):
            q = engine.next_question(session)
            if q:
                engine.record_answer(session, q, chosen_idx=q.correct_idx)
        mastery = engine.finalise(session)
        assert all(0.0 <= v <= 1.0 for v in mastery.values())

    def test_finalise_marks_session_complete(self, engine):
        session = engine.start_session(domain="test")
        engine.finalise(session)
        assert session.completed is True

    def test_no_duplicate_questions(self, engine):
        session = engine.start_session(domain="test", max_questions=5)
        asked = []
        for _ in range(5):
            q = engine.next_question(session)
            if q:
                assert q.topic_id not in asked
                asked.append(q.topic_id)
                engine.record_answer(session, q, chosen_idx=0)

    def test_session_to_dict(self, engine):
        session = engine.start_session(domain="test")
        d = session.to_dict()
        assert "session_id" in d
        assert "num_answered" in d
        assert "completed" in d


# ── SessionStore tests ────────────────────────────────────────────────────────

class TestSessionStore:
    def test_save_and_get(self):
        from plrs.assess.diagnostic import SessionStore, AssessmentSession
        store = SessionStore()
        session = AssessmentSession(session_id="test-id", domain="math", questions=[])
        store.save(session)
        retrieved = store.get("test-id")
        assert retrieved is not None
        assert retrieved.session_id == "test-id"

    def test_get_nonexistent_returns_none(self):
        from plrs.assess.diagnostic import SessionStore
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        from plrs.assess.diagnostic import SessionStore, AssessmentSession
        store = SessionStore()
        session = AssessmentSession(session_id="del-me", domain="math", questions=[])
        store.save(session)
        store.delete("del-me")
        assert store.get("del-me") is None

    def test_len(self):
        from plrs.assess.diagnostic import SessionStore, AssessmentSession
        store = SessionStore()
        for i in range(3):
            s = AssessmentSession(session_id=f"id-{i}", domain="math", questions=[])
            store.save(s)
        assert len(store) == 3


# ── Assessment router tests ───────────────────────────────────────────────────

class TestAssessRouter:
    @pytest.fixture
    def client(self, mock_curriculum):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from plrs.assess.router import assess_router
        from plrs.assess.diagnostic import DiagnosticEngine, register_engine

        engine = DiagnosticEngine(mock_curriculum)
        register_engine("test", engine)

        app = FastAPI()
        app.include_router(assess_router)
        return TestClient(app)

    def test_start_returns_session_id(self, client):
        r = client.post("/assess/start", json={"domain": "test", "max_questions": 5})
        assert r.status_code == 200
        assert "session_id" in r.json()

    def test_start_returns_first_question(self, client):
        r = client.post("/assess/start", json={"domain": "test", "max_questions": 5})
        assert "first_question" in r.json()
        assert "text" in r.json()["first_question"]

    def test_start_unknown_domain_404(self, client):
        r = client.post("/assess/start", json={"domain": "nonexistent"})
        assert r.status_code == 404

    def test_answer_correct(self, client):
        start = client.post("/assess/start", json={"domain": "test", "max_questions": 5})
        sid   = start.json()["session_id"]
        q     = start.json()["first_question"]

        r = client.post(f"/assess/{sid}/answer", json={
            "question_id": q["question_id"],
            "chosen_idx":  0,  # correct answer is always index 0 in synthetic bank
        })
        assert r.status_code == 200
        data = r.json()
        assert "correct" in data
        assert "mastery_after" in data

    def test_get_session(self, client):
        start = client.post("/assess/start", json={"domain": "test", "max_questions": 5})
        sid   = start.json()["session_id"]
        r     = client.get(f"/assess/{sid}")
        assert r.status_code == 200
        assert r.json()["session_id"] == sid

    def test_complete_returns_mastery(self, client):
        start = client.post("/assess/start", json={"domain": "test", "max_questions": 5})
        sid   = start.json()["session_id"]

        # Answer a couple questions
        q = start.json()["first_question"]
        client.post(f"/assess/{sid}/answer", json={"question_id": q["question_id"], "chosen_idx": 0})

        r = client.post(f"/assess/{sid}/complete")
        assert r.status_code == 200
        data = r.json()
        assert "mastery" in data
        assert isinstance(data["mastery"], dict)
        assert len(data["mastery"]) > 0

    def test_get_nonexistent_session_404(self, client):
        r = client.get("/assess/nonexistent-session-id")
        assert r.status_code == 404


# ── OnlineLearner tests ───────────────────────────────────────────────────────

class TestOnlineLearner:
    @pytest.fixture
    def learner(self, mock_pipeline):
        from plrs.model.online import OnlineLearner
        return OnlineLearner(mock_pipeline, mode="bkt")

    def test_load_student(self, learner):
        state = learner.load_student(
            "s1", initial_mastery={"topic_a": 0.5, "topic_b": 0.3}, domain="test"
        )
        assert state.student_id == "s1"
        assert state.mastery["topic_a"] == 0.5

    def test_get_student_after_load(self, learner):
        learner.load_student("s2", initial_mastery={"topic_a": 0.5})
        state = learner.get_student("s2")
        assert state is not None

    def test_get_nonexistent_student_returns_none(self, learner):
        assert learner.get_student("nobody") is None

    def test_update_correct_increases_mastery(self, learner):
        learner.load_student("s3", initial_mastery={"topic_a": 0.5}, domain="test")
        result = learner.update("s3", topic_id="topic_a", correct=True)
        assert result.mastery_after > 0.5
        assert result.delta > 0

    def test_update_incorrect_decreases_mastery(self, learner):
        learner.load_student("s4", initial_mastery={"topic_a": 0.7}, domain="test")
        result = learner.update("s4", topic_id="topic_a", correct=False)
        assert result.mastery_after < 0.7
        assert result.delta < 0

    def test_update_records_interaction(self, learner):
        learner.load_student("s5", initial_mastery={"topic_a": 0.5})
        learner.update("s5", topic_id="topic_a", correct=True)
        learner.update("s5", topic_id="topic_a", correct=False)
        assert learner.get_student("s5").total_interactions == 2

    def test_mastery_clamped(self, learner):
        learner.load_student("s6", initial_mastery={"topic_a": 0.99})
        for _ in range(20):
            result = learner.update("s6", topic_id="topic_a", correct=True)
        assert result.mastery_after <= 0.95

    def test_update_result_has_topic_label(self, learner):
        learner.load_student("s7", initial_mastery={"topic_a": 0.5})
        result = learner.update("s7", topic_id="topic_a", correct=True)
        assert result.topic_label == "Topic A"

    def test_update_result_to_dict(self, learner):
        learner.load_student("s8", initial_mastery={"topic_a": 0.5})
        result = learner.update("s8", topic_id="topic_a", correct=True)
        d = result.to_dict()
        assert "mastery_before" in d
        assert "mastery_after" in d
        assert "delta" in d

    def test_get_or_create_new_student(self, learner):
        state = learner.get_or_create("brand_new_student", domain="test")
        assert state is not None
        assert "topic_a" in state.mastery

    def test_recommend_returns_results(self, learner):
        learner.load_student("s9", initial_mastery={"topic_a": 0.9})
        results = learner.recommend("s9")
        assert "approved" in results

    def test_session_summary(self, learner):
        learner.load_student("s10", initial_mastery={"topic_a": 0.5})
        learner.update("s10", topic_id="topic_a", correct=True)
        summary = learner.session_summary("s10")
        assert "total_interactions" in summary
        assert summary["total_interactions"] == 1

    def test_mode_is_bkt(self, learner):
        assert learner.mode == "bkt"

    def test_bkt_update_pure(self, learner):
        """Direct BKT update should be deterministic."""
        from plrs.assess.diagnostic import AssessmentSession
        state = learner.load_student("s_bkt", initial_mastery={"topic_a": 0.5})
        r1 = learner._bkt_update(state, "topic_a", correct=True)
        r2 = learner._bkt_update(state, "topic_a", correct=True)
        assert r1 == r2  # deterministic
        assert r1 > 0.5  # correct answer increases mastery
