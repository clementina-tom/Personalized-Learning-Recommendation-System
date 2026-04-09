"""
tests/test_spaced_repetition.py
================================
Tests for SuperMemo-2 spaced repetition scoring and ranker integration.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAPS = ROOT / "data" / "knowledge_maps"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def curriculum():
    from plrs.curriculum.loader import load_dag
    return load_dag(MAPS / "math_dag.json")


@pytest.fixture(scope="module")
def pipeline(curriculum):
    from plrs.pipeline import PLRSPipeline
    return PLRSPipeline(curriculum)


# ── SM-2 unit tests ───────────────────────────────────────────────────────────

class TestMasteryToQuality:
    def test_low_mastery_gives_low_quality(self):
        from plrs.ranking.spaced_repetition import mastery_to_quality
        assert mastery_to_quality(0.10) == 1
        assert mastery_to_quality(0.29) == 1

    def test_medium_mastery(self):
        from plrs.ranking.spaced_repetition import mastery_to_quality
        assert mastery_to_quality(0.40) == 2
        assert mastery_to_quality(0.55) == 3
        assert mastery_to_quality(0.65) == 4

    def test_high_mastery_gives_quality_5(self):
        from plrs.ranking.spaced_repetition import mastery_to_quality
        assert mastery_to_quality(0.80) == 5
        assert mastery_to_quality(1.00) == 5

    def test_boundary_values(self):
        from plrs.ranking.spaced_repetition import mastery_to_quality
        assert mastery_to_quality(0.0)  == 1
        assert mastery_to_quality(0.30) == 2
        assert mastery_to_quality(0.75) == 5


class TestSM2State:
    def test_initial_state(self):
        from plrs.ranking.spaced_repetition import SM2State, SM2_INIT_EASE
        s = SM2State(topic_id="test")
        assert s.repetitions == 0
        assert s.ease_factor == SM2_INIT_EASE
        assert s.interval_days == 1.0

    def test_first_correct_interval(self):
        from plrs.ranking.spaced_repetition import SM2State
        s = SM2State(topic_id="test")
        interval = s.next_interval(quality=4)
        assert interval == 1.0
        assert s.repetitions == 1

    def test_second_correct_interval(self):
        from plrs.ranking.spaced_repetition import SM2State
        s = SM2State(topic_id="test")
        s.next_interval(quality=4)
        interval = s.next_interval(quality=4)
        assert interval == 6.0
        assert s.repetitions == 2

    def test_third_interval_uses_ease_factor(self):
        from plrs.ranking.spaced_repetition import SM2State
        s = SM2State(topic_id="test")
        s.next_interval(quality=5)
        s.next_interval(quality=5)
        interval = s.next_interval(quality=5)
        assert interval > 6.0    # must grow beyond 6 days
        assert s.repetitions == 3

    def test_incorrect_resets_repetitions(self):
        from plrs.ranking.spaced_repetition import SM2State
        s = SM2State(topic_id="test")
        s.next_interval(quality=5)
        s.next_interval(quality=5)
        s.next_interval(quality=1)   # incorrect
        assert s.repetitions == 0
        assert s.interval_days == 1.0

    def test_ease_factor_never_below_minimum(self):
        from plrs.ranking.spaced_repetition import SM2State, SM2_MIN_EASE
        s = SM2State(topic_id="test")
        for _ in range(20):
            s.next_interval(quality=1)  # always wrong
        assert s.ease_factor >= SM2_MIN_EASE

    def test_is_due(self):
        from plrs.ranking.spaced_repetition import SM2State
        s = SM2State(topic_id="test", last_reviewed_day=5.0, interval_days=3.0)
        assert s.is_due(8.0) is True   # exactly due
        assert s.is_due(7.9) is False  # not yet due

    def test_overdue_ratio(self):
        from plrs.ranking.spaced_repetition import SM2State
        s = SM2State(topic_id="test", last_reviewed_day=0.0, interval_days=5.0)
        s.last_reviewed_day = 5.0  # reviewed at day 5
        assert s.overdue_ratio(10.0) == pytest.approx(1.0)   # exactly due
        assert s.overdue_ratio(15.0) == pytest.approx(2.0)   # 2x overdue
        assert s.overdue_ratio(5.0)  == pytest.approx(0.0)   # just reviewed

    def test_never_reviewed_gives_neutral(self):
        from plrs.ranking.spaced_repetition import SM2State
        s = SM2State(topic_id="test")  # last_reviewed_day=0.0
        assert s.overdue_ratio(10.0) == 0.5


class TestSpacedRepetitionScorer:
    def test_score_never_seen_is_neutral(self):
        from plrs.ranking.spaced_repetition import SpacedRepetitionScorer
        scorer = SpacedRepetitionScorer()
        score = scorer.score("unknown_topic", mastery=0.5)
        assert score == 0.5

    def test_score_after_update_changes(self):
        from plrs.ranking.spaced_repetition import SpacedRepetitionScorer
        scorer = SpacedRepetitionScorer()
        scorer.update("algebra", mastery=0.8, step=0)
        score_fresh = scorer.score("algebra", mastery=0.8)
        # Advance many steps — topic should be overdue
        scorer._current_step = 200
        score_overdue = scorer.score("algebra", mastery=0.8)
        assert score_overdue > score_fresh

    def test_scores_in_range(self):
        from plrs.ranking.spaced_repetition import SpacedRepetitionScorer
        scorer = SpacedRepetitionScorer()
        for mastery in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            scorer.update("topic", mastery=mastery, step=10)
            score = scorer.score("topic", mastery)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for mastery={mastery}"

    def test_build_from_mastery(self):
        from plrs.ranking.spaced_repetition import SpacedRepetitionScorer
        scorer = SpacedRepetitionScorer()
        mastery_dict = {
            "whole_numbers": 0.90,
            "algebra_basics": 0.60,
            "quadratic_equations": 0.0,
        }
        scorer.build_from_mastery(mastery_dict)

        # High mastery → recently reviewed → lower urgency
        # Low mastery → reviewed less recently → higher urgency
        score_high = scorer.score("whole_numbers", 0.90)
        score_mid  = scorer.score("algebra_basics", 0.60)
        score_zero = scorer.score("quadratic_equations", 0.0)

        assert score_high < score_mid       # high mastery reviewed more recently
        assert score_zero == 0.5             # never seen → neutral

    def test_get_all_scores(self):
        from plrs.ranking.spaced_repetition import SpacedRepetitionScorer
        scorer = SpacedRepetitionScorer()
        mastery_dict = {"a": 0.8, "b": 0.4, "c": 0.0}
        scorer.build_from_mastery(mastery_dict)
        scores = scorer.get_all_scores(mastery_dict)
        assert set(scores.keys()) == {"a", "b", "c"}
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_summary(self):
        from plrs.ranking.spaced_repetition import SpacedRepetitionScorer
        scorer = SpacedRepetitionScorer()
        scorer.update("topic_a", mastery=0.8, step=10)
        scorer.update("topic_b", mastery=0.3, step=10)
        summary = scorer.summary()
        assert summary["tracked_topics"] == 2
        assert "overdue_topics" in summary
        assert "avg_interval_days" in summary


# ── Ranker integration tests ──────────────────────────────────────────────────

class TestRankerWithSpacedRepetition:
    def test_ranker_has_spaced_rep_signal(self, curriculum):
        from plrs.ranking.ranker import MultiObjectiveRanker
        ranker = MultiObjectiveRanker(curriculum, w_spaced_rep=0.15)
        assert ranker.w_spaced_rep > 0

    def test_score_breakdown_includes_spaced_rep(self, curriculum):
        from plrs.constraints.dag import DAGConstraintLayer, MasteryVector
        from plrs.ranking.ranker import MultiObjectiveRanker

        ranker = MultiObjectiveRanker(curriculum, w_spaced_rep=0.15)
        mv = MasteryVector(curriculum)
        mv.update("whole_numbers", 0.90)

        layer = DAGConstraintLayer(curriculum)
        results = layer.validate_all(mv)

        ranked = ranker.rank(results, mv, top_n=3)
        if ranked["approved"]:
            # ranker.rank() returns RankedRecommendation dataclasses
            rec = ranked["approved"][0]
            breakdown = rec.score_breakdown
            assert "spaced_rep" in breakdown
            assert 0.0 <= breakdown["spaced_rep"] <= ranker.w_spaced_rep + 0.01

    def test_spaced_rep_disabled(self, curriculum):
        from plrs.constraints.dag import DAGConstraintLayer, MasteryVector
        from plrs.ranking.ranker import MultiObjectiveRanker

        ranker = MultiObjectiveRanker(
            curriculum,
            w_gap=0.4, w_readiness=0.4, w_downstream=0.2, w_spaced_rep=0.0
        )
        mv = MasteryVector(curriculum)
        mv.update("whole_numbers", 0.90)

        layer = DAGConstraintLayer(curriculum)
        results = layer.validate_all(mv)
        ranked = ranker.rank(results, mv)
        assert ranked["stats"]["spaced_rep_enabled"] is False

    def test_weights_normalised(self, curriculum):
        from plrs.ranking.ranker import MultiObjectiveRanker
        ranker = MultiObjectiveRanker(
            curriculum,
            w_gap=0.4, w_readiness=0.4, w_downstream=0.2, w_spaced_rep=0.2
        )
        total = ranker.w_gap + ranker.w_readiness + ranker.w_downstream + ranker.w_spaced_rep
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_overdue_topic_ranks_higher(self, curriculum):
        """A topic that's overdue for review should outrank an equivalent fresh one."""
        from plrs.constraints.dag import MasteryVector
        from plrs.ranking.ranker import MultiObjectiveRanker
        from plrs.ranking.spaced_repetition import SpacedRepetitionScorer

        # Two topics with equal mastery — one overdue, one fresh
        scorer = SpacedRepetitionScorer()
        nodes = curriculum.nodes
        if len(nodes) < 2:
            pytest.skip("Need at least 2 nodes")

        t1, t2 = nodes[0], nodes[1]
        # t1: reviewed long ago → overdue
        scorer.update(t1, mastery=0.5, step=0)
        scorer._current_step = 500     # advance far in time
        # t2: reviewed very recently → not due
        scorer.update(t2, mastery=0.5, step=500)

        score_overdue = scorer.score(t1, 0.5)
        score_fresh   = scorer.score(t2, 0.5)
        assert score_overdue > score_fresh

    def test_pipeline_recommend_includes_spaced_rep(self, pipeline):
        results = pipeline.recommend_from_mastery({
            "whole_numbers": 0.85,
            "number_bases": 0.80,
        })
        assert "approved" in results
        assert results["stats"]["spaced_rep_enabled"] is True
        if results["approved"]:
            assert "spaced_rep" in results["approved"][0]["score_breakdown"]
