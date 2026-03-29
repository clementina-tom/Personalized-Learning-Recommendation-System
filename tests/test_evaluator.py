"""
tests/test_evaluator.py
=======================
Tests for the PLRS evaluation suite.
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MAPS = ROOT / "data" / "knowledge_maps"


@pytest.fixture(scope="module")
def pipeline():
    from plrs.curriculum.loader import load_dag
    from plrs.pipeline import PLRSPipeline
    curriculum = load_dag(MAPS / "math_dag.json")
    return PLRSPipeline(curriculum)


@pytest.fixture
def fake_sequences():
    np.random.seed(42)
    return [
        (list(np.random.randint(0, 100, 30)), list(np.random.randint(0, 2, 30)))
        for _ in range(20)
    ]


@pytest.fixture
def evaluator(pipeline):
    from plrs.model.evaluator import PLRSEvaluator
    return PLRSEvaluator(pipeline)


class TestBaselines:
    def test_random_baseline(self, fake_sequences):
        from plrs.model.evaluator import PLRSEvaluator, RandomBaseline
        rb = RandomBaseline()
        probs = rb.predict(fake_sequences[0][0], fake_sequences[0][1])
        assert all(v == 0.5 for v in probs.values())

    def test_bkt_baseline(self, fake_sequences):
        from plrs.model.evaluator import BKTBaseline
        bkt = BKTBaseline()
        skill_seq, correct_seq = fake_sequences[0]
        probs = bkt.predict_sequence(skill_seq, correct_seq)
        assert len(probs) == len(skill_seq) - 1
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_popularity_baseline_fit_predict(self, fake_sequences):
        from plrs.model.evaluator import PopularityBaseline
        pop = PopularityBaseline()
        pop.fit(fake_sequences)
        prob = pop.predict_prob(0)
        assert 0.0 <= prob <= 1.0

    def test_popularity_recommend(self, pipeline, fake_sequences):
        from plrs.model.evaluator import PopularityBaseline
        pop = PopularityBaseline()
        skill_to_topic = {i: node for i, node in enumerate(pipeline.curriculum.nodes)}
        pop.fit(fake_sequences, skill_to_topic)
        recs = pop.recommend(pipeline.curriculum, n=5)
        assert len(recs) <= 5
        assert all(t in pipeline.curriculum.nodes for t in recs)


class TestKTMetrics:
    def test_compute_metrics_returns_correct_fields(self):
        from plrs.model.evaluator import PLRSEvaluator
        probs  = [0.8, 0.3, 0.9, 0.1, 0.7]
        labels = [1.0, 0.0, 1.0, 0.0, 1.0]
        m = PLRSEvaluator._compute_kt_metrics("test", probs, labels, elapsed=0.1)
        assert m.model_name == "test"
        assert 0.0 <= m.auc <= 1.0
        assert 0.0 <= m.accuracy <= 1.0
        assert m.log_loss >= 0.0
        assert m.n_samples == 5

    def test_perfect_predictor_auc_1(self):
        from plrs.model.evaluator import PLRSEvaluator
        probs  = [0.99, 0.01, 0.99, 0.01]
        labels = [1.0,  0.0,  1.0,  0.0]
        m = PLRSEvaluator._compute_kt_metrics("perfect", probs, labels, elapsed=0.0)
        assert m.auc == pytest.approx(1.0, abs=0.01)

    def test_random_predictor_auc_half(self):
        from plrs.model.evaluator import PLRSEvaluator
        np.random.seed(0)
        probs  = [0.5] * 1000
        labels = list(np.random.randint(0, 2, 1000).astype(float))
        m = PLRSEvaluator._compute_kt_metrics("random", probs, labels, elapsed=0.0)
        assert m.auc == pytest.approx(0.5, abs=0.05)


class TestRecommendationMetrics:
    def test_zero_violation_when_mastered(self, evaluator, pipeline):
        mastered = {n: 0.95 for n in pipeline.curriculum.nodes}
        results = pipeline.recommend_from_mastery(mastered)
        assert results["stats"]["prerequisite_violation_rate"] == 0.0

    def test_rec_metrics_returned(self, evaluator, fake_sequences):
        report = evaluator.evaluate(
            test_sequences=fake_sequences[:5],
            include_baselines=False,
        )
        assert report.rec_metrics is not None
        r = report.rec_metrics
        assert 0.0 <= r.violation_rate <= 1.0
        assert 0.0 <= r.coverage <= 1.0
        assert r.avg_downstream >= 0.0
        assert 0.0 <= r.mastery_rate <= 1.0


class TestEvaluationReport:
    def test_report_structure(self, evaluator, fake_sequences):
        report = evaluator.evaluate(
            test_sequences=fake_sequences[:5],
            include_baselines=True,
        )
        assert len(report.kt_metrics) >= 3   # random + BKT + popularity
        assert report.rec_metrics is not None
        assert report.timestamp is not None

    def test_report_to_dict(self, evaluator, fake_sequences):
        report = evaluator.evaluate(
            test_sequences=fake_sequences[:5],
            include_baselines=False,
        )
        d = report.to_dict()
        assert "kt_metrics" in d
        assert "rec_metrics" in d
        assert "config" in d
        assert "timestamp" in d

    def test_report_json_serializable(self, evaluator, fake_sequences):
        report = evaluator.evaluate(
            test_sequences=fake_sequences[:5],
            include_baselines=False,
        )
        json_str = json.dumps(report.to_dict())
        loaded = json.loads(json_str)
        assert "kt_metrics" in loaded

    def test_report_print_runs(self, evaluator, fake_sequences, capsys):
        report = evaluator.evaluate(
            test_sequences=fake_sequences[:5],
            include_baselines=False,
        )
        report.print()
        captured = capsys.readouterr()
        assert "EVALUATION REPORT" in captured.out
        assert "AUC" in captured.out
