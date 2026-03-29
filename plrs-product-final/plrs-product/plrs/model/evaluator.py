"""
plrs.model.evaluator
====================
Evaluation suite for PLRS.

Metrics:
  - Knowledge Tracing: AUC-ROC, Accuracy, Binary Cross-Entropy
  - Recommendation: Prerequisite Violation Rate, Coverage, Diversity
  - Baselines: Random, Popularity, BKT (Bayesian Knowledge Tracing)

Usage:
    from plrs.model.evaluator import PLRSEvaluator
    evaluator = PLRSEvaluator(pipeline, curriculum)
    report = evaluator.evaluate(test_sequences, skill_to_topic)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Baseline models ───────────────────────────────────────────────────────────

class RandomBaseline:
    """Predicts 0.5 for every interaction."""
    def predict(self, skill_seq, correct_seq):
        return {i: 0.5 for i in range(len(skill_seq))}

    def recommend(self, curriculum, n=5):
        import random
        return random.sample(curriculum.nodes, min(n, len(curriculum.nodes)))


class PopularityBaseline:
    """Recommends the most-seen skills; predicts by global correctness rate."""

    def __init__(self):
        self.skill_correct: dict[int, list[float]] = {}
        self.topic_count:   dict[str, int] = {}

    def fit(self, sequences, skill_to_topic=None):
        for skill_seq, correct_seq in sequences:
            for skill, correct in zip(skill_seq, correct_seq):
                self.skill_correct.setdefault(skill, []).append(float(correct))
                if skill_to_topic:
                    topic = skill_to_topic.get(skill)
                    if topic:
                        self.topic_count[topic] = self.topic_count.get(topic, 0) + 1

    def predict_prob(self, skill_id: int) -> float:
        history = self.skill_correct.get(skill_id, [])
        return float(np.mean(history)) if history else 0.5

    def recommend(self, curriculum, n=5):
        if not self.topic_count:
            return curriculum.nodes[:n]
        sorted_topics = sorted(self.topic_count, key=self.topic_count.get, reverse=True)
        return [t for t in sorted_topics if t in curriculum.nodes][:n]


class BKTBaseline:
    """
    Bayesian Knowledge Tracing (per-skill).
    Simple 4-parameter model: p_init, p_transit, p_slip, p_guess.
    """

    def __init__(self, p_init=0.3, p_transit=0.1, p_slip=0.1, p_guess=0.2):
        self.p_init    = p_init
        self.p_transit = p_transit
        self.p_slip    = p_slip
        self.p_guess   = p_guess
        self._mastery:  dict[int, float] = {}

    def _update(self, skill: int, correct: int) -> float:
        p = self._mastery.get(skill, self.p_init)
        # Bayes update
        if correct:
            num = p * (1 - self.p_slip)
            den = num + (1 - p) * self.p_guess
        else:
            num = p * self.p_slip
            den = num + (1 - p) * (1 - self.p_guess)
        p_post = num / max(den, 1e-9)
        # Learning
        p_post = p_post + (1 - p_post) * self.p_transit
        self._mastery[skill] = p_post
        return p_post

    def predict_sequence(self, skill_seq: list[int], correct_seq: list[int]) -> list[float]:
        self._mastery = {}
        probs = []
        for skill, correct in zip(skill_seq[:-1], correct_seq[:-1]):
            self._update(skill, correct)
            next_skill = skill_seq[len(probs) + 1]
            probs.append(self._mastery.get(next_skill, self.p_init))
        return probs

    def get_mastery(self) -> dict[int, float]:
        return dict(self._mastery)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class KTMetrics:
    """Knowledge tracing evaluation metrics."""
    model_name:  str
    auc:         float
    accuracy:    float
    log_loss:    float
    n_samples:   int
    elapsed_s:   float


@dataclass
class RecommendMetrics:
    """Recommendation quality metrics."""
    violation_rate:    float   # fraction of recommendations that violate prerequisites
    coverage:          float   # fraction of curriculum covered by recommendations
    avg_downstream:    float   # avg topics unlocked by recommendations
    mastery_rate:      float   # avg student mastery in test set


@dataclass
class EvaluationReport:
    """Full evaluation report."""
    kt_metrics:      list[KTMetrics]
    rec_metrics:     RecommendMetrics | None
    config:          dict[str, Any]
    timestamp:       str

    def print(self) -> None:
        print("\n" + "=" * 62)
        print("  PLRS EVALUATION REPORT")
        print("=" * 62)

        print(f"\n{'Model':<22} {'AUC':>8} {'Accuracy':>10} {'Log Loss':>10} {'Samples':>8}")
        print("-" * 62)
        for m in self.kt_metrics:
            print(f"{m.model_name:<22} {m.auc:>8.4f} {m.accuracy:>10.4f} {m.log_loss:>10.4f} {m.n_samples:>8,}")

        if self.rec_metrics:
            r = self.rec_metrics
            print(f"\n{'Recommendation Metrics':}")
            print(f"  Prerequisite violation rate : {r.violation_rate:.1%}")
            print(f"  Curriculum coverage         : {r.coverage:.1%}")
            print(f"  Avg downstream unlocked     : {r.avg_downstream:.1f}")
            print(f"  Avg student mastery rate    : {r.mastery_rate:.1%}")

        print("=" * 62 + "\n")

    def to_dict(self) -> dict:
        return {
            "kt_metrics": [
                {
                    "model": m.model_name,
                    "auc": round(m.auc, 6),
                    "accuracy": round(m.accuracy, 6),
                    "log_loss": round(m.log_loss, 6),
                    "n_samples": m.n_samples,
                    "elapsed_s": round(m.elapsed_s, 3),
                }
                for m in self.kt_metrics
            ],
            "rec_metrics": {
                "violation_rate": round(self.rec_metrics.violation_rate, 6),
                "coverage": round(self.rec_metrics.coverage, 6),
                "avg_downstream": round(self.rec_metrics.avg_downstream, 3),
                "mastery_rate": round(self.rec_metrics.mastery_rate, 6),
            } if self.rec_metrics else None,
            "config": self.config,
            "timestamp": self.timestamp,
        }


# ── Main evaluator ────────────────────────────────────────────────────────────

class PLRSEvaluator:
    """
    Evaluate PLRS against baselines on held-out student sequences.

    Parameters
    ----------
    pipeline : PLRSPipeline
        A loaded pipeline (with or without SAKT model).
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.curriculum = pipeline.curriculum

    def evaluate(
        self,
        test_sequences: list[tuple[list[int], list[int]]],
        skill_to_topic: dict[int, str] | None = None,
        train_sequences: list[tuple[list[int], list[int]]] | None = None,
        include_baselines: bool = True,
    ) -> EvaluationReport:
        """
        Run full evaluation.

        Parameters
        ----------
        test_sequences : list of (skill_seq, correct_seq)
        skill_to_topic : dict mapping skill_id → curriculum topic_id
        train_sequences : used to fit popularity baseline
        include_baselines : whether to evaluate BKT and popularity baselines

        Returns
        -------
        EvaluationReport
        """
        import datetime

        kt_metrics: list[KTMetrics] = []

        # ── SAKT evaluation ──────────────────────────────────────────
        if self.pipeline._model is not None:
            kt_metrics.append(
                self._eval_sakt(test_sequences)
            )

        # ── Baselines ────────────────────────────────────────────────
        if include_baselines:
            kt_metrics.append(self._eval_random(test_sequences))
            kt_metrics.append(self._eval_bkt(test_sequences))

            pop = PopularityBaseline()
            pop.fit(train_sequences or test_sequences, skill_to_topic)
            kt_metrics.append(self._eval_popularity(test_sequences, pop))

        # ── Recommendation metrics ───────────────────────────────────
        rec_metrics = self._eval_recommendations(test_sequences, skill_to_topic)

        return EvaluationReport(
            kt_metrics=kt_metrics,
            rec_metrics=rec_metrics,
            config={
                "threshold": self.pipeline.threshold,
                "soft_threshold": self.pipeline.soft_threshold,
                "top_n": self.pipeline.top_n,
                "n_test_students": len(test_sequences),
            },
            timestamp=datetime.datetime.now().isoformat(),
        )

    # ── KT evaluation helpers ─────────────────────────────────────────────────

    def _eval_sakt(self, sequences) -> KTMetrics:
        t0 = time.time()
        all_probs, all_labels = [], []

        for skill_seq, correct_seq in sequences:
            if len(skill_seq) < 2:
                continue
            probs = self.pipeline._model.predict_mastery(skill_seq, correct_seq)
            for skill_id, prob in probs.items():
                if skill_id < len(correct_seq):
                    all_probs.append(prob)
                    all_labels.append(float(correct_seq[skill_id]))

        return self._compute_kt_metrics("SAKT", all_probs, all_labels, time.time() - t0)

    def _eval_random(self, sequences) -> KTMetrics:
        t0 = time.time()
        all_probs, all_labels = [], []
        for skill_seq, correct_seq in sequences:
            for correct in correct_seq[1:]:
                all_probs.append(0.5)
                all_labels.append(float(correct))
        return self._compute_kt_metrics("Random (baseline)", all_probs, all_labels, time.time() - t0)

    def _eval_bkt(self, sequences) -> KTMetrics:
        t0 = time.time()
        all_probs, all_labels = [], []
        bkt = BKTBaseline()
        for skill_seq, correct_seq in sequences:
            if len(skill_seq) < 2:
                continue
            probs = bkt.predict_sequence(skill_seq, correct_seq)
            labels = [float(c) for c in correct_seq[1:len(probs) + 1]]
            all_probs.extend(probs)
            all_labels.extend(labels)
        return self._compute_kt_metrics("BKT (baseline)", all_probs, all_labels, time.time() - t0)

    def _eval_popularity(self, sequences, pop: PopularityBaseline) -> KTMetrics:
        t0 = time.time()
        all_probs, all_labels = [], []
        for skill_seq, correct_seq in sequences:
            for skill, correct in zip(skill_seq[1:], correct_seq[1:]):
                all_probs.append(pop.predict_prob(skill))
                all_labels.append(float(correct))
        return self._compute_kt_metrics("Popularity (baseline)", all_probs, all_labels, time.time() - t0)

    @staticmethod
    def _compute_kt_metrics(name, probs, labels, elapsed) -> KTMetrics:
        probs_arr  = np.nan_to_num(np.array(probs),  nan=0.5)
        labels_arr = np.nan_to_num(np.array(labels), nan=0.0)
        n = len(probs_arr)

        if HAS_SKLEARN and n > 0 and len(np.unique(labels_arr)) > 1:
            auc  = float(roc_auc_score(labels_arr, probs_arr))
            acc  = float(accuracy_score(labels_arr, (probs_arr >= 0.5).astype(int)))
            loss = float(log_loss(labels_arr, np.clip(probs_arr, 1e-7, 1 - 1e-7)))
        else:
            auc  = 0.5
            acc  = float(((probs_arr >= 0.5) == labels_arr).mean()) if n > 0 else 0.0
            loss = float(-np.mean(
                labels_arr * np.log(probs_arr + 1e-7) +
                (1 - labels_arr) * np.log(1 - probs_arr + 1e-7)
            )) if n > 0 else 0.0

        return KTMetrics(
            model_name=name, auc=auc, accuracy=acc,
            log_loss=loss, n_samples=n, elapsed_s=elapsed,
        )

    # ── Recommendation evaluation ─────────────────────────────────────────────

    def _eval_recommendations(
        self,
        sequences,
        skill_to_topic,
    ) -> RecommendMetrics:
        violation_rates, coverages, downstreams, mastery_rates = [], [], [], []

        for skill_seq, correct_seq in sequences:
            # Build mastery from sequence
            if skill_to_topic:
                topic_scores: dict[str, float] = {}
                for skill, correct in zip(skill_seq, correct_seq):
                    topic = skill_to_topic.get(skill)
                    if topic and topic in self.curriculum.nodes:
                        topic_scores[topic] = max(topic_scores.get(topic, 0.0), float(correct))
                mastery_scores = {n: 0.0 for n in self.curriculum.nodes}
                mastery_scores.update(topic_scores)
            else:
                mastery_scores = {n: 0.0 for n in self.curriculum.nodes}

            results = self.pipeline.recommend_from_mastery(mastery_scores)
            stats   = results["stats"]
            summary = results["mastery_summary"]

            violation_rates.append(stats["prerequisite_violation_rate"])
            mastery_rates.append(summary["mastery_rate"])

            # Coverage: fraction of curriculum represented in approved+challenging
            rec_topics = set(
                r["topic_id"] for r in results["approved"] + results["challenging"]
            )
            coverages.append(len(rec_topics) / max(self.curriculum.num_nodes, 1))

            # Avg downstream unlock value
            if results["approved"]:
                downstreams.append(
                    np.mean([r["downstream_count"] for r in results["approved"]])
                )

        return RecommendMetrics(
            violation_rate=float(np.mean(violation_rates)) if violation_rates else 0.0,
            coverage=float(np.mean(coverages)) if coverages else 0.0,
            avg_downstream=float(np.mean(downstreams)) if downstreams else 0.0,
            mastery_rate=float(np.mean(mastery_rates)) if mastery_rates else 0.0,
        )
