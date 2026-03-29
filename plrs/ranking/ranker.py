"""
plrs.ranking.ranker
===================
Multi-objective ranking function for approved/challenging topics.

Scoring signals:
  1. Mastery gap       — how close the student is to mastering this topic
  2. Readiness         — fraction of prerequisites met
  3. Downstream value  — how many future topics this unlocks (normalised)

Weights are configurable. Default: gap=0.4, readiness=0.4, downstream=0.2
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from plrs.constraints.dag import ConstraintResult, MasteryVector
from plrs.curriculum.loader import CurriculumGraph


@dataclass
class RankedRecommendation:
    topic_id: str
    topic_label: str
    status: str          # "approved" | "challenging"
    mastery: float
    score: float
    reasoning: str
    prerequisites: list[str]
    unmet_prerequisites: list[str]
    downstream_count: int
    score_breakdown: dict[str, float]


class MultiObjectiveRanker:
    """
    Ranks constraint-validated topics by a weighted combination of signals.

    Parameters
    ----------
    curriculum : CurriculumGraph
    w_gap : float
        Weight for mastery gap signal (default 0.4).
    w_readiness : float
        Weight for prerequisite readiness signal (default 0.4).
    w_downstream : float
        Weight for downstream unlock value (default 0.2).
    """

    def __init__(
        self,
        curriculum: CurriculumGraph,
        w_gap: float = 0.4,
        w_readiness: float = 0.4,
        w_downstream: float = 0.2,
    ) -> None:
        self.curriculum = curriculum
        self.w_gap = w_gap
        self.w_readiness = w_readiness
        self.w_downstream = w_downstream

        # Pre-compute downstream counts (expensive on large graphs; cache it)
        self._downstream_counts = self._compute_downstream_counts()
        max_d = max(self._downstream_counts.values(), default=1)
        self._downstream_norm = {
            node: count / max(max_d, 1)
            for node, count in self._downstream_counts.items()
        }

    def _compute_downstream_counts(self) -> dict[str, int]:
        return {
            node: len(nx.descendants(self.curriculum.graph, node))
            for node in self.curriculum.nodes
        }

    def score(self, result: ConstraintResult, mastery: MasteryVector) -> float:
        """Compute composite score for a single topic."""
        topic_id = result.topic_id

        # 1. Mastery gap: student is close but not mastered → higher priority
        gap = max(0.0, mastery.threshold - mastery.get(topic_id))
        gap_score = gap / mastery.threshold  # normalise to [0, 1]

        # 2. Readiness: fraction of prerequisites above soft threshold
        prereqs = self.curriculum.prerequisites(topic_id)
        if prereqs:
            readiness = sum(
                1 for p in prereqs if mastery.get(p) >= mastery.soft_threshold
            ) / len(prereqs)
        else:
            readiness = 1.0

        # 3. Downstream value
        downstream = self._downstream_norm.get(topic_id, 0.0)

        score = (
            self.w_gap * gap_score
            + self.w_readiness * readiness
            + self.w_downstream * downstream
        )

        return round(score, 4)

    def rank(
        self,
        results: list[ConstraintResult],
        mastery: MasteryVector,
        top_n: int = 5,
        challenging_penalty: float = 0.8,
    ) -> dict[str, list[RankedRecommendation]]:
        """
        Rank a list of constraint results into approved / challenging / vetoed.

        Parameters
        ----------
        results : list[ConstraintResult]
        mastery : MasteryVector
        top_n : int
            Number of top approved recommendations to return.
        challenging_penalty : float
            Score multiplier applied to challenging topics (default 0.8).

        Returns
        -------
        dict with keys: "approved", "challenging", "vetoed", "stats"
        """
        approved: list[RankedRecommendation] = []
        challenging: list[RankedRecommendation] = []
        vetoed: list[RankedRecommendation] = []

        for result in results:
            # Skip already-mastered topics
            if mastery.is_mastered(result.topic_id):
                continue

            base_score = self.score(result, mastery)
            topic_id = result.topic_id

            breakdown = {
                "gap": round(
                    self.w_gap * max(0.0, mastery.threshold - mastery.get(topic_id)) / mastery.threshold, 4
                ),
                "readiness": round(self.w_readiness * (
                    sum(1 for p in self.curriculum.prerequisites(topic_id)
                        if mastery.get(p) >= mastery.soft_threshold)
                    / max(len(self.curriculum.prerequisites(topic_id)), 1)
                ), 4),
                "downstream": round(self.w_downstream * self._downstream_norm.get(topic_id, 0.0), 4),
            }

            rec = RankedRecommendation(
                topic_id=result.topic_id,
                topic_label=result.topic_label,
                status=result.status,
                mastery=round(result.mastery, 3),
                score=round(base_score * (challenging_penalty if result.status == "challenging" else 1.0), 4),
                reasoning=result.reasoning,
                prerequisites=result.prerequisites,
                unmet_prerequisites=result.unmet_prerequisites,
                downstream_count=self._downstream_counts.get(result.topic_id, 0),
                score_breakdown=breakdown,
            )

            if result.status == "approved":
                approved.append(rec)
            elif result.status == "challenging":
                challenging.append(rec)
            else:
                vetoed.append(rec)

        approved.sort(key=lambda r: r.score, reverse=True)
        challenging.sort(key=lambda r: r.score, reverse=True)

        total = len(results)
        return {
            "approved": approved[:top_n],
            "challenging": challenging[:3],
            "vetoed": vetoed[:5],
            "stats": {
                "total_topics": total,
                "approved_count": len(approved),
                "challenging_count": len(challenging),
                "vetoed_count": len(vetoed),
                "prerequisite_violation_rate": round(len(vetoed) / max(total, 1), 3),
            },
        }
