"""
plrs.ranking.ranker
===================
Multi-objective ranking function for approved/challenging topics.

Scoring signals:
  1. Mastery gap        — how close the student is to mastering this topic
  2. Readiness          — fraction of prerequisites met
  3. Downstream value   — how many future topics this unlocks (normalised)
  4. Spaced repetition  — SuperMemo-2 review urgency (optional)

Default weights: gap=0.35, readiness=0.35, downstream=0.15, spaced_rep=0.15
When spaced repetition is disabled: gap=0.4, readiness=0.4, downstream=0.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from plrs.constraints.dag import ConstraintResult, MasteryVector
from plrs.curriculum.loader import CurriculumGraph

if TYPE_CHECKING:
    from plrs.ranking.spaced_repetition import SpacedRepetitionScorer


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
        Weight for mastery gap signal.
    w_readiness : float
        Weight for prerequisite readiness signal.
    w_downstream : float
        Weight for downstream unlock value.
    w_spaced_rep : float
        Weight for spaced repetition urgency (0.0 disables it).
    spaced_rep_scorer : SpacedRepetitionScorer, optional
        Pre-built scorer. If None and w_spaced_rep > 0, one is created
        automatically from the mastery vector on first rank() call.
    """

    def __init__(
        self,
        curriculum: CurriculumGraph,
        w_gap: float = 0.35,
        w_readiness: float = 0.35,
        w_downstream: float = 0.15,
        w_spaced_rep: float = 0.15,
        spaced_rep_scorer: "SpacedRepetitionScorer | None" = None,
    ) -> None:
        self.curriculum    = curriculum
        self.w_gap         = w_gap
        self.w_readiness   = w_readiness
        self.w_downstream  = w_downstream
        self.w_spaced_rep  = w_spaced_rep
        self._sr_scorer    = spaced_rep_scorer

        # Normalise weights to sum to 1.0
        total = w_gap + w_readiness + w_downstream + w_spaced_rep
        if total > 0:
            self.w_gap        /= total
            self.w_readiness  /= total
            self.w_downstream /= total
            self.w_spaced_rep /= total

        # Pre-compute downstream counts
        self._downstream_counts = self._compute_downstream_counts()
        max_d = max(self._downstream_counts.values(), default=1)
        self._downstream_norm = {
            node: count / max(max_d, 1)
            for node, count in self._downstream_counts.items()
        }

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def score(
        self,
        result: ConstraintResult,
        mastery: MasteryVector,
        sr_scores: dict[str, float] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute composite score for a single topic.

        Returns
        -------
        (score, breakdown_dict)
        """
        topic_id = result.topic_id

        # 1. Mastery gap
        gap = max(0.0, mastery.threshold - mastery.get(topic_id))
        gap_score = gap / mastery.threshold

        # 2. Readiness
        prereqs = self.curriculum.prerequisites(topic_id)
        readiness = (
            sum(1 for p in prereqs if mastery.get(p) >= mastery.soft_threshold)
            / len(prereqs)
            if prereqs else 1.0
        )

        # 3. Downstream value
        downstream = self._downstream_norm.get(topic_id, 0.0)

        # 4. Spaced repetition
        sr = sr_scores.get(topic_id, 0.5) if sr_scores else 0.5

        score = (
            self.w_gap       * gap_score
            + self.w_readiness  * readiness
            + self.w_downstream * downstream
            + self.w_spaced_rep * sr
        )

        breakdown = {
            "gap":        round(self.w_gap       * gap_score,  4),
            "readiness":  round(self.w_readiness * readiness,  4),
            "downstream": round(self.w_downstream * downstream, 4),
            "spaced_rep": round(self.w_spaced_rep * sr,         4),
        }

        return round(score, 4), breakdown

    def rank(
        self,
        results: list[ConstraintResult],
        mastery: MasteryVector,
        top_n: int = 5,
        challenging_penalty: float = 0.8,
    ) -> dict:
        """
        Rank constraint results into approved / challenging / vetoed.

        Parameters
        ----------
        results : list[ConstraintResult]
        mastery : MasteryVector
        top_n : int
        challenging_penalty : float

        Returns
        -------
        dict with keys: approved, challenging, vetoed, stats
        """
        # Build or refresh spaced repetition scores
        sr_scores = self._get_sr_scores(mastery)

        approved:   list[RankedRecommendation] = []
        challenging: list[RankedRecommendation] = []
        vetoed:     list[RankedRecommendation] = []

        for result in results:
            if mastery.is_mastered(result.topic_id):
                continue

            base_score, breakdown = self.score(result, mastery, sr_scores)
            final_score = base_score * (
                challenging_penalty if result.status == "challenging" else 1.0
            )

            rec = RankedRecommendation(
                topic_id=result.topic_id,
                topic_label=result.topic_label,
                status=result.status,
                mastery=round(result.mastery, 3),
                score=round(final_score, 4),
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
            "approved":    approved[:top_n],
            "challenging": challenging[:3],
            "vetoed":      vetoed[:5],
            "stats": {
                "total_topics":                total,
                "approved_count":              len(approved),
                "challenging_count":           len(challenging),
                "vetoed_count":                len(vetoed),
                "prerequisite_violation_rate": round(len(vetoed) / max(total, 1), 3),
                "spaced_rep_enabled":          self.w_spaced_rep > 0,
            },
        }

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_sr_scores(self, mastery: MasteryVector) -> dict[str, float] | None:
        """Build or reuse spaced repetition scores."""
        if self.w_spaced_rep == 0:
            return None

        if self._sr_scorer is None:
            from plrs.ranking.spaced_repetition import SpacedRepetitionScorer
            self._sr_scorer = SpacedRepetitionScorer()

        # Bootstrap from current mastery if scorer has no state yet
        if not self._sr_scorer._states:
            self._sr_scorer.build_from_mastery(mastery.to_dict())

        return self._sr_scorer.get_all_scores(mastery.to_dict())

    def _compute_downstream_counts(self) -> dict[str, int]:
        return {
            node: len(nx.descendants(self.curriculum.graph, node))
            for node in self.curriculum.nodes
        }
