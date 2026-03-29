"""
plrs.constraints.dag
====================
DAG-based prerequisite constraint layer.

Three-tier classification:
  - approved    : prerequisites met, topic is ready
  - challenging : prerequisites partially met (above soft threshold)
  - vetoed      : prerequisites not met, topic is blocked
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from plrs.curriculum.loader import CurriculumGraph

Status = Literal["approved", "challenging", "vetoed"]


class MasteryVector:
    """
    Holds a student's estimated mastery probability per topic.

    Parameters
    ----------
    curriculum : CurriculumGraph
    threshold : float
        Mastery threshold — above this, a topic is considered mastered (default 0.70).
    soft_threshold : float
        Soft threshold — above this but below threshold, a topic is "challenging" (default 0.50).
    """

    def __init__(
        self,
        curriculum: CurriculumGraph,
        threshold: float = 0.70,
        soft_threshold: float = 0.50,
    ) -> None:
        self.curriculum = curriculum
        self.threshold = threshold
        self.soft_threshold = soft_threshold
        self._mastery: dict[str, float] = {node: 0.0 for node in curriculum.nodes}

    # ------------------------------------------------------------------ #
    # Mutations                                                            #
    # ------------------------------------------------------------------ #

    def update(self, topic_id: str, probability: float) -> None:
        """Set mastery probability for a topic (clamped to [0, 1])."""
        if topic_id in self._mastery:
            self._mastery[topic_id] = max(0.0, min(1.0, probability))

    def update_batch(self, updates: dict[str, float]) -> None:
        """Update multiple topics at once."""
        for topic_id, prob in updates.items():
            self.update(topic_id, prob)

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get(self, topic_id: str) -> float:
        return self._mastery.get(topic_id, 0.0)

    def is_mastered(self, topic_id: str) -> bool:
        return self.get(topic_id) >= self.threshold

    def is_partial(self, topic_id: str) -> bool:
        """Between soft_threshold and threshold — partially mastered."""
        v = self.get(topic_id)
        return self.soft_threshold <= v < self.threshold

    def summary(self) -> dict:
        mastered = [t for t in self._mastery if self.is_mastered(t)]
        partial  = [t for t in self._mastery if self.is_partial(t)]
        return {
            "total_topics": len(self._mastery),
            "mastered": len(mastered),
            "partial": len(partial),
            "not_started": len(self._mastery) - len(mastered) - len(partial),
            "mastery_rate": round(len(mastered) / max(len(self._mastery), 1), 3),
            "mastered_topics": mastered,
        }

    def to_dict(self) -> dict[str, float]:
        return dict(self._mastery)

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"MasteryVector(mastered={s['mastered']}/{s['total_topics']}, "
            f"rate={s['mastery_rate']:.1%})"
        )


@dataclass
class ConstraintResult:
    topic_id: str
    topic_label: str
    status: Status
    mastery: float
    reasoning: str
    score: float = 0.0
    prerequisites: list[str] = field(default_factory=list)
    unmet_prerequisites: list[str] = field(default_factory=list)


class DAGConstraintLayer:
    """
    Validates topic recommendations against curriculum prerequisite structure.

    Uses three-tier soft constraint logic:
      - mastery >= threshold on ALL prerequisites  → approved
      - mastery >= soft_threshold on ALL prereqs   → challenging
      - any prerequisite below soft_threshold      → vetoed
    """

    def __init__(self, curriculum: CurriculumGraph) -> None:
        self.curriculum = curriculum

    def validate(
        self,
        topic_id: str,
        mastery: MasteryVector,
    ) -> ConstraintResult:
        label = self.curriculum.label(topic_id)
        prereqs = self.curriculum.prerequisites(topic_id)
        topic_mastery = mastery.get(topic_id)

        if not prereqs:
            return ConstraintResult(
                topic_id=topic_id,
                topic_label=label,
                status="approved",
                mastery=topic_mastery,
                reasoning="No prerequisites required.",
                prerequisites=[],
                unmet_prerequisites=[],
            )

        prereq_labels = [self.curriculum.label(p) for p in prereqs]
        unmet_hard = [p for p in prereqs if not mastery.is_mastered(p)]
        unmet_soft = [p for p in prereqs if mastery.get(p) < mastery.soft_threshold]

        if not unmet_soft:
            # All prereqs above soft threshold — at least challenging
            if not unmet_hard:
                status = "approved"
                reasoning = f"All {len(prereqs)} prerequisite(s) met."
            else:
                status = "challenging"
                unmet_labels = [self.curriculum.label(p) for p in unmet_hard]
                reasoning = (
                    f"Prerequisite(s) partially met. "
                    f"Strengthen: {', '.join(unmet_labels)}."
                )
        else:
            status = "vetoed"
            unmet_labels = [self.curriculum.label(p) for p in unmet_soft]
            reasoning = (
                f"Blocked. Master first: {', '.join(unmet_labels)}."
            )

        return ConstraintResult(
            topic_id=topic_id,
            topic_label=label,
            status=status,
            mastery=topic_mastery,
            reasoning=reasoning,
            prerequisites=prereq_labels,
            unmet_prerequisites=[self.curriculum.label(p) for p in (unmet_hard if status == "challenging" else unmet_soft)],
        )

    def validate_all(self, mastery: MasteryVector) -> list[ConstraintResult]:
        """Validate every topic in the curriculum."""
        return [self.validate(node, mastery) for node in self.curriculum.nodes]
