"""
plrs.pipeline
=============
PLRSPipeline: the main entry point.

Orchestrates SAKT inference → DAG constraint validation → multi-objective ranking.

Usage
-----
    from plrs import PLRSPipeline
    from plrs.curriculum import load_dag

    curriculum = load_dag("math_dag.json")
    pipeline   = PLRSPipeline(curriculum, model_path="sakt_model.pt")

    # From raw interaction history
    results = pipeline.recommend_from_history(
        skill_seq=[12, 45, 3, 78],
        correct_seq=[1, 0, 1, 1],
    )

    # From pre-computed mastery dict
    results = pipeline.recommend_from_mastery(
        mastery_scores={"algebra_basics": 0.85, "quadratic_equations": 0.42}
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plrs.constraints.dag import DAGConstraintLayer, MasteryVector
from plrs.curriculum.loader import CurriculumGraph
from plrs.ranking.ranker import MultiObjectiveRanker, RankedRecommendation


class PLRSPipeline:
    """
    End-to-end PLRS recommendation pipeline.

    Parameters
    ----------
    curriculum : CurriculumGraph
    model_path : str or Path, optional
        Path to a trained SAKT .pt file. If None, only mastery-dict mode is available.
    threshold : float
        Mastery threshold (default 0.70).
    soft_threshold : float
        Soft constraint threshold (default 0.50).
    top_n : int
        Number of top approved recommendations (default 5).
    w_gap, w_readiness, w_downstream : float
        Ranker objective weights.
    device : str
        PyTorch device for model inference (default "cpu").
    """

    def __init__(
        self,
        curriculum: CurriculumGraph,
        model_path: str | Path | None = None,
        threshold: float = 0.70,
        soft_threshold: float = 0.50,
        top_n: int = 5,
        w_gap: float = 0.4,
        w_readiness: float = 0.4,
        w_downstream: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.curriculum = curriculum
        self.threshold = threshold
        self.soft_threshold = soft_threshold
        self.top_n = top_n
        self.device = device

        self.constraint_layer = DAGConstraintLayer(curriculum)
        self.ranker = MultiObjectiveRanker(
            curriculum,
            w_gap=w_gap,
            w_readiness=w_readiness,
            w_downstream=w_downstream,
        )

        self._model = None
        if model_path is not None:
            self._load_model(model_path)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def recommend_from_mastery(
        self,
        mastery_scores: dict[str, float],
    ) -> dict[str, Any]:
        """
        Generate recommendations from a pre-computed mastery dict.

        Parameters
        ----------
        mastery_scores : dict[str, float]
            Mapping from topic_id → mastery probability [0, 1].

        Returns
        -------
        dict with keys: approved, challenging, vetoed, stats, mastery_summary
        """
        mastery = self._build_mastery_vector(mastery_scores)
        return self._run(mastery)

    def recommend_from_history(
        self,
        skill_seq: list[int],
        correct_seq: list[int],
        skill_to_topic: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate recommendations from raw student interaction history.

        Requires a loaded SAKT model (pass model_path to __init__).

        Parameters
        ----------
        skill_seq : list[int]
            Sequence of skill IDs from the student's history.
        correct_seq : list[int]
            Corresponding correctness flags (1/0).
        skill_to_topic : dict[int, str], optional
            Mapping from SAKT skill_id → curriculum topic_id.
            Required to map model output back to DAG nodes.

        Returns
        -------
        dict with keys: approved, challenging, vetoed, stats, mastery_summary
        """
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Pass model_path to PLRSPipeline() to use history-based inference."
            )

        skill_probs = self._model.predict_mastery(skill_seq, correct_seq, device=self.device)

        if skill_to_topic:
            mastery_scores = {}
            for skill_id, prob in skill_probs.items():
                topic_id = skill_to_topic.get(skill_id)
                if topic_id:
                    mastery_scores[topic_id] = max(mastery_scores.get(topic_id, 0.0), prob)
        else:
            # Without mapping, return raw skill probabilities (limited utility)
            mastery_scores = {str(k): v for k, v in skill_probs.items()}

        mastery = self._build_mastery_vector(mastery_scores)
        return self._run(mastery)

    def what_if(self, topic_id: str) -> dict[str, Any]:
        """
        What-if analysis: what unlocks if a student masters this topic?

        Parameters
        ----------
        topic_id : str

        Returns
        -------
        dict with direct_unlocks, all_unlocks, blocked_by, total_unlocked
        """
        graph = self.curriculum.graph
        direct = self.curriculum.successors(topic_id)
        all_unlocks = self.curriculum.descendants(topic_id)
        blocked_by = self.curriculum.prerequisites(topic_id)

        return {
            "topic_id": topic_id,
            "topic_label": self.curriculum.label(topic_id),
            "direct_unlocks": [
                {"id": n, "label": self.curriculum.label(n)} for n in direct
            ],
            "all_unlocks": [
                {"id": n, "label": self.curriculum.label(n)} for n in all_unlocks
            ],
            "blocked_by": [
                {"id": n, "label": self.curriculum.label(n)} for n in blocked_by
            ],
            "total_unlocked": len(all_unlocks),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_mastery_vector(self, mastery_scores: dict[str, float]) -> MasteryVector:
        mv = MasteryVector(self.curriculum, self.threshold, self.soft_threshold)
        mv.update_batch(mastery_scores)
        return mv

    def _run(self, mastery: MasteryVector) -> dict[str, Any]:
        constraint_results = self.constraint_layer.validate_all(mastery)
        ranked = self.ranker.rank(constraint_results, mastery, top_n=self.top_n)
        ranked["mastery_summary"] = mastery.summary()

        # Serialise to plain dicts for API/JSON friendliness
        for key in ("approved", "challenging", "vetoed"):
            ranked[key] = [self._rec_to_dict(r) for r in ranked[key]]

        return ranked

    def _load_model(self, path: str | Path) -> None:
        from plrs.model.sakt import SAKTModel
        self._model = SAKTModel.load(path, device=self.device)

    @staticmethod
    def _rec_to_dict(rec: RankedRecommendation) -> dict[str, Any]:
        return {
            "topic_id": rec.topic_id,
            "topic_label": rec.topic_label,
            "status": rec.status,
            "mastery": rec.mastery,
            "score": rec.score,
            "reasoning": rec.reasoning,
            "prerequisites": rec.prerequisites,
            "unmet_prerequisites": rec.unmet_prerequisites,
            "downstream_count": rec.downstream_count,
            "score_breakdown": rec.score_breakdown,
        }
