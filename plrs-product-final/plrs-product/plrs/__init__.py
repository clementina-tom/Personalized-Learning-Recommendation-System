"""
PLRS — Personalized Learning Recommendation System
====================================================
Constraint-aware personalized learning recommendations.
Plug in your curriculum DAG, get intelligent recommendations out.

Quick start:
    from plrs import PLRSPipeline
    from plrs.curriculum import load_dag

    graph    = load_dag("my_curriculum.json")
    pipeline = PLRSPipeline(graph)
    results  = pipeline.recommend(student_history)
"""

from plrs.pipeline import PLRSPipeline
from plrs.model.sakt import SAKTModel
from plrs.constraints.dag import DAGConstraintLayer
from plrs.ranking.ranker import MultiObjectiveRanker
from plrs.curriculum.loader import load_dag, CurriculumGraph

__version__ = "0.1.0"
__all__ = [
    "PLRSPipeline",
    "SAKTModel",
    "DAGConstraintLayer",
    "MultiObjectiveRanker",
    "load_dag",
    "CurriculumGraph",
]
