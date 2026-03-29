"""
tests/test_core.py
==================
Core unit tests for PLRS pipeline components.
Run with: pytest tests/ -v
"""

import json
import tempfile
from pathlib import Path

import pytest

# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

SAMPLE_DAG = {
    "domain": "Test Mathematics",
    "nodes": [
        {"id": "whole_numbers", "label": "Whole Numbers", "level": "JSS1"},
        {"id": "algebra_basics", "label": "Algebra Basics", "level": "JSS2"},
        {"id": "quadratic_equations", "label": "Quadratic Equations", "level": "SS1"},
        {"id": "calculus_intro", "label": "Introduction to Calculus", "level": "SS3"},
    ],
    "edges": [
        {"from": "whole_numbers", "to": "algebra_basics"},
        {"from": "algebra_basics", "to": "quadratic_equations"},
        {"from": "quadratic_equations", "to": "calculus_intro"},
    ],
}


@pytest.fixture
def dag_path(tmp_path):
    p = tmp_path / "test_dag.json"
    p.write_text(json.dumps(SAMPLE_DAG))
    return p


@pytest.fixture
def curriculum(dag_path):
    from plrs.curriculum.loader import load_dag
    return load_dag(dag_path)


@pytest.fixture
def pipeline(curriculum):
    from plrs.pipeline import PLRSPipeline
    return PLRSPipeline(curriculum, threshold=0.70, soft_threshold=0.50)


# ------------------------------------------------------------------ #
# Curriculum loader tests                                             #
# ------------------------------------------------------------------ #

class TestCurriculumLoader:
    def test_loads_correctly(self, curriculum):
        assert curriculum.domain == "Test Mathematics"
        assert curriculum.num_nodes == 4
        assert curriculum.num_edges == 3

    def test_labels(self, curriculum):
        assert curriculum.label("whole_numbers") == "Whole Numbers"

    def test_prerequisites(self, curriculum):
        assert curriculum.prerequisites("algebra_basics") == ["whole_numbers"]
        assert curriculum.prerequisites("whole_numbers") == []

    def test_successors(self, curriculum):
        assert curriculum.successors("whole_numbers") == ["algebra_basics"]

    def test_descendants(self, curriculum):
        desc = curriculum.descendants("whole_numbers")
        assert set(desc) == {"algebra_basics", "quadratic_equations", "calculus_intro"}

    def test_validation_clean_dag(self, curriculum):
        warnings = curriculum.validate()
        assert warnings == []

    def test_missing_nodes_raises(self, tmp_path):
        from plrs.curriculum.loader import load_dag
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"edges": []}))
        with pytest.raises(ValueError, match="nodes"):
            load_dag(bad)

    def test_file_not_found_raises(self):
        from plrs.curriculum.loader import load_dag
        with pytest.raises(FileNotFoundError):
            load_dag("/nonexistent/path.json")


# ------------------------------------------------------------------ #
# MasteryVector tests                                                  #
# ------------------------------------------------------------------ #

class TestMasteryVector:
    def test_initialises_to_zero(self, curriculum):
        from plrs.constraints.dag import MasteryVector
        mv = MasteryVector(curriculum)
        for node in curriculum.nodes:
            assert mv.get(node) == 0.0

    def test_update(self, curriculum):
        from plrs.constraints.dag import MasteryVector
        mv = MasteryVector(curriculum)
        mv.update("whole_numbers", 0.85)
        assert mv.get("whole_numbers") == 0.85

    def test_clamp(self, curriculum):
        from plrs.constraints.dag import MasteryVector
        mv = MasteryVector(curriculum)
        mv.update("whole_numbers", 1.5)
        assert mv.get("whole_numbers") == 1.0
        mv.update("whole_numbers", -0.5)
        assert mv.get("whole_numbers") == 0.0

    def test_is_mastered(self, curriculum):
        from plrs.constraints.dag import MasteryVector
        mv = MasteryVector(curriculum, threshold=0.70)
        mv.update("whole_numbers", 0.75)
        assert mv.is_mastered("whole_numbers") is True
        mv.update("algebra_basics", 0.65)
        assert mv.is_mastered("algebra_basics") is False

    def test_summary(self, curriculum):
        from plrs.constraints.dag import MasteryVector
        mv = MasteryVector(curriculum)
        mv.update("whole_numbers", 0.90)
        mv.update("algebra_basics", 0.55)
        s = mv.summary()
        assert s["mastered"] == 1
        assert s["partial"] == 1
        assert s["total_topics"] == 4


# ------------------------------------------------------------------ #
# DAG Constraint Layer tests                                           #
# ------------------------------------------------------------------ #

class TestDAGConstraintLayer:
    def _mastery(self, curriculum, scores):
        from plrs.constraints.dag import MasteryVector
        mv = MasteryVector(curriculum)
        mv.update_batch(scores)
        return mv

    def test_no_prereqs_approved(self, curriculum):
        from plrs.constraints.dag import DAGConstraintLayer
        layer = DAGConstraintLayer(curriculum)
        mv = self._mastery(curriculum, {})
        result = layer.validate("whole_numbers", mv)
        assert result.status == "approved"

    def test_prereq_met_approved(self, curriculum):
        from plrs.constraints.dag import DAGConstraintLayer
        layer = DAGConstraintLayer(curriculum)
        mv = self._mastery(curriculum, {"whole_numbers": 0.80})
        result = layer.validate("algebra_basics", mv)
        assert result.status == "approved"

    def test_prereq_partial_challenging(self, curriculum):
        from plrs.constraints.dag import DAGConstraintLayer
        layer = DAGConstraintLayer(curriculum)
        mv = self._mastery(curriculum, {"whole_numbers": 0.60})  # above soft(0.50), below threshold(0.70)
        result = layer.validate("algebra_basics", mv)
        assert result.status == "challenging"

    def test_prereq_missing_vetoed(self, curriculum):
        from plrs.constraints.dag import DAGConstraintLayer
        layer = DAGConstraintLayer(curriculum)
        mv = self._mastery(curriculum, {"whole_numbers": 0.20})  # below soft threshold
        result = layer.validate("algebra_basics", mv)
        assert result.status == "vetoed"

    def test_zero_violation_when_all_mastered(self, curriculum):
        from plrs.constraints.dag import DAGConstraintLayer, MasteryVector
        layer = DAGConstraintLayer(curriculum)
        mv = MasteryVector(curriculum)
        mv.update_batch({n: 0.90 for n in curriculum.nodes})
        results = layer.validate_all(mv)
        vetoed = [r for r in results if r.status == "vetoed"]
        assert len(vetoed) == 0


# ------------------------------------------------------------------ #
# PLRSPipeline tests                                                   #
# ------------------------------------------------------------------ #

class TestPLRSPipeline:
    def test_recommend_from_mastery_returns_structure(self, pipeline):
        results = pipeline.recommend_from_mastery({
            "whole_numbers": 0.90,
            "algebra_basics": 0.80,
        })
        assert "approved" in results
        assert "challenging" in results
        assert "vetoed" in results
        assert "stats" in results
        assert "mastery_summary" in results

    def test_top_n_respected(self, pipeline):
        pipeline.top_n = 2
        results = pipeline.recommend_from_mastery({"whole_numbers": 0.90})
        assert len(results["approved"]) <= 2

    def test_what_if(self, pipeline):
        result = pipeline.what_if("whole_numbers")
        assert result["topic_id"] == "whole_numbers"
        assert result["total_unlocked"] == 3  # algebra, quadratic, calculus
        assert len(result["direct_unlocks"]) == 1

    def test_no_model_history_raises(self, pipeline):
        with pytest.raises(RuntimeError, match="No model loaded"):
            pipeline.recommend_from_history([1, 2, 3], [1, 0, 1])
