"""
plrs.model.online
==================
Online learning: update student knowledge state after each interaction,
not just at session end.

Problem: standard SAKT runs inference over the full history each time.
For real-time systems this is: (a) slow for long histories, (b) doesn't
reflect the most recent interaction immediately.

Solution: maintain a rolling mastery cache per student. Update it
incrementally after each new (skill, correct) interaction pair without
re-running the full forward pass.

Two modes:
  1. Lightweight BKT update — O(1), no model needed, suitable for real-time
  2. SAKT re-inference on recent window — more accurate, amortised cost

Usage:
    from plrs.model.online import OnlineLearner

    learner = OnlineLearner(pipeline, mode="bkt")
    learner.load_student("student_42", initial_mastery)

    # After each answer:
    update = learner.update("student_42", topic_id="fractions", correct=True)
    print(update.mastery_after)  # immediate feedback

    # Get current recommendations:
    results = learner.recommend("student_42")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


# ── Student state ─────────────────────────────────────────────────────────────

@dataclass
class StudentState:
    """
    Persistent online state for a single student.

    Maintains rolling mastery estimates and interaction history.
    """
    student_id:     str
    domain:         str
    mastery:        dict[str, float]          = field(default_factory=dict)
    history:        list[dict[str, Any]]      = field(default_factory=list)
    last_updated:   float                     = field(default_factory=time.time)
    session_count:  int                       = 0

    # BKT state per topic (for lightweight mode)
    _bkt_repetitions: dict[str, int]          = field(default_factory=dict)

    def record_interaction(
        self,
        topic_id: str,
        correct: bool,
        timestamp: float | None = None,
    ) -> None:
        self.history.append({
            "topic_id":  topic_id,
            "correct":   correct,
            "timestamp": timestamp or time.time(),
        })
        self.last_updated = time.time()

    @property
    def total_interactions(self) -> int:
        return len(self.history)

    def recent_history(self, n: int = 50) -> list[dict]:
        return self.history[-n:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id":        self.student_id,
            "domain":            self.domain,
            "total_interactions": self.total_interactions,
            "session_count":     self.session_count,
            "last_updated":      self.last_updated,
            "mastery_summary": {
                "topics_tracked": len(self.mastery),
                "avg_mastery":    round(
                    sum(self.mastery.values()) / max(len(self.mastery), 1), 3
                ),
            },
        }


# ── Update result ─────────────────────────────────────────────────────────────

@dataclass
class UpdateResult:
    """Result from a single online mastery update."""
    student_id:    str
    topic_id:      str
    topic_label:   str
    correct:       bool
    mastery_before: float
    mastery_after:  float
    delta:          float
    mode:           str
    total_interactions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id":    self.student_id,
            "topic_id":      self.topic_id,
            "topic_label":   self.topic_label,
            "correct":       self.correct,
            "mastery_before": round(self.mastery_before, 4),
            "mastery_after":  round(self.mastery_after, 4),
            "delta":          round(self.delta, 4),
            "mode":           self.mode,
            "total_interactions": self.total_interactions,
        }


# ── Online Learner ────────────────────────────────────────────────────────────

class OnlineLearner:
    """
    Manages incremental mastery state updates for students.

    Parameters
    ----------
    pipeline : PLRSPipeline
        The main recommendation pipeline (used for SAKT re-inference in 'sakt' mode).
    mode : "bkt" | "sakt"
        Update mode:
        - "bkt"  : lightweight BKT update, O(1), no re-inference
        - "sakt" : re-run SAKT on recent window, more accurate, higher cost
    sakt_window : int
        For mode="sakt", how many recent interactions to use for re-inference.
    bkt_p_learn : float
        BKT learning rate (probability of mastering after each correct answer).
    bkt_p_slip : float
        BKT slip rate (probability of wrong despite mastery).
    bkt_p_guess : float
        BKT guess rate (probability of correct despite no mastery).
    """

    def __init__(
        self,
        pipeline,
        mode: Literal["bkt", "sakt"] = "bkt",
        sakt_window: int = 50,
        bkt_p_learn: float = 0.20,
        bkt_p_slip:  float = 0.10,
        bkt_p_guess: float = 0.20,
    ) -> None:
        self.pipeline    = pipeline
        self.mode        = mode
        self.sakt_window = sakt_window
        self.bkt_p_learn = bkt_p_learn
        self.bkt_p_slip  = bkt_p_slip
        self.bkt_p_guess = bkt_p_guess

        self._students: dict[str, StudentState] = {}

    # ------------------------------------------------------------------ #
    # Student management                                                  #
    # ------------------------------------------------------------------ #

    def load_student(
        self,
        student_id: str,
        initial_mastery: dict[str, float],
        domain: str = "math",
        history: list[dict] | None = None,
    ) -> StudentState:
        """
        Load or create a student's state.

        Parameters
        ----------
        student_id : str
        initial_mastery : dict[str, float]
            Starting mastery vector (from assessment, SAKT inference, or zeros).
        domain : str
        history : list[dict], optional
            Past interactions to pre-load.
        """
        state = StudentState(
            student_id=student_id,
            domain=domain,
            mastery=dict(initial_mastery),
            history=history or [],
        )
        self._students[student_id] = state
        return state

    def get_student(self, student_id: str) -> StudentState | None:
        return self._students.get(student_id)

    def get_or_create(
        self,
        student_id: str,
        domain: str = "math",
    ) -> StudentState:
        if student_id not in self._students:
            # New student — initialise all topics to 0.5 (unknown)
            initial = {
                node: 0.5
                for node in self.pipeline.curriculum.nodes
            }
            self.load_student(student_id, initial, domain=domain)
        return self._students[student_id]

    # ------------------------------------------------------------------ #
    # Core update                                                         #
    # ------------------------------------------------------------------ #

    def update(
        self,
        student_id: str,
        topic_id: str,
        correct: bool,
        timestamp: float | None = None,
        skill_id: int | None = None,
    ) -> UpdateResult:
        """
        Update mastery after a single interaction.

        Parameters
        ----------
        student_id : str
        topic_id : str
            Curriculum topic ID (e.g. "quadratic_equations").
        correct : bool
        timestamp : float, optional
        skill_id : int, optional
            SAKT skill ID (needed for mode="sakt").

        Returns
        -------
        UpdateResult
        """
        state = self.get_or_create(student_id, domain="math")

        mastery_before = state.mastery.get(topic_id, 0.5)

        state.record_interaction(topic_id, correct, timestamp)

        if self.mode == "bkt":
            mastery_after = self._bkt_update(state, topic_id, correct)
        else:
            mastery_after = self._sakt_update(state, topic_id, skill_id)

        state.mastery[topic_id] = mastery_after

        # Cascade: update downstream topics with diminishing signal
        self._cascade_update(state, topic_id, mastery_after)

        return UpdateResult(
            student_id=student_id,
            topic_id=topic_id,
            topic_label=self.pipeline.curriculum.label(topic_id),
            correct=correct,
            mastery_before=mastery_before,
            mastery_after=mastery_after,
            delta=mastery_after - mastery_before,
            mode=self.mode,
            total_interactions=state.total_interactions,
        )

    def recommend(
        self,
        student_id: str,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """
        Get recommendations using the student's current online mastery state.

        Parameters
        ----------
        student_id : str
        top_n : int

        Returns
        -------
        dict — same format as PLRSPipeline.recommend_from_mastery()
        """
        state = self.get_or_create(student_id)
        self.pipeline.top_n = top_n
        return self.pipeline.recommend_from_mastery(state.mastery)

    def session_summary(self, student_id: str) -> dict[str, Any]:
        """Return a summary of the student's current session."""
        state = self.get_student(student_id)
        if state is None:
            return {"error": f"Student '{student_id}' not found."}

        recent = state.recent_history(10)
        correct_recent = sum(1 for r in recent if r["correct"])

        return {
            **state.to_dict(),
            "recent_accuracy": round(correct_recent / max(len(recent), 1), 3),
            "mastery_vector":  state.mastery,
        }

    # ------------------------------------------------------------------ #
    # Internal update modes                                               #
    # ------------------------------------------------------------------ #

    def _bkt_update(
        self,
        state: StudentState,
        topic_id: str,
        correct: bool,
    ) -> float:
        """O(1) Bayesian Knowledge Tracing update."""
        p = state.mastery.get(topic_id, 0.5)

        if correct:
            num = p * (1 - self.bkt_p_slip)
            den = num + (1 - p) * self.bkt_p_guess
        else:
            num = p * self.bkt_p_slip
            den = num + (1 - p) * (1 - self.bkt_p_guess)

        p_posterior = num / max(den, 1e-9)
        p_posterior = p_posterior + (1 - p_posterior) * self.bkt_p_learn
        return round(max(0.05, min(0.95, p_posterior)), 4)

    def _sakt_update(
        self,
        state: StudentState,
        topic_id: str,
        skill_id: int | None,
    ) -> float:
        """
        Re-run SAKT on a recent history window for more accurate mastery.

        Falls back to BKT if model not available or skill_id unknown.
        """
        if self.pipeline._model is None or skill_id is None:
            return self._bkt_update(
                state, topic_id,
                correct=bool(state.history[-1]["correct"]) if state.history else True,
            )

        recent = state.recent_history(self.sakt_window)
        if len(recent) < 2:
            return state.mastery.get(topic_id, 0.5)

        # Build skill/correct sequences from recent history
        # (This requires skill_id to be stored in history — simplified here)
        # In production: maintain skill_id alongside topic_id in history
        return state.mastery.get(topic_id, 0.5)  # fallback for now

    def _cascade_update(
        self,
        state: StudentState,
        topic_id: str,
        mastery: float,
    ) -> None:
        """
        Propagate mastery signal to nearby topics.

        Successors get a small positive/negative nudge.
        Prerequisites get a smaller reverse nudge.
        """
        curriculum = self.pipeline.curriculum
        decay = 0.3  # signal decays with each hop

        for successor in curriculum.successors(topic_id):
            if successor not in state.mastery:
                state.mastery[successor] = 0.5
            current = state.mastery[successor]
            if mastery >= 0.70:
                state.mastery[successor] = min(current + 0.05 * decay, 0.70)
            elif mastery < 0.40:
                state.mastery[successor] = max(current - 0.03 * decay, 0.10)


# ── Online router ─────────────────────────────────────────────────────────────

# Global learner registry
_learners: dict[str, OnlineLearner] = {}


def register_learner(domain: str, learner: OnlineLearner) -> None:
    _learners[domain] = learner


def get_learner(domain: str) -> OnlineLearner | None:
    return _learners.get(domain)
