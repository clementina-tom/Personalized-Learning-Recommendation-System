"""
plrs.assess.diagnostic
=======================
Cold start diagnostic assessment for new students.

Problem: SAKT needs interaction history to produce meaningful mastery predictions.
New students have none. This module bootstraps the mastery vector via a short
adaptive assessment (10-15 questions) before handing off to SAKT.

Algorithm:
  1. Start with root topics (no prerequisites)
  2. Select next question based on current mastery estimate (binary search on difficulty)
  3. Update estimate after each answer (Bayesian update, BKT-style)
  4. Stop when confidence is high enough or max questions reached
  5. Return mastery dict suitable for PLRSPipeline.recommend_from_mastery()

REST endpoints (in assess_router.py):
  POST /assess/start           — begin session, get first question
  POST /assess/answer          — submit answer, get next question or completion
  GET  /assess/{session_id}    — get session state
  POST /assess/{session_id}/complete  — force completion, get mastery vector
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ── Question bank ─────────────────────────────────────────────────────────────

@dataclass
class DiagnosticQuestion:
    """A single diagnostic question."""
    question_id:  str
    topic_id:     str
    topic_label:  str
    text:         str
    options:      list[str]          # A, B, C, D
    correct_idx:  int                # 0-based index into options
    difficulty:   float              # 0.0 (easy) → 1.0 (hard)
    explanation:  str = ""


def build_question_bank(curriculum_nodes: list[dict]) -> list[DiagnosticQuestion]:
    """
    Generate a minimal question bank from curriculum node metadata.

    In production: load from a curated JSON file or database.
    This generates synthetic questions for any curriculum — suitable for
    bootstrapping before a curated bank is available.
    """
    questions: list[DiagnosticQuestion] = []

    for node in curriculum_nodes:
        topic_id    = node["id"]
        topic_label = node.get("label", topic_id)
        level       = node.get("level", "")

        # Difficulty based on how deep the node is (root = easy, leaf = hard)
        prereq_count = len(node.get("prerequisites", []))
        difficulty   = min(0.9, 0.2 + prereq_count * 0.15)

        questions.append(DiagnosticQuestion(
            question_id=f"q_{topic_id}",
            topic_id=topic_id,
            topic_label=topic_label,
            text=f"Which of the following best describes '{topic_label}'?",
            options=[
                f"A core concept in {topic_label}",
                f"An unrelated concept",
                f"A prerequisite of {topic_label}",
                f"An application of {topic_label}",
            ],
            correct_idx=0,
            difficulty=difficulty,
            explanation=f"'{topic_label}' is a topic in the {level} curriculum.",
        ))

    return questions


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class AssessmentSession:
    """
    State for a single student's diagnostic assessment session.

    Parameters
    ----------
    session_id : str
    domain : str
    questions : list[DiagnosticQuestion]
    max_questions : int
    confidence_threshold : float
        Stop early if mastery estimates are confident enough.
    """
    session_id:           str
    domain:               str
    questions:            list[DiagnosticQuestion]
    max_questions:        int   = 12
    confidence_threshold: float = 0.80

    # Mutable state
    answered:         list[dict]        = field(default_factory=list)
    mastery_estimate: dict[str, float]  = field(default_factory=dict)
    confidence:       dict[str, float]  = field(default_factory=dict)
    asked_topics:     set[str]          = field(default_factory=set)
    started_at:       float             = field(default_factory=time.time)
    completed:        bool              = False
    completed_at:     float | None      = None

    def __post_init__(self):
        # Initialise all topics to 0.5 mastery (maximum uncertainty)
        for q in self.questions:
            if q.topic_id not in self.mastery_estimate:
                self.mastery_estimate[q.topic_id] = 0.5
                self.confidence[q.topic_id]       = 0.0

    @property
    def num_answered(self) -> int:
        return len(self.answered)

    @property
    def is_complete(self) -> bool:
        if self.completed:
            return True
        if self.num_answered >= self.max_questions:
            return True
        # Early stop: all asked topics have high confidence
        if self.asked_topics and all(
            self.confidence.get(t, 0) >= self.confidence_threshold
            for t in self.asked_topics
        ):
            return True
        return False

    @property
    def elapsed_seconds(self) -> float:
        end = self.completed_at or time.time()
        return round(end - self.started_at, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id":      self.session_id,
            "domain":          self.domain,
            "num_answered":    self.num_answered,
            "max_questions":   self.max_questions,
            "completed":       self.is_complete,
            "elapsed_seconds": self.elapsed_seconds,
            "mastery_summary": {
                "topics_assessed": len(self.asked_topics),
                "avg_mastery":     round(
                    sum(self.mastery_estimate.get(t, 0) for t in self.asked_topics)
                    / max(len(self.asked_topics), 1), 3
                ),
            },
        }


# ── Adaptive engine ───────────────────────────────────────────────────────────

class DiagnosticEngine:
    """
    Adaptive question selection and mastery estimation engine.

    Uses a simple Bayesian Knowledge Tracing update for mastery estimation.
    Question selection targets topics with highest uncertainty (confidence near 0).
    """

    # BKT parameters
    P_LEARN  = 0.20   # probability of learning between questions
    P_SLIP   = 0.10   # probability of wrong answer despite knowing
    P_GUESS  = 0.20   # probability of correct answer despite not knowing

    def __init__(self, curriculum) -> None:
        """
        Parameters
        ----------
        curriculum : CurriculumGraph
        """
        self.curriculum = curriculum

        # Build question bank from curriculum nodes
        nodes = [
            {
                "id":            node,
                "label":         curriculum.label(node),
                "level":         curriculum.level(node),
                "prerequisites": curriculum.prerequisites(node),
            }
            for node in curriculum.nodes
        ]
        self.question_bank: dict[str, DiagnosticQuestion] = {
            q.topic_id: q for q in build_question_bank(nodes)
        }

    def start_session(
        self,
        domain: str,
        max_questions: int = 12,
        confidence_threshold: float = 0.80,
    ) -> AssessmentSession:
        """Create a new assessment session."""
        session = AssessmentSession(
            session_id=str(uuid.uuid4()),
            domain=domain,
            questions=list(self.question_bank.values()),
            max_questions=max_questions,
            confidence_threshold=confidence_threshold,
        )
        return session

    def next_question(
        self,
        session: AssessmentSession,
    ) -> DiagnosticQuestion | None:
        """
        Select the next question to ask.

        Strategy:
          1. Prefer root topics (no prerequisites) first
          2. Among available topics, pick the one with lowest confidence
          3. Skip already-asked topics
          4. Skip topics whose prerequisites haven't been assessed yet
        """
        if session.is_complete:
            return None

        available = [
            q for q in self.question_bank.values()
            if q.topic_id not in session.asked_topics
        ]

        if not available:
            return None

        # Prefer topics with all prerequisites already assessed
        with_prereqs_done = [
            q for q in available
            if all(
                p in session.asked_topics
                for p in self.curriculum.prerequisites(q.topic_id)
            )
        ]

        candidates = with_prereqs_done or available

        # Sort by lowest confidence (most uncertain) first
        candidates.sort(key=lambda q: session.confidence.get(q.topic_id, 0.0))

        return candidates[0]

    def record_answer(
        self,
        session: AssessmentSession,
        question: DiagnosticQuestion,
        chosen_idx: int,
        time_ms: int | None = None,
    ) -> dict[str, Any]:
        """
        Record a student's answer and update mastery estimates.

        Parameters
        ----------
        session : AssessmentSession
        question : DiagnosticQuestion
        chosen_idx : int — 0-based index of chosen option
        time_ms : int, optional — response time in milliseconds

        Returns
        -------
        dict with result and updated estimates
        """
        correct = (chosen_idx == question.correct_idx)
        topic   = question.topic_id

        session.asked_topics.add(topic)

        # BKT update
        p_prior = session.mastery_estimate.get(topic, 0.5)

        if correct:
            p_posterior = (p_prior * (1 - self.P_SLIP)) / (
                p_prior * (1 - self.P_SLIP) + (1 - p_prior) * self.P_GUESS
            )
        else:
            p_posterior = (p_prior * self.P_SLIP) / (
                p_prior * self.P_SLIP + (1 - p_prior) * (1 - self.P_GUESS)
            )

        # Learning update
        p_posterior = p_posterior + (1 - p_posterior) * self.P_LEARN
        p_posterior = max(0.05, min(0.95, p_posterior))

        session.mastery_estimate[topic] = round(p_posterior, 4)

        # Confidence: how far from 0.5 (certainty grows with each answer)
        session.confidence[topic] = round(abs(p_posterior - 0.5) * 2, 4)

        # Cascade mastery to downstream topics (partial signal)
        for successor in self.curriculum.successors(topic):
            if successor not in session.asked_topics:
                # If student knows A, give partial credit for A's successors
                current = session.mastery_estimate.get(successor, 0.5)
                if correct:
                    session.mastery_estimate[successor] = min(current + 0.08, 0.65)
                # If student doesn't know A, successors probably unknown too
                else:
                    session.mastery_estimate[successor] = max(current - 0.05, 0.15)

        # Record answer
        session.answered.append({
            "question_id":  question.question_id,
            "topic_id":     topic,
            "correct":      correct,
            "chosen_idx":   chosen_idx,
            "correct_idx":  question.correct_idx,
            "time_ms":      time_ms,
            "mastery_after": p_posterior,
        })

        return {
            "correct":        correct,
            "correct_answer": question.options[question.correct_idx],
            "explanation":    question.explanation,
            "mastery_after":  round(p_posterior, 3),
            "confidence":     round(session.confidence[topic], 3),
        }

    def finalise(self, session: AssessmentSession) -> dict[str, float]:
        """
        Finalise the session and return a mastery dict.

        Fills in unasked topics using prerequisite propagation:
        - Topics whose prerequisites are all mastered → partial mastery
        - Topics with no signal → 0.5 (unknown)

        Returns
        -------
        dict[str, float] — topic_id → mastery probability, ready for PLRSPipeline
        """
        session.completed    = True
        session.completed_at = time.time()

        mastery = dict(session.mastery_estimate)

        # Fill unasked topics using prerequisite signal
        for node in self.curriculum.nodes:
            if node in session.asked_topics:
                continue
            prereqs = self.curriculum.prerequisites(node)
            if not prereqs:
                # Root topic — leave as uncertain
                mastery.setdefault(node, 0.5)
                continue

            # Estimate from prerequisites
            prereq_mastery = [mastery.get(p, 0.5) for p in prereqs]
            avg_prereq = sum(prereq_mastery) / len(prereq_mastery)

            # If prerequisites suggest mastery, bump topic slightly
            if avg_prereq >= 0.70:
                mastery[node] = min(mastery.get(node, 0.3) + 0.1, 0.6)
            elif avg_prereq < 0.40:
                mastery[node] = max(mastery.get(node, 0.5) - 0.1, 0.1)
            else:
                mastery.setdefault(node, 0.4)

        return {k: round(v, 4) for k, v in mastery.items()}


# ── In-memory session store ───────────────────────────────────────────────────

class SessionStore:
    """In-memory session store. Replace with Redis for production."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._sessions: dict[str, tuple[AssessmentSession, float]] = {}
        self._ttl = ttl_seconds

    def save(self, session: AssessmentSession) -> None:
        self._sessions[session.session_id] = (session, time.time())

    def get(self, session_id: str) -> AssessmentSession | None:
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        session, created_at = entry
        if time.time() - created_at > self._ttl:
            del self._sessions[session_id]
            return None
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)


# ── Global instances ──────────────────────────────────────────────────────────

_engines: dict[str, DiagnosticEngine] = {}
_session_store = SessionStore()


def register_engine(domain: str, engine: DiagnosticEngine) -> None:
    _engines[domain] = engine


def get_engine(domain: str) -> DiagnosticEngine | None:
    return _engines.get(domain)


def get_session_store() -> SessionStore:
    return _session_store
