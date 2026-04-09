"""
plrs.ranking.spaced_repetition
===============================
SuperMemo-2 spaced repetition scoring for the PLRS ranker.

The SM-2 algorithm computes an optimal review interval for each topic based on:
  - Performance quality (derived from mastery probability)
  - Ease factor (how difficult the topic is for this student)
  - Number of successful repetitions

Topics that are OVERDUE for review get a high spaced repetition score,
boosting them in the ranker. Topics recently reviewed or never seen get lower scores.

Reference: Wozniak (1990) — "Optimization of Learning"
           https://www.supermemo.com/en/archives1990-2015/english/ol/sm2

Integration with PLRS
---------------------
Since PLRS uses mastery probabilities (not discrete grades), we map:
    mastery probability → SM-2 quality score (0–5)

    mastery < 0.30  → quality = 1  (complete failure)
    mastery < 0.50  → quality = 2  (incorrect, but easy recall)
    mastery < 0.60  → quality = 3  (correct with serious difficulty)
    mastery < 0.75  → quality = 4  (correct with hesitation)
    mastery < 0.90  → quality = 5  (correct with slight hesitation)
    mastery >= 0.90 → quality = 5  (perfect)

The "steps since last seen" maps to elapsed time:
    If no timestamp data: use interaction_step / avg_interactions_per_day
    Default: assume 10 interactions/day (conservative for OULAD data)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── SM-2 constants ────────────────────────────────────────────────────────────

SM2_MIN_EASE    = 1.3   # minimum ease factor (SM-2 spec)
SM2_INIT_EASE   = 2.5   # initial ease factor for new topics
SM2_EASE_DELTA  = 0.1   # ease factor adjustment step


# ── Quality mapping ───────────────────────────────────────────────────────────

def mastery_to_quality(mastery: float) -> int:
    """
    Map a mastery probability [0, 1] to SM-2 quality score [0, 5].

    Quality scores:
        0–2 : incorrect response (review immediately)
        3   : correct but difficult
        4   : correct with hesitation
        5   : perfect recall
    """
    if mastery < 0.30:
        return 1
    if mastery < 0.50:
        return 2
    if mastery < 0.60:
        return 3
    if mastery < 0.75:
        return 4
    return 5


# ── SM-2 state per topic ──────────────────────────────────────────────────────

@dataclass
class SM2State:
    """Tracks SM-2 state for a single topic."""
    topic_id:          str
    repetitions:       int         = 0              # number of successful reviews
    ease_factor:       float       = SM2_INIT_EASE
    interval_days:     float       = 1.0            # current optimal interval in days
    last_reviewed_day: float|None  = None           # None = never reviewed

    def next_interval(self, quality: int) -> float:
        """
        Compute the next review interval using SM-2 formula.

        Returns the new interval in days.
        """
        if quality < 3:
            # Incorrect — reset repetitions, start over
            self.repetitions = 0
            self.interval_days = 1.0
        else:
            if self.repetitions == 0:
                self.interval_days = 1.0
            elif self.repetitions == 1:
                self.interval_days = 6.0
            else:
                self.interval_days = round(self.interval_days * self.ease_factor, 1)
            self.repetitions += 1

        # Update ease factor
        self.ease_factor = max(
            SM2_MIN_EASE,
            self.ease_factor + SM2_EASE_DELTA * (5 - quality) * (-0.8 + 0.28 * (5 - quality))
        )
        return self.interval_days

    def is_due(self, current_day: float) -> bool:
        """Return True if the topic is due for review."""
        if self.last_reviewed_day is None:
            return False
        return current_day >= self.last_reviewed_day + self.interval_days

    def overdue_ratio(self, current_day: float) -> float:
        """
        How overdue is this topic? Returns a ratio ≥ 0.

        0.0  = just reviewed
        1.0  = exactly at the optimal interval
        >1.0 = overdue (higher = more urgent)
        None reviewed → return 0.5 (neutral)
        """
        if self.last_reviewed_day is None:
            # Never reviewed — neutral
            return 0.5
        elapsed = current_day - self.last_reviewed_day
        return elapsed / max(self.interval_days, 1.0)


# ── Spaced Repetition Scorer ──────────────────────────────────────────────────

@dataclass
class SpacedRepetitionScorer:
    """
    Computes spaced repetition urgency scores for all curriculum topics.

    Parameters
    ----------
    interactions_per_day : float
        Assumed interaction rate (default 10 per day). Used to convert
        interaction steps to simulated days when no timestamp data is available.
    """
    interactions_per_day: float = 10.0
    _states: dict[str, SM2State] = field(default_factory=dict)
    _current_step: int = 0

    @property
    def current_day(self) -> float:
        return self._current_step / self.interactions_per_day

    def update(
        self,
        topic_id: str,
        mastery: float,
        step: int | None = None,
        day: float | None = None,
    ) -> None:
        """
        Update SM-2 state for a topic after a student interaction.

        Parameters
        ----------
        topic_id : str
        mastery : float
            Current mastery probability for this topic [0, 1].
        step : int, optional
            Interaction step number (used if no day timestamp available).
        day : float, optional
            Actual day number (overrides step-based estimation).
        """
        if step is not None:
            self._current_step = step
        current_day = day if day is not None else self.current_day

        if topic_id not in self._states:
            self._states[topic_id] = SM2State(topic_id=topic_id)

        state = self._states[topic_id]
        quality = mastery_to_quality(mastery)
        state.next_interval(quality)
        state.last_reviewed_day = float(current_day)

    def score(self, topic_id: str, mastery: float) -> float:
        """
        Compute spaced repetition urgency score for a topic [0, 1].

        Topics that are overdue for review score high.
        Topics never seen score 0.5 (neutral).
        Topics recently reviewed score low.

        Parameters
        ----------
        topic_id : str
        mastery : float
            Current mastery probability (used to init state if not seen before).

        Returns
        -------
        float in [0, 1]
        """
        if topic_id not in self._states:
            # Never interacted with — neutral score
            return 0.5

        state = self._states[topic_id]
        ratio = state.overdue_ratio(self.current_day)

        # Sigmoid-like mapping: ratio → [0, 1]
        # ratio=0   → score≈0.1  (just reviewed)
        # ratio=0.5 → score≈0.5  (never seen)
        # ratio=1.0 → score≈0.73 (exactly due)
        # ratio=2.0 → score≈0.88 (2x overdue)
        # ratio=5.0 → score≈0.97 (very overdue)
        score = 1.0 / (1.0 + math.exp(-2.0 * (ratio - 0.5)))
        return round(score, 4)

    def build_from_mastery(
        self,
        mastery_dict: dict[str, float],
        current_step: int = 100,
    ) -> None:
        """
        Bootstrap SM-2 states from a mastery dictionary when no interaction
        history is available.

        Higher mastery → assume reviewed more recently.
        Lower mastery → assume reviewed less recently (more urgent).

        Parameters
        ----------
        mastery_dict : dict[str, float]
            topic_id → mastery probability
        current_step : int
            Simulated current step (default 100 = ~10 days of activity)
        """
        self._current_step = current_step

        for topic_id, mastery in mastery_dict.items():
            if mastery <= 0.0:
                # Never seen — leave state uninitialised for neutral score
                continue

            state = SM2State(topic_id=topic_id)
            quality = mastery_to_quality(mastery)

            # Simulate number of successful repetitions based on mastery
            reps = max(0, int(mastery * 5))
            for _ in range(reps):
                state.next_interval(quality)

            # Last reviewed: higher mastery → reviewed more recently
            # mastery=1.0 → last reviewed at current_step (0 days ago)
            # mastery=0.3 → last reviewed long ago (many intervals ago)
            days_since = state.interval_days * (1.0 - mastery) * 3
            state.last_reviewed_day = max(0, self.current_day - days_since)

            self._states[topic_id] = state

    def get_all_scores(self, mastery_dict: dict[str, float]) -> dict[str, float]:
        """Return spaced repetition scores for all topics in mastery_dict."""
        return {
            topic_id: self.score(topic_id, mastery)
            for topic_id, mastery in mastery_dict.items()
        }

    def summary(self) -> dict:
        """Return summary of SM-2 states."""
        if not self._states:
            return {"tracked_topics": 0}
        overdue = sum(
            1 for s in self._states.values()
            if s.is_due(self.current_day)
        )
        avg_interval = sum(s.interval_days for s in self._states.values()) / len(self._states)
        return {
            "tracked_topics": len(self._states),
            "overdue_topics": overdue,
            "avg_interval_days": round(avg_interval, 2),
            "current_day": round(self.current_day, 1),
        }
