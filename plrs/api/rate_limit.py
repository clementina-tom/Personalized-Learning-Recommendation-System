"""
plrs.api.rate_limit
====================
Sliding window rate limiter for the PLRS API.

Two windows per key:
  - Per-minute  : rolling 60-second window
  - Per-day     : rolling 86400-second window

In-memory implementation using deque (O(1) amortised per check).
Swap the backend for Redis in production (hosted version).

Usage:
    limiter = RateLimiter()
    ok, info = limiter.check("plrs_abc123", requests_per_minute=60, requests_per_day=1000)
    if not ok:
        raise HTTPException(429, detail=info)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


# ── Window sizes ──────────────────────────────────────────────────────────────

MINUTE_WINDOW = 60        # seconds
DAY_WINDOW    = 86_400    # seconds


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RateLimitResult:
    allowed:          bool
    limit_type:       str | None    # "minute" | "day" | None
    requests_minute:  int           # requests in last 60s
    requests_day:     int           # requests in last 24h
    limit_minute:     int
    limit_day:        int
    retry_after:      int           # seconds to wait (0 if allowed)

    @property
    def headers(self) -> dict[str, str]:
        """Headers to include in the HTTP response."""
        h = {
            "X-RateLimit-Limit-Minute":     str(self.limit_minute),
            "X-RateLimit-Limit-Day":        str(self.limit_day),
            "X-RateLimit-Remaining-Minute": str(max(0, self.limit_minute - self.requests_minute)),
            "X-RateLimit-Remaining-Day":    str(max(0, self.limit_day    - self.requests_day)),
        }
        if not self.allowed:
            h["Retry-After"] = str(self.retry_after)
        return h

    def to_dict(self) -> dict:
        return {
            "error":         "Rate limit exceeded",
            "limit_type":    self.limit_type,
            "requests_minute": self.requests_minute,
            "requests_day":  self.requests_day,
            "limit_minute":  self.limit_minute,
            "limit_day":     self.limit_day,
            "retry_after_s": self.retry_after,
        }


# ── Per-key sliding window state ──────────────────────────────────────────────

class _KeyWindow:
    """Sliding window counters for a single API key."""

    __slots__ = ("_minute", "_day")

    def __init__(self) -> None:
        self._minute: deque[float] = deque()  # timestamps in last 60s
        self._day:    deque[float] = deque()  # timestamps in last 24h

    def record(self, now: float) -> None:
        """Record a new request at timestamp `now`."""
        self._minute.append(now)
        self._day.append(now)

    def counts(self, now: float) -> tuple[int, int]:
        """
        Return (requests_in_last_minute, requests_in_last_day).
        Evicts stale entries as a side effect.
        """
        self._evict(now)
        return len(self._minute), len(self._day)

    def _evict(self, now: float) -> None:
        """Remove timestamps outside the windows."""
        minute_cutoff = now - MINUTE_WINDOW
        day_cutoff    = now - DAY_WINDOW
        while self._minute and self._minute[0] < minute_cutoff:
            self._minute.popleft()
        while self._day and self._day[0] < day_cutoff:
            self._day.popleft()

    def next_available(self, now: float, limit_minute: int, limit_day: int) -> int:
        """
        Seconds until the key is under the limit again.
        Returns 0 if already under limit.
        """
        self._evict(now)
        retry = 0

        if len(self._minute) >= limit_minute and self._minute:
            oldest_minute = self._minute[0]
            retry = max(retry, int(oldest_minute + MINUTE_WINDOW - now) + 1)

        if len(self._day) >= limit_day and self._day:
            oldest_day = self._day[0]
            retry = max(retry, int(oldest_day + DAY_WINDOW - now) + 1)

        return retry


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """
    In-memory sliding window rate limiter.

    Thread-safe enough for single-process deployments.
    For multi-process / multi-replica: swap with Redis backend.
    """

    def __init__(self) -> None:
        self._windows: dict[str, _KeyWindow] = {}

    def check(
        self,
        key_id: str,
        requests_per_minute: int,
        requests_per_day: int,
        record: bool = True,
    ) -> RateLimitResult:
        """
        Check whether a request is allowed for this key.

        Parameters
        ----------
        key_id : str
            The API key identifier.
        requests_per_minute : int
        requests_per_day : int
        record : bool
            If True (default), record this request in the window.
            Set False to peek without consuming a slot.

        Returns
        -------
        RateLimitResult
        """
        now = time.time()

        if key_id not in self._windows:
            self._windows[key_id] = _KeyWindow()

        window = self._windows[key_id]
        n_min, n_day = window.counts(now)

        # Check limits BEFORE recording
        if n_min >= requests_per_minute:
            retry = window.next_available(now, requests_per_minute, requests_per_day)
            return RateLimitResult(
                allowed=False, limit_type="minute",
                requests_minute=n_min, requests_day=n_day,
                limit_minute=requests_per_minute, limit_day=requests_per_day,
                retry_after=max(retry, 1),
            )

        if n_day >= requests_per_day:
            retry = window.next_available(now, requests_per_minute, requests_per_day)
            return RateLimitResult(
                allowed=False, limit_type="day",
                requests_minute=n_min, requests_day=n_day,
                limit_minute=requests_per_minute, limit_day=requests_per_day,
                retry_after=max(retry, 1),
            )

        if record:
            window.record(now)
            n_min += 1
            n_day += 1

        return RateLimitResult(
            allowed=True, limit_type=None,
            requests_minute=n_min, requests_day=n_day,
            limit_minute=requests_per_minute, limit_day=requests_per_day,
            retry_after=0,
        )

    def reset(self, key_id: str) -> None:
        """Clear all window state for a key (e.g. after revocation)."""
        self._windows.pop(key_id, None)

    def stats(self, key_id: str) -> dict:
        """Return current usage stats for a key."""
        if key_id not in self._windows:
            return {"requests_minute": 0, "requests_day": 0}
        now = time.time()
        n_min, n_day = self._windows[key_id].counts(now)
        return {"requests_minute": n_min, "requests_day": n_day}


# ── Global limiter instance ───────────────────────────────────────────────────

_limiter = RateLimiter()


def get_limiter() -> RateLimiter:
    return _limiter
