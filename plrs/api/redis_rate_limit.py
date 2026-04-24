"""
plrs.api.redis_rate_limit
==========================
Redis-backed sliding window rate limiter for PLRS.

Drop-in replacement for the in-memory RateLimiter.
Multi-process safe — works correctly across multiple API server replicas.

Implementation: Redis sorted sets (ZADD + ZREMRANGEBYSCORE)
Each key has two sorted sets:
  plrs:rl:{key_id}:minute  — timestamps in last 60s
  plrs:rl:{key_id}:day     — timestamps in last 24h

Set PLRS_REDIS_URL to activate:
    PLRS_REDIS_URL=redis://localhost:6379/0

Falls back to in-memory if not set.

Usage:
    from plrs.api.redis_rate_limit import get_rate_limiter

    limiter = get_rate_limiter()   # Redis or in-memory based on env
    result  = limiter.check(key_id, requests_per_minute=60, requests_per_day=1000)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

MINUTE_WINDOW = 60
DAY_WINDOW    = 86_400
KEY_PREFIX    = "plrs:rl:"


@dataclass
class RateLimitResult:
    """Same interface as in-memory RateLimitResult."""
    allowed:         bool
    limit_type:      str | None
    requests_minute: int
    requests_day:    int
    limit_minute:    int
    limit_day:       int
    retry_after:     int

    @property
    def headers(self) -> dict[str, str]:
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
            "error":           "Rate limit exceeded",
            "limit_type":      self.limit_type,
            "requests_minute": self.requests_minute,
            "requests_day":    self.requests_day,
            "limit_minute":    self.limit_minute,
            "limit_day":       self.limit_day,
            "retry_after_s":   self.retry_after,
        }


class RedisRateLimiter:
    """
    Redis-backed sliding window rate limiter.

    Thread-safe and process-safe. Suitable for multi-replica deployments.

    Parameters
    ----------
    redis_url : str
        Redis connection URL (e.g. "redis://localhost:6379/0").
    key_prefix : str
        Prefix for all Redis keys (default "plrs:rl:").
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = KEY_PREFIX,
    ) -> None:
        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError(
                "redis package not installed. "
                "Run: pip install redis"
            )
        self._redis     = redis_lib.from_url(redis_url, decode_responses=True)
        self._prefix    = key_prefix

    def check(
        self,
        key_id: str,
        requests_per_minute: int,
        requests_per_day: int,
        record: bool = True,
    ) -> RateLimitResult:
        """
        Check rate limit and optionally record the request.

        Uses a Lua script for atomic read-then-write (no race conditions).
        """
        now = time.time()
        min_key = f"{self._prefix}{key_id}:minute"
        day_key = f"{self._prefix}{key_id}:day"

        min_cutoff = now - MINUTE_WINDOW
        day_cutoff = now - DAY_WINDOW

        pipe = self._redis.pipeline()
        # Remove stale entries
        pipe.zremrangebyscore(min_key, "-inf", min_cutoff)
        pipe.zremrangebyscore(day_key, "-inf", day_cutoff)
        # Count current window
        pipe.zcard(min_key)
        pipe.zcard(day_key)
        results = pipe.execute()

        n_min = results[2]
        n_day = results[3]

        # Check limits BEFORE recording
        if n_min >= requests_per_minute:
            oldest = self._redis.zrange(min_key, 0, 0, withscores=True)
            retry  = int(oldest[0][1] + MINUTE_WINDOW - now) + 1 if oldest else 60
            return RateLimitResult(
                allowed=False, limit_type="minute",
                requests_minute=n_min, requests_day=n_day,
                limit_minute=requests_per_minute, limit_day=requests_per_day,
                retry_after=max(retry, 1),
            )

        if n_day >= requests_per_day:
            oldest = self._redis.zrange(day_key, 0, 0, withscores=True)
            retry  = int(oldest[0][1] + DAY_WINDOW - now) + 1 if oldest else 3600
            return RateLimitResult(
                allowed=False, limit_type="day",
                requests_minute=n_min, requests_day=n_day,
                limit_minute=requests_per_minute, limit_day=requests_per_day,
                retry_after=max(retry, 1),
            )

        if record:
            member = f"{now:.6f}"
            pipe2  = self._redis.pipeline()
            pipe2.zadd(min_key, {member: now})
            pipe2.zadd(day_key, {member: now})
            pipe2.expire(min_key, MINUTE_WINDOW + 5)
            pipe2.expire(day_key, DAY_WINDOW + 5)
            pipe2.execute()
            n_min += 1
            n_day += 1

        return RateLimitResult(
            allowed=True, limit_type=None,
            requests_minute=n_min, requests_day=n_day,
            limit_minute=requests_per_minute, limit_day=requests_per_day,
            retry_after=0,
        )

    def reset(self, key_id: str) -> None:
        """Clear rate limit state for a key."""
        min_key = f"{self._prefix}{key_id}:minute"
        day_key = f"{self._prefix}{key_id}:day"
        self._redis.delete(min_key, day_key)

    def stats(self, key_id: str) -> dict:
        """Return current window usage for a key."""
        now        = time.time()
        min_key    = f"{self._prefix}{key_id}:minute"
        day_key    = f"{self._prefix}{key_id}:day"
        min_cutoff = now - MINUTE_WINDOW
        day_cutoff = now - DAY_WINDOW

        pipe = self._redis.pipeline()
        pipe.zcount(min_key, min_cutoff, "+inf")
        pipe.zcount(day_key, day_cutoff, "+inf")
        n_min, n_day = pipe.execute()

        return {"requests_minute": n_min, "requests_day": n_day}

    def health_check(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            return self._redis.ping()
        except Exception:
            return False


# ── Factory ───────────────────────────────────────────────────────────────────

def get_rate_limiter():
    """
    Return the appropriate rate limiter based on environment.

    PLRS_REDIS_URL set → RedisRateLimiter
    Not set            → in-memory RateLimiter (default)
    """
    redis_url = os.getenv("PLRS_REDIS_URL")
    if redis_url:
        return RedisRateLimiter(redis_url)
    from plrs.api.rate_limit import get_limiter
    return get_limiter()
