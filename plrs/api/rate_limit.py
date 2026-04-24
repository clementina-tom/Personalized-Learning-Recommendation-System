"""
plrs.api.rate_limit
===================
In-memory sliding window rate limiter — see full implementation in complete repo zip.
"""
import time
from collections import deque
from dataclasses import dataclass

MINUTE_WINDOW = 60; DAY_WINDOW = 86_400

@dataclass
class RateLimitResult:
    allowed: bool; limit_type: str | None
    requests_minute: int; requests_day: int
    limit_minute: int; limit_day: int; retry_after: int
    @property
    def headers(self):
        h = {"X-RateLimit-Limit-Minute": str(self.limit_minute), "X-RateLimit-Limit-Day": str(self.limit_day), "X-RateLimit-Remaining-Minute": str(max(0, self.limit_minute - self.requests_minute)), "X-RateLimit-Remaining-Day": str(max(0, self.limit_day - self.requests_day))}
        if not self.allowed: h["Retry-After"] = str(self.retry_after)
        return h
    def to_dict(self): return {"error": "Rate limit exceeded", "limit_type": self.limit_type, "requests_minute": self.requests_minute, "requests_day": self.requests_day, "limit_minute": self.limit_minute, "limit_day": self.limit_day, "retry_after_s": self.retry_after}

class _KeyWindow:
    __slots__ = ("_minute", "_day")
    def __init__(self): self._minute = deque(); self._day = deque()
    def record(self, now): self._minute.append(now); self._day.append(now)
    def counts(self, now):
        mc = now - MINUTE_WINDOW; dc = now - DAY_WINDOW
        while self._minute and self._minute[0] < mc: self._minute.popleft()
        while self._day and self._day[0] < dc: self._day.popleft()
        return len(self._minute), len(self._day)
    def next_available(self, now, lm, ld):
        r = 0
        if len(self._minute) >= lm and self._minute: r = max(r, int(self._minute[0] + MINUTE_WINDOW - now) + 1)
        if len(self._day) >= ld and self._day: r = max(r, int(self._day[0] + DAY_WINDOW - now) + 1)
        return r

class RateLimiter:
    def __init__(self): self._windows = {}
    def check(self, key_id, requests_per_minute, requests_per_day, record=True):
        now = time.time()
        if key_id not in self._windows: self._windows[key_id] = _KeyWindow()
        w = self._windows[key_id]; nm, nd = w.counts(now)
        if nm >= requests_per_minute:
            return RateLimitResult(False, "minute", nm, nd, requests_per_minute, requests_per_day, max(w.next_available(now, requests_per_minute, requests_per_day), 1))
        if nd >= requests_per_day:
            return RateLimitResult(False, "day", nm, nd, requests_per_minute, requests_per_day, max(w.next_available(now, requests_per_minute, requests_per_day), 1))
        if record: w.record(now); nm += 1; nd += 1
        return RateLimitResult(True, None, nm, nd, requests_per_minute, requests_per_day, 0)
    def reset(self, key_id): self._windows.pop(key_id, None)
    def stats(self, key_id):
        if key_id not in self._windows: return {"requests_minute": 0, "requests_day": 0}
        nm, nd = self._windows[key_id].counts(time.time()); return {"requests_minute": nm, "requests_day": nd}

_limiter = RateLimiter()
def get_limiter(): return _limiter
