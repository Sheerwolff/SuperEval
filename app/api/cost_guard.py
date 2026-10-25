"""Keeps a public demo from becoming a public bill.

- per-session/IP sliding-window rate limit
- global daily USD budget (agent + judge spend, from recorded token usage)
When either trips, requests are served by the simulator instead of the live model,
and the response says so.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from app.config import settings
from app.db import SessionLocal, utcnow
from app.monitor.alerts import spend_since


class CostGuard:
    def __init__(self, per_hour: int, daily_budget: float):
        self.per_hour = per_hour
        self.daily_budget = daily_budget
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._spend_cache: tuple[float, float] = (0.0, 0.0)  # (checked_at, value)

    def allow_request(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and q[0] < now - 3600:
                q.popleft()
            if len(q) >= self.per_hour:
                return False
            q.append(now)
            return True

    def spend_today(self) -> float:
        checked_at, value = self._spend_cache
        if time.time() - checked_at < 15:
            return value
        day_start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with SessionLocal() as s:
            value = spend_since(s, day_start)
        self._spend_cache = (time.time(), value)
        return value

    def mode_for(self, key: str) -> tuple[str, str | None]:
        """('live' | 'sim', reason)."""
        if not settings.live_llm:
            return "sim", "no_api_key"
        if not self.allow_request(key):
            return "sim", "rate_limited"
        if self.spend_today() >= self.daily_budget:
            return "sim", "daily_budget_exhausted"
        return "live", None


guard = CostGuard(settings.rate_limit_per_hour, settings.daily_budget_usd)
