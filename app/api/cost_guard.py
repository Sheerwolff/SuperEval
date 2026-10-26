"""Keeps a public demo from becoming a public bill: each user may spend USER_DAILY_BUDGET_USD of live LLM
calls per UTC day, across chat and golden-set runs. Past the cap, chat is served by the simulator (and
says so) and golden-set runs are refused.
"""
from __future__ import annotations

import threading

from app.config import settings
from app.db import SessionLocal, UserSpend, utcnow


def _today() -> str:
    return utcnow().strftime("%Y-%m-%d")


class CostGuard:
    def __init__(self, user_daily_budget: float):
        self.budget = user_daily_budget
        self._lock = threading.Lock()

    def spent_today(self, user_key: str) -> float:
        with SessionLocal() as s:
            row = s.get(UserSpend, (user_key, _today()))
            return row.usd if row else 0.0

    def remaining(self, user_key: str) -> float:
        return max(0.0, self.budget - self.spent_today(user_key))

    def charge(self, user_key: str, usd: float) -> None:
        if usd <= 0:
            return
        with self._lock, SessionLocal() as s:
            row = s.get(UserSpend, (user_key, _today()))
            if row is None:
                s.add(UserSpend(user_key=user_key, day=_today(), usd=usd))
            else:
                row.usd += usd
            s.commit()

    def mode_for(self, user_key: str) -> tuple[str, str | None]:
        """('live' | 'sim', reason)."""
        if not settings.live_llm:
            return "sim", "no_api_key"
        if self.remaining(user_key) <= 0:
            return "sim", "user_budget_exhausted"
        return "live", None


guard = CostGuard(settings.user_daily_budget_usd)
