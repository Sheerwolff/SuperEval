"""Question type for an analytical question (used for slicing metrics and for input-drift PSI)."""
from __future__ import annotations

import re

from app.agent.textutil import INJECTION_PATTERNS

TOPICS = ["lookup", "comparison", "spread", "promo", "geo", "unanswerable", "ambiguous", "injection"]


def classify(question: str, cities: list[str] | None = None) -> str:
    from app.evals.online import expects_decline, is_ambiguous

    q = question
    if INJECTION_PATTERNS.search(q):
        return "injection"
    if expects_decline(q):
        return "unanswerable"
    if cities is not None and is_ambiguous(q, cities):
        return "ambiguous"
    if re.search(r"promo|on sale|deal|מבצע", q, re.I):
        return "promo"
    if re.search(r"gap|spread|differen|פער", q, re.I):
        return "spread"
    if re.search(r"\bor in\b|cities|city is|באיזו עיר", q, re.I):
        return "geo"
    if re.search(r"which chain|cheapest chain|most expensive|cheaper|compare|רשת|זולה|יקרה", q, re.I):
        return "comparison"
    return "lookup"
