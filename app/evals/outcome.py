"""The result of one evaluator on one trace."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalOutcome:
    evaluator: str
    score: float | None  # None = not applicable to this trace
    passed: bool | None
    details: dict = field(default_factory=dict)
    kind: str = "deterministic"
