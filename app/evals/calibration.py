"""Judge vs. human agreement. Labels come from the Calibrate page in the dashboard."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from app.db import EvalResult, HumanLabel, SessionLocal


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement() -> dict:
    with SessionLocal() as s:
        labels = pd.DataFrame(s.execute(select(HumanLabel.trace_id, HumanLabel.helpful, HumanLabel.grounded)).all())
        judge = pd.DataFrame(s.execute(
            select(EvalResult.trace_id, EvalResult.evaluator, EvalResult.passed)
            .where(EvalResult.evaluator.in_(["judge_helpfulness", "judge_groundedness"]))
        ).all())
    if labels.empty or judge.empty:
        return {"n": 0}
    labels = labels.drop_duplicates("trace_id", keep="last")
    wide = judge.pivot_table(index="trace_id", columns="evaluator", values="passed", aggfunc="last").reset_index()
    m = labels.merge(wide, on="trace_id", how="inner").dropna()
    if m.empty:
        return {"n": 0}
    out = {"n": len(m)}
    for human, jcol in [("helpful", "judge_helpfulness"), ("grounded", "judge_groundedness")]:
        h, j = m[human].astype(bool).tolist(), m[jcol].astype(bool).tolist()
        out[human] = {
            "agreement": sum(x == y for x, y in zip(h, j)) / len(h),
            "kappa": cohen_kappa(h, j),
            "judge_too_lenient": sum((not x) and y for x, y in zip(h, j)),
            "judge_too_strict": sum(x and (not y) for x, y in zip(h, j)),
        }
    return out
