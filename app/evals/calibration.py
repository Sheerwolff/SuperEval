"""Judge vs. human agreement (Cohen's kappa), from owner judgments on the Golden set page.

  helpful  human "is the headline answer correct?" (yes/no)  vs. judge_helpfulness passed
  grounded human "was the method sound?" (yes/no)            vs. judge_method passed
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from app.db import EvalResult, HumanJudgment, SessionLocal


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement() -> dict:
    with SessionLocal() as s:
        labels = pd.DataFrame(s.execute(
            select(HumanJudgment.trace_id, HumanJudgment.created_at, HumanJudgment.correct, HumanJudgment.method_ok)
            .where(HumanJudgment.role == "owner")).all())
        judge = pd.DataFrame(s.execute(
            select(EvalResult.trace_id, EvalResult.evaluator, EvalResult.passed)
            .where(EvalResult.evaluator.in_(["judge_helpfulness", "judge_method"]))).all())
    if labels.empty or judge.empty:
        return {"n": 0}
    labels = labels.sort_values("created_at").drop_duplicates("trace_id", keep="last")
    wide = judge.pivot_table(index="trace_id", columns="evaluator", values="passed", aggfunc="last").reset_index()
    m = labels.merge(wide, on="trace_id", how="inner")
    out = {"n": len(m)}
    pairs = [("helpful", m[m.correct.isin(["yes", "no"])], lambda d: d.correct == "yes", "judge_helpfulness"),
             ("grounded", m[m.method_ok.notna()], lambda d: d.method_ok.astype(bool), "judge_method")]
    for name, d, human_fn, jcol in pairs:
        d = d[d[jcol].notna()] if jcol in d else d.iloc[0:0]
        if d.empty:
            out[name] = None
            continue
        h, j = human_fn(d).tolist(), d[jcol].astype(bool).tolist()
        out[name] = {"n": len(h), "agreement": sum(x == y for x, y in zip(h, j, strict=True)) / len(h),
                     "kappa": cohen_kappa(h, j),
                     "judge_too_lenient": sum((not x) and y for x, y in zip(h, j, strict=True)),
                     "judge_too_strict": sum(x and (not y) for x, y in zip(h, j, strict=True))}
    out["n"] = max((out[k]["n"] for k in ("helpful", "grounded") if out.get(k)), default=0)
    return out
