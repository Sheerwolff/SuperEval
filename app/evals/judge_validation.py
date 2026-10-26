"""Judge validation: the LLM judge's verdicts against a human-annotated gold set.

Gold set = the owner's latest judgment of each answer on the Review page ("is the answer correct?" yes/no).
The judge 'fails' an answer when any of its dimensions fails (grounded, method, answers the question, caveats).
Positive class = a wrong answer, so recall is the share of human-flagged wrong answers the judge also failed.

'Calibrate' (owner) saves a new evidence set for the live judge (the approved golden questions, see evidence.py) and
re-judges the gold set without and with that evidence: the ablation diff on the Judge Validation page.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from app.config import settings
from app.db import EvalResult, GoldenItem, HumanJudgment, JudgeCalibration, SessionLocal, Trace
from app.evals.calibration import cohen_kappa

JUDGE_DIMS = ["judge_groundedness", "judge_method", "judge_helpfulness", "judge_caveats"]


def gold_set(session=None) -> list[dict]:
    """Answers the owner judged correct or wrong (latest judgment per answer) that have a structured answer."""
    own = session is None
    s = session or SessionLocal()
    try:
        rows = s.execute(
            select(HumanJudgment.trace_id, HumanJudgment.created_at, HumanJudgment.id, HumanJudgment.correct,
                   HumanJudgment.note, Trace.question, Trace.tool_calls, Trace.analysis)
            .join(Trace, Trace.id == HumanJudgment.trace_id)
            .where(HumanJudgment.role == "owner", Trace.analysis.is_not(None))
            .order_by(HumanJudgment.created_at, HumanJudgment.id)).all()
    finally:
        if own:
            s.close()
    latest = {r.trace_id: r for r in rows}  # later judgments overwrite earlier ones
    return [{"trace_id": r.trace_id, "question": r.question, "tool_calls": r.tool_calls or [], "analysis": r.analysis,
             "human_wrong": r.correct == "no", "note": r.note}
            for r in latest.values() if r.correct in ("yes", "no")]


def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (human says wrong, judge failed it). Wrong answers are the positive class."""
    tp = sum(h and j for h, j in pairs)
    fn = sum(h and not j for h, j in pairs)
    fp = sum((not h) and j for h, j in pairs)
    tn = sum((not h) and (not j) for h, j in pairs)
    n = len(pairs)
    return {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "agreement": (tp + tn) / n if n else None,
            "kappa": cohen_kappa([h for h, _ in pairs], [j for _, j in pairs]) if n else None}


def judge_failed(outcomes) -> bool | None:
    """Any judge dimension failed; None when the judge didn't produce verdicts (e.g. an error)."""
    verdicts = [o.passed for o in outcomes if o.evaluator in JUDGE_DIMS and o.passed is not None]
    return (not all(verdicts)) if verdicts else None


def production_matrix() -> dict:
    """The live judge's stored verdicts on the gold set (only answers the judge sampled)."""
    gold = {g["trace_id"]: g["human_wrong"] for g in gold_set()}
    if not gold:
        return confusion([])
    with SessionLocal() as s:
        ev = pd.DataFrame(s.execute(select(EvalResult.trace_id, EvalResult.evaluator, EvalResult.passed)
                                    .where(EvalResult.trace_id.in_(list(gold)),
                                           EvalResult.evaluator.in_(JUDGE_DIMS))).all())
    if ev.empty:
        return confusion([])
    failed = ev.groupby("trace_id").passed.apply(lambda p: not p.dropna().astype(bool).all())
    return confusion([(gold[t], bool(f)) for t, f in failed.items()])


def snapshot_evidence() -> list[dict]:
    """The evidence set the live judge is fed: approved golden questions."""
    with SessionLocal() as s:
        return [{"id": g.id, "question": g.question, "reference_sql": g.reference_sql, "answer_type": g.answer_type}
                for g in s.query(GoldenItem).filter(GoldenItem.status == "approved").order_by(GoldenItem.id)]


def calibrate(judge_fn=None, on_progress=None, max_cost: float | None = None) -> dict:
    """Save a new evidence set and re-judge the gold set without and with evidence. `judge_fn(question, queries,
    analysis, evidence)` returns (verdict, meta) and defaults to the live judge."""
    from app.evals import evidence as ev
    from app.evals.analyst_judge import judge_live, outcomes

    judge_fn = judge_fn or (lambda q, qs, a, e: judge_live(q, qs, a, evidence=e))
    items = snapshot_evidence()
    gold = gold_set()
    rows, spent = [], 0.0
    for g in gold:
        evid = ev.retrieve(g["question"], items)
        out = {"trace_id": g["trace_id"], "question": g["question"], "human_wrong": g["human_wrong"],
               "note": g["note"], "evidence": (evid or {}).get("source")}
        for mode, e in (("without", None), ("with", evid)):
            if mode == "with" and e is None:  # nothing to add: same verdict as without
                out["with"], out["with_reasoning"] = out["without"], out["without_reasoning"]
                continue
            v, meta = judge_fn(g["question"], g["tool_calls"], g["analysis"], e)
            spent += meta.get("cost_usd") or 0.0
            out[mode] = judge_failed(outcomes(v, meta))
            out[f"{mode}_reasoning"] = v.reasoning
            if on_progress:
                on_progress(meta.get("cost_usd") or 0.0)
        rows.append(out)
        if max_cost is not None and spent > max_cost:
            break
    summary = {mode: confusion([(r["human_wrong"], bool(r[mode])) for r in rows if r[mode] is not None])
               for mode in ("without", "with")}
    with SessionLocal() as s:
        cal = JudgeCalibration(judge_model=settings.judge_model, evidence=items, summary=summary, rows=rows,
                               cost_usd=round(spent, 4))
        s.add(cal)
        s.commit()
        cal_id = cal.id
    ev._cache.update(at=0.0, items=None)  # the live judge picks up the new evidence set right away
    return {"id": cal_id, "n": len(rows), "evidence_items": len(items), "summary": summary, "cost_usd": spent}


def latest_calibration() -> dict | None:
    with SessionLocal() as s:
        c = s.execute(select(JudgeCalibration).order_by(JudgeCalibration.created_at.desc())).scalars().first()
        if c is None:
            return None
        return {"id": c.id, "created_at": c.created_at, "judge_model": c.judge_model, "evidence": c.evidence,
                "summary": c.summary, "rows": c.rows, "cost_usd": c.cost_usd}
