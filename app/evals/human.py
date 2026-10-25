"""The human layer: judgments on traces (Review page), the human override rate, agreement between
humans and the automated evals, and promoting judged traces into the golden-set review queue."""
from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import func, select

from app.agent.textutil import detect_language
from app.db import EvalResult, GoldenItem, HumanJudgment, SessionLocal, Trace

FAILURE_TAGS = ["wrong number", "ungrounded number", "bad method", "ignored ambiguity", "should have declined",
                "declined wrongly", "misleading chart", "wrong language", "other"]
LAYER1 = ["grounding", "sql_health", "method_lint", "canonical_check", "chart_lint", "decline_check",
          "assumption_check"]


def judgments_frame(role: str | None = None, session=None) -> pd.DataFrame:
    own = session is None
    session = session or SessionLocal()
    try:
        q = select(HumanJudgment.id, HumanJudgment.created_at, HumanJudgment.trace_id, HumanJudgment.role,
                   HumanJudgment.correct, HumanJudgment.method_ok, HumanJudgment.tags,
                   HumanJudgment.corrected_answer)
        if role:
            q = q.where(HumanJudgment.role == role)
        df = pd.DataFrame(session.execute(q).all())
    finally:
        if own:
            session.close()
    if df.empty:
        return pd.DataFrame(columns=["id", "created_at", "trace_id", "role", "correct", "method_ok", "tags",
                                     "corrected_answer"])
    # latest judgment per (trace, role) wins
    return df.sort_values("created_at").drop_duplicates(["trace_id", "role"], keep="last")


def is_override(row) -> bool:
    return row["correct"] == "no" or row["method_ok"] is False or bool((row.get("corrected_answer") or "").strip())


def override_rate(df: pd.DataFrame) -> tuple[float | None, int]:
    """Share of reviewed traces a human judged wrong, unsound or corrected (the article's leading indicator)."""
    if df.empty:
        return None, 0
    flags = df.apply(is_override, axis=1)
    return float(flags.mean()), len(df)


def agreement(df: pd.DataFrame, session=None) -> dict:
    """Do the automated Layer-1 evals flag what humans flag? 2x2 of human verdict vs. 'any Layer-1 check
    failed', plus per-evaluator recall on human-flagged traces."""
    if df.empty:
        return {"n": 0}
    own = session is None
    session = session or SessionLocal()
    try:
        ev = pd.DataFrame(session.execute(
            select(EvalResult.trace_id, EvalResult.evaluator, EvalResult.passed)
            .where(EvalResult.trace_id.in_(df.trace_id.tolist()), EvalResult.evaluator.in_(LAYER1))).all())
    finally:
        if own:
            session.close()
    d = df.assign(human_bad=df.apply(is_override, axis=1))
    if ev.empty:
        d["auto_bad"] = False
        per = {}
    else:
        failed = ev[ev.passed == False]  # noqa: E712
        d["auto_bad"] = d.trace_id.isin(failed.trace_id)
        bad_ids = set(d[d.human_bad].trace_id)
        per = {e: round(len(set(failed[failed.evaluator == e].trace_id) & bad_ids) / len(bad_ids), 3)
               for e in LAYER1} if bad_ids else {}
    matrix = {
        "both_flag": int((d.human_bad & d.auto_bad).sum()),
        "human_only": int((d.human_bad & ~d.auto_bad).sum()),  # what the evals miss
        "auto_only": int((~d.human_bad & d.auto_bad).sum()),  # eval false alarms, or humans missed it
        "neither": int((~d.human_bad & ~d.auto_bad).sum()),
    }
    caught = matrix["both_flag"] + matrix["human_only"]
    return {"n": len(d), **matrix, "eval_recall_on_human_flags": round(matrix["both_flag"] / caught, 3) if caught
            else None, "per_evaluator_recall": per}


def _guess_shape(trace: Trace, judgment: HumanJudgment) -> tuple[str, str, dict]:
    """(qtype, answer_type, answer_key) draft for a promoted trace; the owner fixes it in the queue."""
    if "should have declined" in (judgment.tags or []):
        return "trap_unanswerable", "decline", {}
    if "ignored ambiguity" in (judgment.tags or []):
        return "trap_ambiguous", "assumption", {}
    sql = judgment.corrected_sql or next(
        (q["sql"] for q in (trace.tool_calls or []) if not q.get("error")), None)
    cols = next((q["columns"] for q in (trace.tool_calls or []) if q.get("sql") == sql), []) if sql else []
    if re.search(r"chain_price_index|which chain|cheapest|priciest|most expensive", trace.question, re.I):
        return "comparison", "ranking", {"column": cols[0] if cols else "chain"}
    return "lookup", "scalar", {"column": cols[0] if cols else "v"}


def promote(judgment_id: int) -> str:
    """Create a golden draft from an owner-judged trace; returns the new item id."""
    with SessionLocal() as s:
        j = s.get(HumanJudgment, judgment_id)
        if j is None or j.role != "owner":
            raise ValueError("only owner judgments can be promoted")
        if j.promoted_to:
            return j.promoted_to
        t = s.get(Trace, j.trace_id)
        qtype, answer_type, key = _guess_shape(t, j)
        sql = None if answer_type in ("decline", "assumption") else (
            j.corrected_sql or next((q["sql"] for q in (t.tool_calls or []) if not q.get("error")), None))
        n = s.execute(select(func.count(GoldenItem.id))).scalar_one()
        gid = f"g{n + 1:03d}"
        while s.get(GoldenItem, gid):
            n += 1
            gid = f"g{n + 1:03d}"
        notes = "Promoted from a reviewed trace." + (f" Human correction: {j.corrected_answer}" if j.corrected_answer
                                                       else "")
        s.add(GoldenItem(id=gid, question=t.question, language=detect_language(t.question), qtype=qtype,
                         difficulty="medium", answer_type=answer_type, reference_sql=sql, answer_key=key,
                         status="draft", source="promoted", source_trace_id=t.id, notes=notes))
        j.promoted_to = gid
        s.commit()
        return gid
