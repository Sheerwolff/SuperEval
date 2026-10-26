"""The human layer: judgments on traces (Review page), the human override rate, agreement between
humans and the automated evals, and the golden set: adding traces to it and curating it from owner judgments."""
from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import func, select

from app.agent.textutil import detect_language
from app.db import EvalResult, GoldenItem, HumanJudgment, JudgeFeedback, SessionLocal, Trace, utcnow

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
                   HumanJudgment.corrected_answer) \
            .join(Trace, Trace.id == HumanJudgment.trace_id) \
            .where(Trace.source != "golden")  # reviewing golden-set test answers curates the set, not traffic
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


def save_judge_feedback(trace_id: str, role: str, reviewer: str | None, agree: bool,
                        disputed: list[str] | None = None, note: str | None = None) -> int:
    """Store a chat user's agree/disagree on the eval verdicts for one answer; returns the row id."""
    with SessionLocal() as s:
        fb = JudgeFeedback(trace_id=trace_id, role=role, reviewer=reviewer, agree=agree,
                           disputed=[] if agree else list(disputed or []), note=(note or "").strip() or None)
        s.add(fb)
        s.commit()
        fid = fb.id
    if not agree:  # every disagreement goes into the golden set as a draft for the owner to check
        what = ", ".join(disputed or []) or "no specific check"
        draft_from_trace(trace_id, "feedback", f"From chat feedback: disputed {what}." + (
            f" Note: {note.strip()}" if (note or "").strip() else ""))
    return fid


def judge_feedback_summary(since=None, session=None) -> dict:
    """Chat users' agree/disagree on eval verdicts: agreement rate, which checks get disputed (split by what the
    check said: disputing a pass means too lenient, disputing a fail too strict) and the latest disagreements."""
    own = session is None
    session = session or SessionLocal()
    try:
        q = select(JudgeFeedback.id, JudgeFeedback.created_at, JudgeFeedback.trace_id, JudgeFeedback.role,
                   JudgeFeedback.agree, JudgeFeedback.disputed, JudgeFeedback.note, Trace.question) \
            .join(Trace, Trace.id == JudgeFeedback.trace_id)
        if since is not None:
            q = q.where(JudgeFeedback.created_at >= since)
        fb = pd.DataFrame(session.execute(q).all())
        ev = pd.DataFrame()
        if not fb.empty:
            ev = pd.DataFrame(session.execute(
                select(EvalResult.trace_id, EvalResult.evaluator, EvalResult.passed, EvalResult.created_at)
                .where(EvalResult.trace_id.in_(fb.trace_id.tolist()))).all())
    finally:
        if own:
            session.close()
    disputed = pd.DataFrame(columns=["evaluator", "said_pass", "said_fail", "total"])
    if fb.empty:
        return {"n": 0, "agree_rate": None, "n_disagree": 0, "disputed": disputed, "recent": fb}
    rows = fb[~fb.agree].explode("disputed").dropna(subset=["disputed"])
    if not rows.empty:
        if not ev.empty:
            latest = ev.sort_values("created_at").drop_duplicates(["trace_id", "evaluator"], keep="last")
            rows = rows.merge(latest[["trace_id", "evaluator", "passed"]], how="left",
                              left_on=["trace_id", "disputed"], right_on=["trace_id", "evaluator"])
        else:
            rows["passed"] = None
        disputed = (rows.assign(said_pass=rows.passed == True, said_fail=rows.passed == False)  # noqa: E712
                    .groupby("disputed").agg(said_pass=("said_pass", "sum"), said_fail=("said_fail", "sum"),
                                             total=("said_pass", "size")).astype(int)
                    .reset_index().rename(columns={"disputed": "evaluator"})
                    .sort_values(["total", "evaluator"], ascending=[False, True], ignore_index=True))
    recent = fb[~fb.agree].sort_values(["created_at", "id"], ascending=False)[  # id breaks same-timestamp ties
        ["created_at", "role", "question", "disputed", "note"]].head(20)
    return {"n": len(fb), "agree_rate": float(fb.agree.mean()), "n_disagree": int((~fb.agree).sum()),
            "disputed": disputed, "recent": recent}


def _guess_shape(trace: Trace, tags=(), corrected_sql: str | None = None) -> tuple[str, str, dict]:
    """(qtype, answer_type, answer_key) draft for a trace going into the golden set; the owner fixes it on Review."""
    if "should have declined" in tags:
        return "trap_unanswerable", "decline", {}
    if "ignored ambiguity" in tags:
        return "trap_ambiguous", "assumption", {}
    sql = corrected_sql or next((q["sql"] for q in (trace.tool_calls or []) if not q.get("error")), None)
    cols = next((q["columns"] for q in (trace.tool_calls or []) if q.get("sql") == sql), []) if sql else []
    if re.search(r"chain_price_index|which chain|cheapest|priciest|most expensive", trace.question, re.I):
        return "comparison", "ranking", {"column": cols[0] if cols else "chain"}
    return "lookup", "scalar", {"column": cols[0] if cols else "v"}


def draft_from_trace(trace_id: str, source: str, notes: str, tags=(), corrected_sql: str | None = None,
                     session=None) -> str:
    """Add a trace's question to the golden set as a draft (source: promoted | trace | feedback). The reference SQL
    is the correction if given, else the agent's own first working query, so the owner should check it. One item
    per trace: returns the existing item's id if the trace is already in the set."""
    own = session is None
    s = session or SessionLocal()
    try:
        existing = s.execute(select(GoldenItem.id).where(GoldenItem.source_trace_id == trace_id)).scalar()
        if existing:
            return existing
        t = s.get(Trace, trace_id)
        if t is None:
            raise ValueError(f"no trace {trace_id}")
        qtype, answer_type, key = _guess_shape(t, tags or (), corrected_sql)
        sql = None if answer_type in ("decline", "assumption") else (
            corrected_sql or next((q["sql"] for q in (t.tool_calls or []) if not q.get("error")), None))
        n = s.execute(select(func.count(GoldenItem.id))).scalar_one()
        gid = f"g{n + 1:03d}"
        while s.get(GoldenItem, gid):
            n += 1
            gid = f"g{n + 1:03d}"
        s.add(GoldenItem(id=gid, question=t.question, language=detect_language(t.question), qtype=qtype,
                         difficulty="medium", answer_type=answer_type, reference_sql=sql, answer_key=key,
                         status="draft", source=source, source_trace_id=t.id, notes=notes))
        s.commit()
        return gid
    finally:
        if own:
            s.close()


def promote(judgment_id: int) -> str:
    """Create a golden draft from an owner-judged trace; returns the new item id."""
    with SessionLocal() as s:
        j = s.get(HumanJudgment, judgment_id)
        if j is None or j.role != "owner":
            raise ValueError("only owner judgments can be promoted")
        if j.promoted_to:
            return j.promoted_to
        notes = "Promoted from a reviewed trace." + (f" Human correction: {j.corrected_answer}" if j.corrected_answer
                                                       else "")
        gid = draft_from_trace(j.trace_id, "promoted", notes, tags=j.tags or (), corrected_sql=j.corrected_sql,
                               session=s)
        j.promoted_to = gid
        s.commit()
        return gid


def golden_for_trace(trace_id: str, session=None) -> tuple[str | None, bool | None]:
    """(golden item id, whether the scorer matched the answer to the reference) for a golden-set test answer."""
    own = session is None
    s = session or SessionLocal()
    try:
        e = s.execute(select(EvalResult).where(EvalResult.trace_id == trace_id,
                                               EvalResult.evaluator == "golden_match")).scalars().first()
        return (e.details.get("golden_id"), e.passed) if e else (None, None)
    finally:
        if own:
            s.close()


def curate_golden(trace_id: str, correct: str, corrected_sql: str | None = None, bad_question: bool = False) -> str:
    """Apply an owner's judgment of a golden-set test answer to its golden item; returns what happened.

    The scorer already compared the answer with the reference, so the owner's verdict tells us about the reference:
    agreeing with the scorer confirms it (approve), disagreeing means the reference is suspect (draft, unless the
    owner supplies corrected SQL, which becomes the new reference). A bad question is rejected."""
    with SessionLocal() as s:
        gid, matched = golden_for_trace(trace_id, session=s)
        g = s.get(GoldenItem, gid) if gid else None
        if g is None:
            return "not a golden-set answer"
        sql = (corrected_sql or "").strip()
        if bad_question:
            g.status, msg = "rejected", f"{gid} rejected"
        elif sql:
            g.reference_sql, g.status, msg = sql, "approved", f"{gid} approved with your SQL as the reference"
        elif correct == "unsure" or matched is None:
            return f"{gid} unchanged"
        elif (correct == "yes") == bool(matched):
            g.status, msg = "approved", f"{gid} approved: you agree with its reference"
        else:
            g.status = "draft"
            said = "correct" if correct == "yes" else "wrong"
            g.notes = ((g.notes or "") + f" Reference disputed: the owner judged the answer {said}, the scorer "
                       "disagreed. Add correct SQL to fix it.").strip()
            msg = f"{gid} kept as a draft: your verdict disagrees with its reference"
        g.reviewed_by = "owner"
        g.updated_at = utcnow()
        s.commit()
        return msg


def approve_into_golden(judgment_id: int) -> str:
    """Owner's 'approve and add to the golden set' from the Review page: put the judged trace's question in the golden
    set (or reuse its item) and approve it when the reference can be trusted: the owner gave correct SQL, judged the
    agent's answer correct, or the question is about declining / stating an assumption. Otherwise it stays a draft
    until correct SQL is added. Returns what happened."""
    with SessionLocal() as s:
        j = s.get(HumanJudgment, judgment_id)
        if j is None or j.role != "owner":
            raise ValueError("only owner judgments can approve golden questions")
        notes = "Approved on the Review page." + (f" A good answer: {j.corrected_answer}" if j.corrected_answer else "")
        gid = draft_from_trace(j.trace_id, "promoted", notes, tags=j.tags or (), corrected_sql=j.corrected_sql,
                               session=s)
        g = s.get(GoldenItem, gid)
        sql = (j.corrected_sql or "").strip()
        if sql:
            g.reference_sql = sql
        j.promoted_to = gid
        if sql or j.correct == "yes" or g.answer_type in ("decline", "assumption"):
            g.status, g.reviewed_by = "approved", "owner"
            msg = f"{gid} approved and in the golden set"
        else:
            msg = f"{gid} added as a draft: the answer was judged wrong, so add the correct SQL to approve it"
        g.updated_at = utcnow()
        s.commit()
        return msg
