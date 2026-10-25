import pandas as pd
import streamlit as st
from sqlalchemy import select

from app.db import EvalResult, HumanJudgment, SessionLocal, Trace
from app.evals.human import FAILURE_TAGS, agreement, judgments_frame, override_rate, promote
from components import owner_unlock, role
from lib import eval_badges, setup, tiles
from render import render_chart, render_queries

setup("Review", "🧑‍⚖️")
st.markdown("## 🧑‍⚖️ Review traces")
st.caption("Read what the analyst did and judge it yourself. Judgments are stored with your role. Owner judgments "
           "calibrate the LLM judge and can be promoted into the golden set; visitor judgments are tracked "
           "separately. Together they measure the **human override rate**, and whether the automated evals catch "
           "what people catch.")
owner_unlock()
me = role()
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = uuid.uuid4().hex[:12]

# ---- headline metrics ----------------------------------------------------------------
owner_df, visitor_df = judgments_frame("owner"), judgments_frame("visitor")
o_rate, o_n = override_rate(owner_df)
v_rate, v_n = override_rate(visitor_df)
agr = agreement(pd.concat([owner_df, visitor_df]).drop_duplicates("trace_id", keep="first")
                if len(owner_df) + len(visitor_df) else owner_df)


def pct(v):
    return "—" if v is None else f"{v:.0%}"


tiles([("Human override rate · owner", pct(o_rate), f"n = {o_n} reviewed", False),
       ("Human override rate · visitors", pct(v_rate), f"n = {v_n} reviewed", False),
       ("Evals catch what humans flag", pct(agr.get("eval_recall_on_human_flags")),
        f"{agr.get('both_flag', 0)} caught · {agr.get('human_only', 0)} missed", False),
       ("Eval-only flags", str(agr.get("auto_only", 0)), "automated fail, human said fine", False)])
with st.expander("Agreement matrix and per-evaluator recall"):
    if agr.get("n"):
        st.dataframe(pd.DataFrame({"automated: flagged": [agr["both_flag"], agr["auto_only"]],
                                   "automated: passed": [agr["human_only"], agr["neither"]]},
                                  index=["human: wrong/corrected", "human: fine"]), use_container_width=True)
        if agr["per_evaluator_recall"]:
            st.caption("Share of human-flagged traces each evaluator also failed:")
            st.dataframe(pd.Series(agr["per_evaluator_recall"], name="recall").to_frame(), use_container_width=True)
    else:
        st.caption("No judgments yet.")

# ---- trace list ---------------------------------------------------------------------
st.divider()
f1, f2, f3 = st.columns(3)
only_unreviewed = f1.toggle("Unreviewed only", value=True)
only_flagged = f2.toggle("Failed an automated check", value=False)
source = f3.selectbox("Source", ["user", "any", "synthetic", "backfill", "offline"])
with SessionLocal() as s:
    q = select(Trace).where(Trace.analysis.is_not(None) | Trace.error.is_not(None)).order_by(
        Trace.created_at.desc()).limit(300)
    if source != "any":
        q = q.where(Trace.source == source)
    traces = s.execute(q).scalars().all()
    reviewed = {tid for (tid,) in s.execute(select(HumanJudgment.trace_id).where(HumanJudgment.role == me))}
    failed = {tid for (tid,) in s.execute(select(EvalResult.trace_id).where(EvalResult.passed.is_(False)))}
    rows = [{"time": t.created_at.strftime("%m-%d %H:%M"), "version": t.agent_version, "question": t.question,
             "automated": "⚠️ failed" if t.id in failed else "✓", "id": t.id} for t in traces
            if (not only_unreviewed or t.id not in reviewed) and (not only_flagged or t.id in failed)]
if not rows:
    st.info("Nothing to review with these filters. Ask the analyst something on the Chat page first.")
    st.stop()
table = pd.DataFrame(rows)
pick = st.dataframe(table.drop(columns=["id"]), hide_index=True, use_container_width=True, height=240,
                    on_select="rerun", selection_mode="single-row")
idx = pick.selection.rows[0] if pick.selection.rows else 0
tid = table.iloc[idx]["id"]

with SessionLocal() as s:
    t = s.get(Trace, tid)
    tr = {"id": t.id, "question": t.question, "answer": t.answer, "analysis": t.analysis, "tool_calls": t.tool_calls,
          "error": t.error, "version": t.agent_version, "model": t.model}
    evals = [{"evaluator": e.evaluator, "kind": e.kind, "passed": e.passed, "details": e.details} for e in t.evals]
    prior = s.execute(select(HumanJudgment).where(HumanJudgment.trace_id == tid)
                      .order_by(HumanJudgment.created_at.desc())).scalars().all()
    prior = [{"id": j.id, "role": j.role, "correct": j.correct, "method_ok": j.method_ok, "tags": j.tags,
              "corrected_answer": j.corrected_answer, "note": j.note, "promoted_to": j.promoted_to} for j in prior]

st.divider()
a, b = st.columns([3, 2], gap="large")
with a:
    st.markdown(f"#### {tr['question']}")
    st.caption(f"{tr['version']} · {tr['model']} · trace `{tr['id'][:8]}`")
    st.markdown(tr["answer"] or f"*(no answer: {tr['error']})*")
    render_chart(tr, key="_review")
    render_queries(tr)
with b:
    st.markdown("**Automated verdicts**")
    st.markdown(eval_badges(evals) or "—", unsafe_allow_html=True)
    for e in evals:
        if e["passed"] is False and e["details"].get("message"):
            st.caption(f"✗ {e['evaluator']}: {e['details']['message']}")
    if prior:
        st.markdown("**Earlier human judgments**")
        for j in prior:
            st.caption(f"{j['role']}: correct={j['correct']}, method_ok={j['method_ok']}, tags={j['tags'] or '—'}"
                       + (f", corrected: {j['corrected_answer']}" if j["corrected_answer"] else "")
                       + (f" · promoted to {j['promoted_to']}" if j["promoted_to"] else ""))

st.markdown(f"#### Your judgment · as **{me}**")
with st.form(f"judge_{tid}"):
    c1, c2 = st.columns(2)
    correct = c1.radio("Is the headline answer correct?", ["yes", "no", "unsure"], horizontal=True)
    method = c2.radio("Was the method sound?", ["yes", "no", "n/a"], horizontal=True,
                      help="Compared like with like, sensible filters, honest caveats.")
    tags = st.multiselect("What went wrong (if anything)", FAILURE_TAGS)
    corrected = st.text_input("Correct answer, if you know it")
    corrected_sql = st.text_area("Correct SQL (optional, becomes the golden reference)", height=80) \
        if me == "owner" else None
    note = st.text_input("Note")
    if st.form_submit_button("Save judgment", type="primary"):
        with SessionLocal() as s:
            s.add(HumanJudgment(trace_id=tid, role=me, reviewer=st.session_state.session_id, correct=correct,
                                method_ok=None if method == "n/a" else method == "yes", tags=tags,
                                corrected_answer=corrected or None, corrected_sql=corrected_sql or None,
                                note=note or None))
            s.commit()
        st.toast("Saved. Thanks!")
        st.rerun()

owner_prior = [j for j in prior if j["role"] == "owner" and not j["promoted_to"]]
if me == "owner" and owner_prior:
    if st.button("🏅 Promote to the golden set (draft for review)"):
        gid = promote(owner_prior[0]["id"])
        st.success(f"Created {gid} in the golden-set review queue. Approve it there to make it gate CI.")
