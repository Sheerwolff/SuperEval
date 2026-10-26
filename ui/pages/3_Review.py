import pandas as pd
import streamlit as st
from sqlalchemy import select

from app.analyst.semantic import get_semantic
from app.db import EvalResult, GoldenItem, HumanJudgment, SessionLocal, Trace
from app.evals import golden
from app.evals.human import (
    FAILURE_TAGS,
    agreement,
    curate_golden,
    draft_from_trace,
    golden_for_trace,
    judgments_frame,
    override_rate,
    promote,
)
from components import owner_unlock, role
from lib import eval_badges, setup, tiles
from render import render_chart, render_queries

setup("Review", "🧑‍⚖️")
st.markdown("## 🧑‍⚖️ Review traces")
st.caption("Read what the analyst did and judge it yourself. Judgments are stored with your role. Owner judgments "
           "calibrate the LLM judge and can add answers to the golden set; visitor judgments are tracked "
           "separately. Together they measure the **human override rate**, and whether the automated evals catch "
           "what people catch. Pick source **golden** to review golden-set test answers: an owner's judgment there "
           "approves, fixes or rejects the golden question.")
owner_unlock()
me = role()
if toast := st.session_state.pop("review_toast", None):
    st.toast(toast)
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
FAILING = ["any", "any check", "grounding", "method_lint", "canonical_check", "sql_health", "completion",
           "chart_lint", "decline_check", "assumption_check", "language_match", "judge_method", "judge_groundedness",
           "judge_helpfulness", "golden_match"]
TOPICS = ["any", "comparison", "lookup", "spread", "promo", "geo", "ambiguous", "unanswerable", "injection"]
st.divider()
f1, f2, f3 = st.columns(3)
source = f1.selectbox("Source", ["user", "golden", "any", "synthetic", "backfill", "offline"])
failing = f2.selectbox("Failing check", FAILING, help="'any check' = failed at least one automated check")
topic = f3.selectbox("Question type", TOPICS)
only_unreviewed = st.toggle("Unreviewed only", value=True)
with SessionLocal() as s:
    q = select(Trace).where(Trace.analysis.is_not(None) | Trace.error.is_not(None)).order_by(
        Trace.created_at.desc()).limit(300)
    if source != "any":
        q = q.where(Trace.source == source)
    if topic != "any":
        q = q.where(Trace.topic == topic)
    if failing != "any":
        fails = select(EvalResult.trace_id).where(EvalResult.passed.is_(False))
        q = q.where(Trace.id.in_(fails if failing == "any check" else fails.where(EvalResult.evaluator == failing)))
    traces = s.execute(q).scalars().all()
    reviewed = {tid for (tid,) in s.execute(select(HumanJudgment.trace_id).where(HumanJudgment.role == me))}
    rows = [{"time": t.created_at.strftime("%m-%d %H:%M"), "source": t.source, "type": t.topic,
             "question": t.question,
             "failed": ", ".join(e.evaluator for e in t.evals if e.passed is False) or "—", "id": t.id}
            for t in traces if not only_unreviewed or t.id not in reviewed]
if not rows:
    st.info("Nothing to review with these filters. Ask the analyst something on the Chat page first.")
    st.stop()
table = pd.DataFrame(rows)
st.caption(f"{len(table)} most recent matching traces")
pick = st.dataframe(table.drop(columns=["id"]), hide_index=True, use_container_width=True, height=280,
                    on_select="rerun", selection_mode="single-row")
idx = pick.selection.rows[0] if pick.selection.rows else 0
tid = table.iloc[idx]["id"]

with SessionLocal() as s:
    t = s.get(Trace, tid)
    tr = {"id": t.id, "question": t.question, "answer": t.answer, "analysis": t.analysis, "tool_calls": t.tool_calls,
          "error": t.error, "version": t.agent_version, "model": t.model}
    meta = (f"{t.created_at:%Y-%m-%d %H:%M} UTC · {t.source} · {t.agent_version} · {t.model} · {t.steps} steps · "
            f"{t.latency_ms / 1000:.1f}s · {t.input_tokens + t.output_tokens:,} tokens · ${t.cost_usd:.4f} · "
            f"trace `{t.id[:8]}`")
    evals = [{"evaluator": e.evaluator, "kind": e.kind, "passed": e.passed, "details": e.details} for e in t.evals]
    prior = s.execute(select(HumanJudgment).where(HumanJudgment.trace_id == tid)
                      .order_by(HumanJudgment.created_at.desc())).scalars().all()
    prior = [{"id": j.id, "role": j.role, "correct": j.correct, "method_ok": j.method_ok, "tags": j.tags,
              "corrected_answer": j.corrected_answer, "note": j.note, "promoted_to": j.promoted_to} for j in prior]
    gid, matched = golden_for_trace(tid, session=s)
    gitem = s.get(GoldenItem, gid) if gid else None
    gold = golden.as_dict(gitem) if gitem else None
    in_golden = s.execute(select(GoldenItem.id).where(GoldenItem.source_trace_id == tid)).scalar()

st.divider()
a, b = st.columns([3, 2], gap="large")
with a:
    st.markdown(f"#### {tr['question']}")
    st.caption(meta.replace("$", "\\$"))
    st.markdown(tr["answer"] or f"*(no answer: {tr['error']})*")
    render_chart(tr, key="_review")
    render_queries(tr, expanded=failing != "any")
with b:
    st.markdown("**Automated verdicts**")
    st.markdown(eval_badges(evals) or "—", unsafe_allow_html=True)
    for e in evals:
        if e["passed"] is False:
            with st.expander(f"✗ {e['evaluator']}", expanded=True):
                if e["details"].get("message"):
                    st.markdown(e["details"]["message"])
                st.json(e["details"], expanded=False)
    judge = next((e for e in evals if e["kind"] == "judge" and e["details"].get("reasoning")), None)
    if judge:
        st.caption(f"Judge ({judge['details'].get('judge_model')}): {judge['details']['reasoning']}")
    owner_prior = [j for j in prior if j["role"] == "owner" and not j["promoted_to"]]
    if in_golden and not gold:
        st.caption(f"🏅 In the golden set as **{in_golden}**.")
    elif not gold and st.button("🏅 Add to the golden set", key=f"golden_{tid}", use_container_width=True):
        # an owner judgment carries corrected SQL / tags into the draft; otherwise the agent's own SQL is used
        new = promote(owner_prior[0]["id"]) if me == "owner" and owner_prior else draft_from_trace(
            tid, "trace", "Added from the Review page.")
        st.session_state.review_toast = (f"Added as draft {new}. The next golden-set test on Agent Evals answers it; "
                                         "review that answer here (source: golden) to approve it.")
        st.rerun()
    if prior:
        st.markdown("**Earlier human judgments**")
        for j in prior:
            st.caption(f"{j['role']}: correct={j['correct']}, method_ok={j['method_ok']}, tags={j['tags'] or '—'}"
                       + (f", corrected: {j['corrected_answer']}" if j["corrected_answer"] else "")
                       + (f" · promoted to {j['promoted_to']}" if j["promoted_to"] else ""))

if gold:
    with st.container(border=True):
        st.markdown(f"**🏅 Golden question {gold['id']}** · status **{gold['status']}** · {gold['qtype']} · "
                    f"answer type {gold['answer_type']} · scorer: "
                    + ("✓ matches the reference" if matched else "✗ doesn't match the reference"))
        if gold.get("notes"):
            st.caption(gold["notes"])
        if gold.get("reference_sql"):
            st.code(gold["reference_sql"], language="sql")
            ref = get_semantic().run(gold["reference_sql"], "ref")
            if ref.error:
                st.error(ref.error)
            else:
                st.caption(f"Today's reference result · {ref.row_count} rows")
                st.dataframe(pd.DataFrame(ref.rows[:10], columns=ref.columns), hide_index=True,
                             use_container_width=True)
        else:
            st.caption("No reference SQL: this question is scored on whether the agent declines or states an "
                       "assumption.")
        if me == "owner":
            st.caption("Your judgment below curates this question: agreeing with the scorer approves it, correct "
                       "SQL replaces its reference, and 'bad question' rejects it.")

st.markdown(f"#### Your judgment · as **{me}**")
with st.form(f"judge_{tid}"):
    c1, c2 = st.columns(2)
    correct = c1.radio("Is the headline answer correct?", ["yes", "no", "unsure"], horizontal=True)
    method = c2.radio("Was the method sound?", ["yes", "no", "n/a"], horizontal=True,
                      help="Compared like with like, sensible filters, honest caveats.")
    tags = st.multiselect("What went wrong (if anything)", FAILURE_TAGS)
    corrected = st.text_input("Correct answer, if you know it")
    corrected_sql = st.text_area("Correct reference SQL (optional, replaces the golden reference)" if gold else
                                 "Correct SQL (optional, becomes the golden reference)", height=80) \
        if me == "owner" else None
    bad_question = st.checkbox("This golden question is bad: reject it") if gold and me == "owner" else False
    note = st.text_input("Note")
    if st.form_submit_button("Save judgment", type="primary"):
        with SessionLocal() as s:
            s.add(HumanJudgment(trace_id=tid, role=me, reviewer=st.session_state.session_id, correct=correct,
                                method_ok=None if method == "n/a" else method == "yes", tags=tags,
                                corrected_answer=corrected or None, corrected_sql=corrected_sql or None,
                                note=note or None))
            s.commit()
        msg = "Saved. Thanks!"
        if gold and me == "owner":
            msg += " " + curate_golden(tid, correct, corrected_sql, bad_question) + "."
        st.session_state.review_toast = msg
        st.rerun()
