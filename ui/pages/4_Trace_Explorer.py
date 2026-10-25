import pandas as pd
import streamlit as st
from sqlalchemy import select

from app.db import EvalResult, SessionLocal, Trace
from lib import eval_badges, setup
from render import render_chart, render_queries

setup("Trace explorer", "🔎")
st.markdown("## 🔎 Trace explorer")
st.caption("Every answer the analyst gave, with the SQL it ran and every automated verdict. Filter to a failing "
           "check to see exactly what went wrong.")

FAILING = ["any", "grounding", "method_lint", "canonical_check", "sql_health", "completion", "chart_lint",
           "decline_check", "assumption_check", "language_match", "judge_method", "judge_groundedness",
           "judge_helpfulness"]
TOPICS = ["any", "comparison", "lookup", "spread", "promo", "geo", "ambiguous", "unanswerable", "injection"]
preset = st.session_state.pop("trace_filter_failing", None)
f1, f2, f3 = st.columns(3)
failing = f1.selectbox("Failing check", FAILING, index=FAILING.index(preset) if preset in FAILING else 0)
topic = f2.selectbox("Question type", TOPICS)
source = f3.selectbox("Source", ["any", "user", "synthetic", "backfill"])

with SessionLocal() as s:
    q = select(Trace).order_by(Trace.created_at.desc()).limit(300)
    if source != "any":
        q = q.where(Trace.source == source)
    if topic != "any":
        q = q.where(Trace.topic == topic)
    if failing != "any":
        q = q.where(Trace.id.in_(select(EvalResult.trace_id).where(
            EvalResult.evaluator == failing, EvalResult.passed.is_(False))))
    rows = s.execute(q).scalars().all()
    table = pd.DataFrame([{
        "time": t.created_at.strftime("%m-%d %H:%M"), "source": t.source, "type": t.topic, "lang": t.language,
        "question": t.question, "failed": ", ".join(e.evaluator for e in t.evals if e.passed is False) or "—",
        "latency_s": round(t.latency_ms / 1000, 1), "id": t.id} for t in rows])

st.caption(f"{len(table)} most recent matching traces")
if table.empty:
    st.info("No traces match these filters.")
    st.stop()
event = st.dataframe(table.drop(columns=["id"]), use_container_width=True, hide_index=True, on_select="rerun",
                     selection_mode="single-row", height=300)
sel = event.selection.rows[0] if event.selection.rows else 0
trace_id = table.iloc[sel]["id"]

with SessionLocal() as s:
    t = s.get(Trace, trace_id)
    tr = {"id": t.id, "question": t.question, "answer": t.answer, "analysis": t.analysis, "tool_calls": t.tool_calls,
          "error": t.error}
    meta = (f"{t.created_at:%Y-%m-%d %H:%M} UTC · {t.source} · {t.model} · {t.steps} steps · "
            f"{t.latency_ms / 1000:.1f}s · {t.input_tokens + t.output_tokens:,} tokens · ${t.cost_usd:.4f}")
    evals = [{"evaluator": e.evaluator, "kind": e.kind, "passed": e.passed, "details": e.details} for e in t.evals]

st.divider()
a, b = st.columns([3, 2], gap="large")
with a:
    st.markdown(f"#### {tr['question']}")
    st.caption(meta)
    st.markdown(tr["answer"] or f"*(no answer: {tr['error']})*")
    render_chart(tr, key="_explorer")
    render_queries(tr, expanded=failing != "any")
with b:
    st.markdown("#### Verdicts")
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
