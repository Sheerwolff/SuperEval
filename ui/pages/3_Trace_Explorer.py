import pandas as pd
import streamlit as st
from sqlalchemy import select

from app.db import EvalResult, SessionLocal, Trace
from lib import eval_badges, setup

setup("Trace explorer", "🔎")
st.markdown("## 🔎 Trace explorer")

f1, f2, f3, f4 = st.columns(4)
version = f1.selectbox("Version", ["any", "v1", "v2"])
source = f2.selectbox("Source", ["any", "user", "synthetic", "backfill"])
failing = f3.selectbox("Failing evaluator", ["any", "price_accuracy", "grounded_prices", "language_match",
                                             "scope_handling", "tool_use", "promo_mentioned", "judge_helpfulness"])
topic = f4.selectbox("Topic", ["any", "price_lookup", "basket", "stores", "off_topic", "injection"])

with SessionLocal() as s:
    q = select(Trace).order_by(Trace.created_at.desc()).limit(300)
    if version != "any":
        q = q.where(Trace.agent_version == version)
    if source != "any":
        q = q.where(Trace.source == source)
    if topic != "any":
        q = q.where(Trace.topic == topic)
    if failing != "any":
        q = q.where(Trace.id.in_(select(EvalResult.trace_id).where(
            EvalResult.evaluator == failing, EvalResult.passed.is_(False))))
    rows = s.execute(q).scalars().all()
    table = pd.DataFrame([{
        "time": t.created_at.strftime("%m-%d %H:%M"), "version": t.agent_version, "source": t.source,
        "topic": t.topic, "lang": t.language, "question": t.question,
        "failed": ", ".join(e.evaluator for e in t.evals if e.passed is False) or "—",
        "latency_s": round(t.latency_ms / 1000, 1), "id": t.id,
    } for t in rows])

st.caption(f"{len(table)} most recent matching traces")
if table.empty:
    st.info("No traces match these filters.")
    st.stop()

event = st.dataframe(table.drop(columns=["id"]), use_container_width=True, hide_index=True,
                     on_select="rerun", selection_mode="single-row", height=320)
sel = event.selection.rows[0] if event.selection.rows else 0
trace_id = table.iloc[sel]["id"]

with SessionLocal() as s:
    t = s.get(Trace, trace_id)
    evals = [{"evaluator": e.evaluator, "kind": e.kind, "score": e.score, "passed": e.passed, "details": e.details}
             for e in t.evals]

st.divider()
a, b = st.columns([3, 2], gap="large")
with a:
    st.markdown(f"#### Trace `{t.id[:8]}` · <span class='pill pill-{t.agent_version}'>{t.agent_version}</span>",
                unsafe_allow_html=True)
    st.markdown(f"**Q:** {t.question}")
    st.markdown(f"**A:** {t.answer}")
    st.caption(f"{t.created_at:%Y-%m-%d %H:%M} UTC · {t.source} · {t.model} · prompt {t.prompt_hash} · "
               f"snapshot {t.snapshot_date} · {t.latency_ms / 1000:.1f}s · "
               f"{t.input_tokens}+{t.output_tokens} tokens · ${t.cost_usd:.4f}")
    st.markdown("**Tool calls**")
    if not t.tool_calls:
        st.caption("None. The model answered without looking anything up.")
    for c in t.tool_calls:
        with st.expander(f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in (c.get('args') or {}).items())})"):
            st.json(c.get("result") if "result" in c else {"error": c.get("error")})
with b:
    st.markdown("#### Evals")
    st.markdown(eval_badges(evals), unsafe_allow_html=True)
    for e in evals:
        if e["passed"] is False or e["kind"] == "judge":
            with st.expander(f"{'✗' if e['passed'] is False else '•'} {e['evaluator']}", expanded=e["passed"] is False):
                st.json(e["details"])
