import streamlit as st

from lib import fmt, frames, kpis, setup

setup("Home", "🛒")

st.markdown(
    """
<div class="hero">
<h1>PriceWatch Evals</h1>
<p>A live LLM agent answers grocery price questions over <b>real, daily-published Israeli supermarket
data</b>. Every answer is evaluated in production, exactly where possible and by an LLM judge where
not, and monitored for regressions, drift and stale data.</p>
</div>
""",
    unsafe_allow_html=True,
)

traces, evals = frames(days=1)
metrics = ["price_accuracy", "grounded_prices", "language_match", "judge_helpfulness", "latency_p95_ms"]
vals = kpis(evals, traces, metrics)
labels = {
    "price_accuracy": "Price accuracy",
    "grounded_prices": "Grounded prices",
    "language_match": "Language match",
    "judge_helpfulness": "Judge helpfulness",
    "latency_p95_ms": "Latency p95",
}
st.markdown("##### Last 24 hours")
cols = st.columns(len(metrics) + 1)
cols[0].markdown(
    f"<div class='tile'><div class='label'>Requests</div><div class='value'>{len(traces):,}</div>"
    f"<div class='sub'>all sources</div></div>",
    unsafe_allow_html=True,
)
for col, m in zip(cols[1:], metrics):
    v, n = vals[m]
    col.markdown(
        f"<div class='tile'><div class='label'>{labels[m]}</div><div class='value'>{fmt(m, v)}</div>"
        f"<div class='sub'>n = {n:,}</div></div>",
        unsafe_allow_html=True,
    )

st.write("")
a, b, c = st.columns(3)
with a:
    st.markdown("#### 1 · Talk to the agent")
    st.write("Ask about prices in Hebrew or English. Each reply comes back with its eval scorecard: "
             "exact price checks against the database, grounding, language and scope checks, plus the judge's verdict.")
    st.page_link("pages/1_Chat.py", label="Open chat", icon="💬")
with b:
    st.markdown("#### 2 · Break production")
    st.write("Deploy **v2**, a plausible 'faster & cheaper' prompt. Send traffic and watch the SLO "
             "alerts fire on the monitor. Then roll back.")
    st.page_link("pages/2_Live_Monitor.py", label="Open live monitor", icon="📈")
with c:
    st.markdown("#### 3 · Investigate")
    st.write("Drill into any trace: tool calls, the numbers that didn't match, the judge's reasoning. "
             "Compare versions with confidence intervals.")
    st.page_link("pages/3_Trace_Explorer.py", label="Open trace explorer", icon="🔎")

st.divider()
st.markdown("#### What's monitored")
st.markdown(
    """
| Layer | Signal | How |
|---|---|---|
| **Correctness** | Every quoted ₪ price matches a tool result | deterministic, 100% of traffic |
| **Grounding** | No price stated without a price lookup; no chains we don't cover | deterministic |
| **Behaviour** | Right tool used · reply in user's language · off-topic and prompt-injection declined | deterministic |
| **Quality** | Helpfulness and groundedness on a 1–5 rubric | LLM judge, sampled, calibrated against human labels |
| **Ops** | p95 latency, cost/request, errors, daily budget | from traces |
| **Inputs** | Topic and language mix vs. last 7 days | PSI drift |
| **Data** | Each chain's price feed age vs. 36h SLA | ingest runs |
"""
)
st.caption("The 30-day history is seeded by a simulator with four scripted incidents; the 'backfill' source tag marks "
           "it everywhere. Live chats and synthetic bursts are added on top in real time.")
