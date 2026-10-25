import json
from datetime import timedelta

import plotly.express as px
import streamlit as st
from sqlalchemy import select

from app.db import Alert, SessionLocal, utcnow
from app.evals.bench import RUNS
from app.evals.human import judgments_frame, override_rate
from app.evals.judge_monitor import JUDGE_METRICS, canary_runs, load_canaries, load_judge_frame
from app.monitor.aggregates import metric_value
from lib import (
    CATEGORICAL,
    SEVERITY,
    api,
    base_layout,
    fmt,
    frames,
    incident_windows,
    ink,
    metric_chart,
    mode,
    setup,
    tiles,
)

setup("Live monitor", "📈")
st.markdown("## 📈 Live monitor")
st.caption("Every analyst answer is evaluated as it's served: deterministic checks on 100% of traffic (Layer 1) and a "
           "sampled LLM judge, whose own reliability is monitored too (Layer 2). SLOs only fire when a breach is "
           "statistically clear.")

if toast := st.session_state.pop("fault_toast", None):
    st.toast(toast)
traces, evals = frames(days=30)
now = utcnow()
day_t = traces[traces.created_at >= now - timedelta(days=1)]
day_e = evals[evals.created_at >= now - timedelta(days=1)]

# ---- headline KPIs -------------------------------------------------------------------
last_bench = None
if RUNS.exists():
    lines = [json.loads(x) for x in RUNS.read_text(encoding="utf-8").splitlines() if x.strip()]
    live = [r for r in lines if r.get("live")]
    last_bench = (live or lines)[-1] if (live or lines) else None
g, g_n = metric_value("grounding", day_e, day_t)
hj = judgments_frame()
ovr, ovr_n = override_rate(hj)
completed = day_t[day_t.error.isna()]
cpt = (day_t.cost_usd.sum() / len(completed)) if len(completed) else None
tiles([
    ("Correctness (offline)", f"{last_bench['summary']['overall']['rate']:.0%}" if last_bench else "—",
     f"golden set · {last_bench['model']} · {last_bench['at'][:10]}" if last_bench else "run the benchmark", False),
    ("Grounding (online)", fmt("grounding", g), f"last 24h · n = {g_n:,}", False),
    ("Human override rate", "—" if ovr is None else f"{ovr:.0%}", f"{ovr_n} reviewed traces", False),
    ("Cost per completed task", "—" if cpt is None else f"${cpt:.4f}", "agent + judge · last 24h", False),
])

# ---- fault injection ---------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Try it: inject a fault into production traffic**")
    st.caption("Each button sends 40 synthetic questions through the simulated analyst (free), then the monitors run. "
               "Watch which layer catches what.")
    faults = [("✅ Normal traffic", "none"), ("🧠 Model regression", "degraded_method"),
              ("🗄️ Schema change", "schema_broken"), ("⚖️ Lenient judge", "lenient_judge"),
              ("🐢 Slow LLM", "slow_llm")]
    for col, (label, fault) in zip(st.columns(len(faults)), faults, strict=True):
        if col.button(label, use_container_width=True, key=f"fault_{fault}"):
            with st.spinner("Sending traffic and running monitors…"):
                r = api("POST", "/simulate", json={"n": 40, "fault": fault})
            st.cache_data.clear()
            if r:  # rerun so the tiles above reflect the new traffic; the toast survives via session state
                st.session_state.fault_toast = (f"{r['generated']} requests · {r['opened']} alert(s) opened, "
                                                f"{r['resolved']} resolved")
                st.rerun()
    st.caption("**Model regression**: the analyst averages each chain's own products (the Simpson's trap). Every "
               "number is still grounded, so only the method checks fire. **Schema change**: queries fail. "
               "**Lenient judge**: Layer 1 looks *better*; only Layer 2 (canaries) notices.")

# ---- layer 1 & 2 status, alerts -----------------------------------------------------
with SessionLocal() as s:
    alerts = s.execute(select(Alert).order_by(Alert.created_at.desc()).limit(200)).scalars().all()
open_alerts = [a for a in alerts if a.status == "open"]
alerting = {a.metric for a in open_alerts}


def alert_html(a: Alert) -> str:
    _, icon = SEVERITY.get(a.severity, ("warning", "🟡"))
    state = "" if a.status == "open" else " alert-resolved"
    resolved = f" · resolved {a.resolved_at:%b %d %H:%M}" if a.resolved_at else ""
    layer = "Layer 2 · judge" if a.layer == "judge" else "Layer 1 · analyst"
    return (f"<div class='alert-row alert-{a.severity}{state}'>{icon} <b>{a.severity.upper()}</b> · {a.message}"
            f"<div class='meta'>{layer} · {a.rule} · opened {a.created_at:%b %d %H:%M} UTC{resolved}</div></div>")


left, right = st.columns([3, 2], gap="large")
with left:
    recent_t = traces[traces.created_at >= now - timedelta(hours=3)]
    recent_e = evals[evals.created_at >= now - timedelta(hours=3)]
    st.markdown("#### Layer 1 · the analyst, last 3 hours")
    slos = [("completion", 0.95), ("grounding", 0.98), ("sql_health", 0.95), ("method_lint", 0.90),
            ("decline_check", 0.90), ("latency_p95_ms", 20000)]
    items = []
    for m, slo in slos:
        v, n = metric_value(m, recent_e, recent_t)
        op = "≤" if m.endswith("_ms") else "≥"
        label = {"completion": "Task completion", "grounding": "Grounding", "sql_health": "SQL health",
                 "method_lint": "Sound method", "decline_check": "Correct declines", "latency_p95_ms": "Latency p95"}[m]
        items.append((label, fmt(m, v), f"n = {n} · SLO {op} {fmt(m, slo)}", m in alerting))
    tiles(items)
    st.markdown("#### Layer 2 · is the judge still trustworthy?")
    jf = load_judge_frame(now - timedelta(days=3), now)
    runs = canary_runs(load_canaries(now - timedelta(days=2)))
    rec, rec_n = JUDGE_METRICS["judge_recall_on_bad"][1](jf)
    tiles([("Judge catches known-bad answers", "—" if rec is None else f"{rec:.0%}", f"72h · n = {rec_n} · SLO ≥ 80%",
            "judge_recall_on_bad" in alerting),
           ("Latest judge canary run", "—" if runs.empty else f"{runs.iloc[-1].accuracy:.0%}",
            "no runs yet" if runs.empty else f"{runs.iloc[-1].created_at:%b %d %H:%M} · SLO ≥ 80%",
            "canary_accuracy" in alerting)])
    st.page_link("pages/6_Judge_Monitor.py", label="Open the judge monitor", icon="⚖️")
with right:
    st.markdown(f"#### Alerts · {len(open_alerts)} open")
    if not open_alerts:
        st.markdown("<div class='alert-row'>✅ No open alerts</div>", unsafe_allow_html=True)
    for a in open_alerts:
        st.markdown(alert_html(a), unsafe_allow_html=True)
    with st.expander(f"History ({len(alerts) - len(open_alerts)} resolved)"):
        for a in [a for a in alerts if a.status != "open"][:40]:
            st.markdown(alert_html(a), unsafe_allow_html=True)

# ---- trends ---------------------------------------------------------------------------
st.divider()
h1, h2 = st.columns([3, 1])
h1.markdown("#### Trends")
rng = h2.segmented_control("Range", ["24h", "7d", "30d"], default="30d", label_visibility="collapsed")
days = {"24h": 1, "7d": 7, "30d": 30}[rng or "30d"]
freq = {"24h": "1h", "7d": "3h", "30d": "6h"}[rng or "30d"]
cut = now - timedelta(days=days)
t_r, e_r = traces[traces.created_at >= cut], evals[evals.created_at >= cut]
incidents = [i for i in incident_windows() if i["end"] >= cut]
g1, g2 = st.columns(2)
g1.plotly_chart(metric_chart("grounding", e_r, t_r, freq=freq, threshold=0.98, by_version=False, incidents=incidents),
                use_container_width=True)
g2.plotly_chart(metric_chart("method_lint", e_r, t_r, freq=freq, threshold=0.90, by_version=False,
                             incidents=incidents), use_container_width=True)
g3, g4 = st.columns(2)
g3.plotly_chart(metric_chart("completion", e_r, t_r, freq=freq, threshold=0.95, by_version=False,
                             incidents=incidents), use_container_width=True)
g4.plotly_chart(metric_chart("latency_p95_ms", e_r, t_r, freq=freq, threshold=20000, by_version=False,
                             incidents=incidents), use_container_width=True)
if incidents:
    st.caption("Shaded: seeded incidents · " + " · ".join(
        f"**{i['start']:%b %d}** {i['title']}" for i in sorted(incidents, key=lambda i: i["start"])))

st.markdown("#### Input mix")
if not t_r.empty:
    mix = (t_r.assign(day=t_r.created_at.dt.floor("1D")).groupby(["day", "topic"]).size()
           .groupby(level=0).transform(lambda s: s / s.sum()).rename("share").reset_index())
    order = ["comparison", "lookup", "spread", "promo", "geo", "ambiguous", "unanswerable", "injection"]
    fig = px.bar(mix, x="day", y="share", color="topic", category_orders={"topic": order},
                 color_discrete_sequence=CATEGORICAL[mode()])
    fig.update_traces(marker_line_width=0, hovertemplate="%{x|%b %d}: %{y:.0%}<extra>%{fullData.name}</extra>")
    base_layout(fig, height=240, y_pct=True)
    fig.update_layout(bargap=0.15, hovermode="closest")
    fig.update_xaxes(gridcolor=ink()["grid"])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Daily question-type share. The PSI drift rule compares the last 24h with the previous 7 days.")
