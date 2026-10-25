from datetime import timedelta

import pandas as pd
import streamlit as st
from sqlalchemy import select

from app.db import Alert, SessionLocal, utcnow
from app.monitor.aggregates import metric_value
from lib import SEVERITY, api, deployment, fmt, frames, incident_windows, metric_chart, setup

setup("Live monitor", "📈")
st.markdown("## 📈 Live monitor")

dep = deployment() or {"current": "v1", "versions": {}, "history": []}
current = dep["current"]

# ---- controls ---------------------------------------------------------------
with st.container(border=True):
    c1, c2, c3 = st.columns([2.2, 1.6, 1.2])
    with c1:
        st.markdown(f"**Deployed:** <span class='pill pill-{current}'>{current}</span> "
                    f"{dep['versions'].get(current, {}).get('label', '')}", unsafe_allow_html=True)
        st.caption(dep["versions"].get(current, {}).get("description", ""))
    with c2:
        other = "v2" if current == "v1" else "v1"
        verb = "🚨 Deploy v2 (buggy)" if other == "v2" else "↩️ Roll back to v1"
        if st.button(verb, use_container_width=True, type="primary" if other == "v1" else "secondary"):
            api("POST", "/deployment", json={"version": other, "note": "toggled from the live monitor"})
            st.cache_data.clear()
            st.rerun()
    with c3:
        if st.button("⚡ Send 40 requests", use_container_width=True,
                     help="Synthetic shoppers hit the deployed version (simulated, free), then monitors run"):
            with st.spinner("Generating traffic and running monitors…"):
                r = api("POST", "/simulate", json={"n": 40})
            st.cache_data.clear()
            if r:
                st.toast(f"{r['generated']} requests on {r['version']} · {r['opened']} alert(s) opened, "
                         f"{r['resolved']} resolved")
    st.caption("Try it: deploy v2, send 40 requests, and watch the critical alerts open. Roll back and send "
               "traffic again: SLOs are judged on the version that's serving, so they resolve.")

# ---- open alerts -------------------------------------------------------------
with SessionLocal() as s:
    alerts = s.execute(select(Alert).order_by(Alert.created_at.desc()).limit(200)).scalars().all()
open_alerts = [a for a in alerts if a.status == "open"]
alerting = {a.metric for a in open_alerts}


def alert_html(a: Alert) -> str:
    _, icon = SEVERITY.get(a.severity, ("warning", "🟡"))
    state = "" if a.status == "open" else " alert-resolved"
    when = a.created_at.strftime("%b %d %H:%M")
    resolved = f" · resolved {a.resolved_at.strftime('%b %d %H:%M')}" if a.resolved_at else ""
    return (f"<div class='alert-row alert-{a.severity}{state}'>{icon} <b>{a.severity.upper()}</b> · {a.message}"
            f"<div class='meta'>{a.rule} · opened {when} UTC{resolved}</div></div>")


left, right = st.columns([3, 2], gap="large")
with right:
    st.markdown(f"#### Alerts · {len(open_alerts)} open")
    if not open_alerts:
        st.markdown("<div class='alert-row'>✅ No open alerts</div>", unsafe_allow_html=True)
    for a in open_alerts:
        st.markdown(alert_html(a), unsafe_allow_html=True)
    with st.expander(f"History ({len(alerts) - len(open_alerts)} resolved)", expanded=False):
        for a in [a for a in alerts if a.status != "open"][:40]:
            st.markdown(alert_html(a), unsafe_allow_html=True)

# ---- KPIs, last 3h ---------------------------------------------------------------
with left:
    traces, evals = frames(days=30)
    now = utcnow()
    # same scope as the SLO rules: the deployed version over the alerting window
    recent_t = traces[(traces.created_at >= now - timedelta(hours=3)) & (traces.agent_version == current)]
    recent_e = evals[(evals.created_at >= now - timedelta(hours=3)) & (evals.agent_version == current)]
    st.markdown(f"#### {current}, last 3 hours (the alerting window)")
    metrics = [("price_accuracy", 0.95, "<"), ("grounded_prices", 0.95, "<"), ("language_match", 0.90, "<"),
               ("scope_handling", 0.90, "<"), ("latency_p95_ms", 8000, ">"), ("cost_per_request", None, None)]
    cols = st.columns(3)
    for i, (m, slo, op) in enumerate(metrics):
        v, n = metric_value(m, recent_e, recent_t)
        bad = m in alerting  # red only when a statistically significant alert is open, not on one miss
        sub = f"n = {n}" + (f" · SLO {op} {fmt(m, slo)}" if slo else "")
        cols[i % 3].markdown(
            f"<div class='tile{' bad' if bad else ''}'><div class='label'>"
            f"{m.replace('_', ' ').replace('p95 ms', 'p95').capitalize()}</div>"
            f"<div class='value'>{fmt(m, v)}</div><div class='sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )
        if i % 3 == 2:
            st.write("")

# ---- time series ----------------------------------------------------------------
st.divider()
h1, h2 = st.columns([3, 1])
h1.markdown("#### Trends by version")
rng = h2.segmented_control("Range", ["24h", "7d", "30d"], default="30d", label_visibility="collapsed")
days = {"24h": 1, "7d": 7, "30d": 30}[rng or "30d"]
freq = {"24h": "1h", "7d": "3h", "30d": "6h"}[rng or "30d"]
cut = now - timedelta(days=days)
t_r, e_r = traces[traces.created_at >= cut], evals[evals.created_at >= cut]
incidents = [i for i in incident_windows() if i["end"] >= cut]

g1, g2 = st.columns(2)
g1.plotly_chart(metric_chart("price_accuracy", e_r, t_r, freq=freq, threshold=0.95, incidents=incidents),
                use_container_width=True)
g2.plotly_chart(metric_chart("judge_helpfulness", e_r, t_r, freq=freq, incidents=incidents),
                use_container_width=True)
g3, g4 = st.columns(2)
g3.plotly_chart(metric_chart("language_match", e_r, t_r, freq=freq, threshold=0.90, incidents=incidents),
                use_container_width=True)
g4.plotly_chart(metric_chart("latency_p95_ms", e_r, t_r, freq=freq, threshold=8000, by_version=False,
                             incidents=incidents), use_container_width=True)

st.markdown("#### Input mix")
if not t_r.empty:
    import plotly.express as px

    from lib import base_layout

    mix = (t_r.assign(day=t_r.created_at.dt.floor("1D")).groupby(["day", "topic"]).size()
           .groupby(level=0).transform(lambda s: s / s.sum()).rename("share").reset_index())
    order = ["price_lookup", "basket", "stores", "off_topic", "injection"]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig = px.bar(mix, x="day", y="share", color="topic", category_orders={"topic": order},
                 color_discrete_sequence=palette)
    fig.update_traces(marker_line_width=1, marker_line_color="#fcfcfb",
                      hovertemplate="%{x|%b %d}: %{y:.0%}<extra>%{fullData.name}</extra>")
    base_layout(fig, height=220, y_pct=True)
    fig.update_layout(bargap=0.15, hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Daily topic share. The PSI drift rule compares the last 24h against the previous 7 days.")

with st.expander("Deployment history"):
    st.dataframe(pd.DataFrame(dep["history"]), use_container_width=True, hide_index=True)
