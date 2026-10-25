import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.monitor.aggregates import METRIC_LABELS, METRICS
from app.monitor.rules import wilson
from lib import SERIES, base_layout, frames, setup

setup("Version compare", "⚖️")
st.markdown("## ⚖️ Version compare")
st.caption("Every metric, v1 vs v2, over all traffic each version has served. Bars show the 95% Wilson "
           "interval, so you can see which differences are real and which are just small samples.")

traces, evals = frames(days=30)
rows = []
for metric, (evaluator, agg) in METRICS.items():
    for version in ["v1", "v2"]:
        sub = evals[(evals.evaluator == evaluator) & (evals.agent_version == version) & evals.score.notna()]
        if sub.empty:
            continue
        n = len(sub)
        if agg == "rate":
            p = float(sub.passed.astype(float).mean())
            lo, hi = wilson(p, n)
        else:
            p = float(sub.score.mean())
            se = float(sub.score.std(ddof=1) / n ** 0.5) if n > 1 else 0.0
            lo, hi = p - 1.96 * se, p + 1.96 * se
        rows.append({"metric": METRIC_LABELS[metric], "version": version, "value": p, "lo": lo, "hi": hi, "n": n})
df = pd.DataFrame(rows)

if df.empty:
    st.info("No data yet.")
    st.stop()

fig = go.Figure()
for version in ["v1", "v2"]:
    d = df[df.version == version]
    fig.add_trace(go.Bar(
        y=d.metric, x=d.value, name=version, orientation="h", marker_color=SERIES[version],
        marker_line_color="#fcfcfb", marker_line_width=2,
        error_x=dict(type="data", symmetric=False, array=d.hi - d.value, arrayminus=d.value - d.lo,
                     color="#52514e", thickness=1.2, width=3),
        customdata=d[["n", "lo", "hi"]].values,
        hovertemplate="%{x:.1%} (95% CI %{customdata[1]:.1%}–%{customdata[2]:.1%}, n=%{customdata[0]})"
                      "<extra>" + version + "</extra>",
    ))
base_layout(fig, height=480)
fig.update_layout(barmode="group", hovermode="closest", bargap=0.25, bargroupgap=0.1)
fig.update_xaxes(tickformat=".0%", range=[0, 1.05], gridcolor="#e1e0d9", showgrid=True)
fig.update_yaxes(autorange="reversed", showgrid=False)
st.plotly_chart(fig, use_container_width=True)
st.caption("Judge scores are rescaled from 1–5 to 0–100%.")

ops = traces.groupby("agent_version").agg(
    requests=("id", "count"),
    p50_latency_s=("latency_ms", lambda s: round(s.quantile(0.5) / 1000, 2)),
    p95_latency_s=("latency_ms", lambda s: round(s.quantile(0.95) / 1000, 2)),
    avg_tokens=("input_tokens", "mean"),
    cost_per_request=("cost_usd", "mean"),
).round({"avg_tokens": 0, "cost_per_request": 5})
st.markdown("#### Cost & latency")
st.dataframe(ops, use_container_width=True)
st.caption("v2 was meant to be faster and cheaper. It is, a little, and it's wrong most of the time. "
           "That trade is exactly what an eval-gated deploy should block.")

with st.expander("Table view"):
    st.dataframe(df.assign(value=df.value.round(3), lo=df.lo.round(3), hi=df.hi.round(3)),
                 use_container_width=True, hide_index=True)
