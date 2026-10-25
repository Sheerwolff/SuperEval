import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.monitor.aggregates import METRIC_LABELS, METRICS
from app.monitor.rules import wilson
from lib import base_layout, frames, ink, series, setup

setup("Version compare", "⚖️")
st.markdown("## ⚖️ Version compare")
st.caption("Every metric, per prompt version, over all traffic each version has served. Bars show the 95% Wilson "
           "interval, so you can see which differences are real and which are just small samples.")

traces, evals = frames(days=30)
SERIES = series()
versions = sorted(v for v in evals.agent_version.dropna().unique() if v in SERIES)
rows = []
for metric, (evaluator, agg) in METRICS.items():
    for version in versions:
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
for version in versions:
    d = df[df.version == version]
    fig.add_trace(go.Bar(
        y=d.metric, x=d.value, name=version, orientation="h", marker_color=SERIES[version],
        marker_line_color=ink()["surface"], marker_line_width=2,
        error_x=dict(type="data", symmetric=False, array=d.hi - d.value, arrayminus=d.value - d.lo,
                     color=ink()["secondary"], thickness=1.2, width=3),
        customdata=d[["n", "lo", "hi"]].values,
        hovertemplate="%{x:.1%} (95% CI %{customdata[1]:.1%}–%{customdata[2]:.1%}, n=%{customdata[0]})"
                      "<extra>" + version + "</extra>",
    ))
base_layout(fig, height=480)
fig.update_layout(barmode="group", hovermode="closest", bargap=0.25, bargroupgap=0.1)
fig.update_xaxes(tickformat=".0%", range=[0, 1.05], gridcolor=ink()["grid"], showgrid=True)
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

st.markdown("#### Offline golden set: prompt × model, before shipping")
from app.evals.offline_runner import RUNS  # noqa: E402

if RUNS.exists():
    runs = pd.DataFrame([json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines() if line.strip()])
    summ = pd.json_normalize(runs.pop("summary"))
    table = pd.concat([runs, summ], axis=1)
    table["cost / question"] = table.cost_usd / table.n
    cols = ["at", "version", "model", "n", "overall_pass_rate", "price_accuracy", "grounded_prices",
            "product_relevance", "language_match", "scope_handling", "cost / question", "latency_p50_ms"]
    table = table[[c for c in cols if c in table]].sort_values("at", ascending=False)
    pct = [c for c in ["overall_pass_rate", "price_accuracy", "grounded_prices", "product_relevance",
                       "language_match", "scope_handling"] if c in table]
    st.dataframe(table.style.format({**{c: "{:.0%}" for c in pct}, "cost / question": "${:.4f}",
                                     "latency_p50_ms": "{:,.0f} ms"}, na_rep="—"),
                 hide_index=True, use_container_width=True)
    st.caption("Recorded with `python -m app.evals.offline_runner --live --record [--model …]`. The same 60 "
               "fixed questions, scored by the same deterministic evaluators the production monitors use.")
else:
    st.caption("No recorded runs yet.")
