"""Rendering an analyst trace: chart from the agent's spec, and the queries behind the answer.
Shared by the Chat, Trace Explorer and Review pages."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import base_layout, ink, series


def render_chart(trace: dict, key: str = "") -> None:
    spec = (trace.get("analysis") or {}).get("chart")
    if not spec:
        return
    q = next((q for q in trace.get("tool_calls") or [] if q.get("id") == spec.get("query_id")), None)
    if not q or q.get("error"):
        return
    df = pd.DataFrame(q["rows"], columns=q["columns"])
    if spec["x"] not in df or spec["y"] not in df:
        st.caption(f"⚠️ Chart skipped: columns {spec['x']!r}/{spec['y']!r} aren't in {spec['query_id']}.")
        return
    S, INK = series(), ink()
    df = df.head(25)
    kind = spec["type"]
    if kind in ("bar", "horizontal_bar"):
        horiz = kind == "horizontal_bar"
        fig = go.Figure(go.Bar(x=df[spec["y"]] if horiz else df[spec["x"]], y=df[spec["x"]] if horiz else df[spec["y"]],
                               orientation="h" if horiz else "v", marker_color=S["v1"],
                               hovertemplate=("%{y}: %{x}" if horiz else "%{x}: %{y}") + "<extra></extra>"))
        if horiz:
            fig.update_yaxes(autorange="reversed", showgrid=False)
    elif kind == "line":
        fig = go.Figure(go.Scatter(x=df[spec["x"]], y=df[spec["y"]], mode="lines+markers",
                                   line=dict(width=2, color=S["v1"]), marker=dict(size=8)))
    else:
        fig = go.Figure(go.Scatter(x=df[spec["x"]], y=df[spec["y"]], mode="markers", marker=dict(size=9, color=S["v1"])))
    base_layout(fig, height=300 if kind != "horizontal_bar" else 60 + 26 * len(df), title=spec.get("title"))
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(showgrid=kind == "horizontal_bar", gridcolor=INK["grid"])
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{trace['id']}{key}")


def render_queries(trace: dict, expanded: bool = False) -> None:
    queries = trace.get("tool_calls") or []
    if not queries:
        return
    with st.expander(f"🧮 {len(queries)} quer{'ies' if len(queries) != 1 else 'y'} behind this answer", expanded=expanded):
        cited = {c["query_id"] for c in (trace.get("analysis") or {}).get("claims", [])}
        for q in queries:
            tag = " · cited" if q["id"] in cited else ""
            st.markdown(f"**{q['id']}**{tag} · {q.get('row_count', 0)} rows · {q.get('ms', 0):.0f} ms")
            st.code(q["sql"], language="sql")
            if q.get("error"):
                st.error(q["error"])
            elif q.get("rows"):
                st.dataframe(pd.DataFrame(q["rows"][:20], columns=q["columns"]), hide_index=True,
                             use_container_width=True)
