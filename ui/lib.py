"""Shared UI helpers: API client, cached data loaders, theme and chart styling."""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.db import init_db, utcnow  # noqa: E402
from app.monitor.aggregates import METRIC_LABELS, load_evals, load_traces, metric_value, timeseries  # noqa: E402

# reference palette (validated categorical order; status colors reserved for state)
SERIES = {"v1": "#2a78d6", "v2": "#eb6834", "all": "#2a78d6"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEVERITY = {"critical": ("critical", "🔴"), "warn": ("serious", "🟠"), "info": ("warning", "🟡")}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}


def setup(title: str, icon: str = "📊") -> None:
    st.set_page_config(page_title=f"{title} · PriceWatch Evals", page_icon=icon, layout="wide")
    init_db()
    css = (Path(__file__).parent / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🛒 PriceWatch Evals")
        st.caption("Production eval monitoring for an LLM shopping agent over real Israeli supermarket prices.")
        dep = deployment()
        if dep:
            v = dep["current"]
            st.markdown(f"**Live version:** <span class='pill pill-{v}'>{v}</span>", unsafe_allow_html=True)
        st.caption("LLM: " + (f"OpenAI `{settings.agent_model}`" if settings.live_llm else "simulated (no API key)"))


# ---- API ---------------------------------------------------------------
def api(method: str, path: str, **kw):
    try:
        r = httpx.request(method, settings.api_url + path, timeout=120, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        st.error(f"API error on {path}: {exc}")
        return None


@st.cache_data(ttl=5, show_spinner=False)
def deployment():
    try:
        return httpx.get(settings.api_url + "/deployment", timeout=5).json()
    except httpx.HTTPError:
        return None


# ---- data ----------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def frames(days: float = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    since = utcnow() - timedelta(days=days)
    return load_traces(since), load_evals(since)


def kpis(evals: pd.DataFrame, traces: pd.DataFrame, metrics: list[str]) -> dict:
    return {m: metric_value(m, evals, traces) for m in metrics}


def fmt(metric: str, v: float | None) -> str:
    if v is None:
        return "—"
    if metric.endswith("_ms"):
        return f"{v / 1000:.1f}s"
    if metric == "cost_per_request":
        return f"${v:.4f}"
    if metric.startswith("judge_"):
        return f"{1 + 4 * v:.2f} / 5"
    return f"{v:.1%}"


# ---- charts ----------------------------------------------------------------
def base_layout(fig: go.Figure, height: int = 260, y_pct: bool = False, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36 if title else 8, b=8),
        title=dict(text=title, font=dict(size=14, color=INK["primary"]), x=0, xanchor="left") if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK["secondary"]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=INK["axis"], tickfont=dict(color=INK["muted"]))
    fig.update_yaxes(gridcolor=INK["grid"], zeroline=False, tickfont=dict(color=INK["muted"]),
                     tickformat=".0%" if y_pct else None)
    return fig


def metric_chart(metric: str, evals: pd.DataFrame, traces: pd.DataFrame, freq: str = "3h",
                 by_version: bool = True, height: int = 240, threshold: float | None = None,
                 incidents: list[dict] | None = None) -> go.Figure:
    ts = timeseries(metric, evals, traces, freq=freq, by="agent_version" if by_version else None)
    fig = go.Figure()
    pct = metric not in ("latency_p95_ms", "cost_per_request") and not metric.startswith("judge_")
    if not ts.empty:
        groups = ts.groupby("agent_version") if by_version else [("all", ts)]
        for version, g in groups:
            g = g.sort_values("bucket")
            y = g.value if not metric.startswith("judge_") else 1 + 4 * g.value
            if metric.endswith("_ms"):
                y = g.value / 1000
            # versions are discontinuous in time; keep gaps instead of drawing across them
            full = pd.date_range(ts.bucket.min(), ts.bucket.max(), freq=freq)
            s = pd.Series(y.values, index=g.bucket).reindex(full)
            n = pd.Series(g.n.values, index=g.bucket).reindex(full)
            fig.add_trace(go.Scatter(
                x=full, y=s, name=version, mode="lines+markers" if s.notna().sum() < 40 else "lines",
                line=dict(width=2, color=SERIES.get(version, SERIES["all"])),
                marker=dict(size=8), connectgaps=False,
                customdata=n, hovertemplate="%{y:" + (".1%" if pct else ".2f") + "} (n=%{customdata})<extra>"
                + str(version) + "</extra>",
            ))
    if threshold is not None:
        t = threshold / 1000 if metric.endswith("_ms") else threshold
        fig.add_hline(y=t, line=dict(color=STATUS["critical"], width=1, dash="dot"),
                      annotation_text="SLO", annotation_position="top left",
                      annotation_font=dict(color=INK["muted"], size=11))
    for inc in incidents or []:
        fig.add_vrect(x0=inc["start"], x1=inc["end"], fillcolor=INK["grid"], opacity=0.5, line_width=0,
                      annotation_text=inc.get("short", ""), annotation_position="top left",
                      annotation_font=dict(size=10, color=INK["muted"]))
    base_layout(fig, height=height, y_pct=pct, title=METRIC_LABELS.get(metric, metric))
    if metric.endswith("_ms"):
        fig.update_yaxes(ticksuffix="s")
    return fig


def incident_windows() -> list[dict]:
    """Seeded backfill incidents in absolute time (for chart annotations)."""
    from app.traffic.backfill import INCIDENTS

    today = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    short = {"bad_deploy": "bad deploy", "stale_feed": "stale feed", "latency": "latency", "drift": "drift"}
    out = []
    for inc in INCIDENTS:
        d0 = today + timedelta(days=inc["day"])
        out.append({**inc, "start": d0 + timedelta(hours=inc["start"]), "end": d0 + timedelta(hours=inc["end"]),
                    "short": short[inc["kind"]]})
    return out


def eval_badges(evals: list[dict]) -> str:
    chips = []
    for e in evals:
        if e["passed"] is None:
            continue
        cls = "ok" if e["passed"] else "bad"
        name = e["evaluator"].replace("judge_", "judge · ").replace("_", " ")
        extra = ""
        if e["evaluator"].startswith("judge_") and "raw" in e.get("details", {}):
            extra = f" {e['details']['raw']}/5"
        chips.append(f"<span class='chip chip-{cls}'>{'✓' if e['passed'] else '✗'} {name}{extra}</span>")
    return " ".join(chips)
