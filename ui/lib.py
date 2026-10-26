"""Shared UI helpers: API client, cached data loaders, theme and chart styling."""
from __future__ import annotations

import hashlib
import sys
from datetime import timedelta
from html import escape
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.db import init_db, utcnow  # noqa: E402
from app.monitor.aggregates import METRIC_LABELS, load_evals, load_traces, metric_value, timeseries  # noqa: E402

# Reference palette (validated categorical order; status colors reserved for state).
# Dark mode uses the same hues stepped for the dark surface, not an automatic flip.
_SERIES = {  # project palette: blue (text-500), orange (secondary-500), red (accent-500), gold (primary-600)
    "light": {"v1": "#1c83e3", "v2": "#e97816", "v3": "#bf980d", "all": "#1c83e3"},
    "dark": {"v1": "#499ce9", "v2": "#ee9344", "v3": "#f2cb40", "all": "#499ce9"},
}
CATEGORICAL = {
    "light": ["#1c83e3", "#e97816", "#bf980d", "#f73008", "#114e88", "#8c480d", "#77b4ee", "#f2ae73"],
    "dark": ["#499ce9", "#ee9344", "#f2cb40", "#f95939", "#a4cdf4", "#f6c9a2", "#77b4ee", "#fcac9c"],
}
STATUS = {"good": "#0ca30c", "warning": "#eebe11", "serious": "#e97816", "critical": "#f73008"}
SEVERITY = {"critical": ("critical", "🔴"), "warn": ("serious", "🟠"), "info": ("warning", "🟡")}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def ink() -> dict:
    """Chart chrome that reads on both light and dark backgrounds (Streamlit decides the theme
    client-side, so the server can't reliably know it)."""
    return {"primary": "#8c7a4a", "secondary": "#8c7a4a", "muted": "#8c7a4a", "grid": "rgba(204,153,0,0.16)",
            "axis": "rgba(204,153,0,0.45)", "surface": "rgba(0,0,0,0)", "band": "rgba(238,190,17,0.16)"}


def series() -> dict:
    # mode() only drives series hues; both steps read on either background
    return _SERIES[mode()]


def setup(title: str, icon: str = "📊") -> None:
    st.set_page_config(page_title=f"{title} · SuperEval", page_icon=icon, layout="wide")
    init_db()
    css = (Path(__file__).parent / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🛒 SuperEval")
        st.caption("Production-grade evals for an AI agent answering questions about [Israeli supermarket prices](https://www.gov.il/he/pages/cpfta_prices_regulations).")
        st.caption("Analyst: " + (f"OpenAI `{settings.agent_model}`" if settings.live_llm else "simulated (no API key)")
                   + f" · judge `{settings.judge_model}`")
        theme_switch()
        st.divider()
        from components import connect_compact

        connect_compact()


PAGES = ["", "Chat", "Price_Explorer", "Agent_Evals", "Judge_Validation", "Architecture",
         "Golden_Set"]
THEMES = {"☀️ Light": "Light", "🌙 Dark": "Dark", "💻 Auto": "System"}


def theme_switch() -> None:
    """Explicit theme control. Streamlit has no API to change the theme from Python, but it keeps the
    visitor's choice in localStorage ("stActiveTheme-/<page>-v2"), so we write that for every page and
    reload. Needs both [theme.light] and [theme.dark] in .streamlit/config.toml."""
    current = "🌙 Dark" if mode() == "dark" else "☀️ Light"
    choice = st.segmented_control("Theme", list(THEMES), default=current, key="pw_theme",
                                  label_visibility="collapsed")
    if choice and choice != current and st.session_state.get("pw_theme_applied") != choice:
        st.session_state.pw_theme_applied = choice
        value = THEMES[choice]
        keys = [f"stActiveTheme-/{p}-v2" for p in PAGES]
        import streamlit.components.v1 as components

        components.html(
            "<script>const ls = window.parent.localStorage;"
            f"for (const k of {keys!r}) ls.setItem(k, JSON.stringify({value!r}));"
            "for (const k of Object.keys(ls)) if (k.startsWith('stActiveTheme')) "
            f"ls.setItem(k, JSON.stringify({value!r}));"
            "window.parent.location.reload();</script>",
            height=0,
        )


def tiles(items: list[tuple]) -> None:
    """KPI tiles as a wrapping grid (reflows to 2 columns on phones). items: (label, value, sub, bad[, help]);
    `help` adds an ⓘ next to the label that shows the definition on hover or tap."""
    html = ""
    for label, value, sub, bad, *rest in items:
        tip = ""
        if rest and rest[0]:
            tip = f" <span class='tip' tabindex='0' data-tip='{escape(rest[0], quote=True)}'>ⓘ</span>"
        html += (f"<div class='tile{' bad' if bad else ''}'><div class='label'>{label}{tip}</div>"
                 f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>")
    st.markdown(f"<div class='tile-grid'>{html}</div>", unsafe_allow_html=True)


def user_key() -> str:
    """Who pays for live LLM calls: a hash of the visitor's IP (first X-Forwarded-For hop behind a proxy). Streamlit
    reports no IP for localhost connections, so those share one 'localhost' key. Sent to the API, which only ever
    sees this server."""
    headers = st.context.headers or {}
    ip = str(headers.get("X-Forwarded-For") or "").split(",")[0].strip() or st.context.ip_address or "localhost"
    return hashlib.sha256(str(ip).encode()).hexdigest()[:32]


# ---- API ---------------------------------------------------------------
def api(method: str, path: str, **kw):
    try:
        r = httpx.request(method, settings.api_url + path, timeout=120, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        st.error(f"API error on {path}: {exc}")
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
    INK = ink()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36 if title else 8, b=8),
        title=dict(text=title, font=dict(size=14), x=0, xanchor="left") if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=INK["axis"], tickfont=dict(color=INK["muted"]))
    fig.update_yaxes(gridcolor=INK["grid"], zeroline=False, tickfont=dict(color=INK["muted"]),
                     tickformat=".0%" if y_pct else None)
    return fig


def metric_chart(metric: str, evals: pd.DataFrame, traces: pd.DataFrame, freq: str = "3h",
                 by_version: bool = True, height: int = 240, threshold: float | None = None,
                 incidents: list[dict] | None = None) -> go.Figure:
    ts = timeseries(metric, evals, traces, freq=freq, by="agent_version" if by_version else None)
    INK, SERIES = ink(), series()
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
        # bands only; labels collide at dashboard widths, so pages list incidents in a caption
        fig.add_vrect(x0=inc["start"], x1=inc["end"], fillcolor=INK["band"], opacity=0.6, line_width=0)
    base_layout(fig, height=height, y_pct=pct, title=METRIC_LABELS.get(metric, metric))
    if metric.endswith("_ms"):
        fig.update_yaxes(ticksuffix="s")
    return fig


def incident_windows() -> list[dict]:
    """Seeded backfill incidents in absolute time (for chart annotations)."""
    from app.traffic.backfill import INCIDENTS

    today = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    for inc in INCIDENTS:
        d0 = today + timedelta(days=inc["day"])
        out.append({**inc, "start": d0 + timedelta(hours=inc["start"]), "end": d0 + timedelta(hours=inc["end"]),
                    "short": inc["kind"].replace("_", " ")})
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
