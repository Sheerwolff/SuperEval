import random
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from app.db import Alert, SessionLocal, utcnow
from app.evals.analyst_judge import RUBRIC_HASH
from app.evals.calibration import agreement
from app.evals.judge_monitor import JUDGE_METRICS, canary_runs, load_canaries, load_judge_frame
from components import tour_banner
from lib import SEVERITY, base_layout, incident_windows, ink, series, setup, tiles

setup("Judge monitor", "⚖️")
tour_banner("judge")
st.markdown("## ⚖️ Judge monitor · Layer 2")
st.caption("Layer 1 trusts the LLM judge's verdicts about the agent. If the judge silently goes lenient, after a "
           "provider model update or a rubric edit, the agent looks *better* while it isn't. So the judge gets its "
           "own evals, SLOs and alerts.")


@st.cache_data(ttl=15, show_spinner=False)
def data(days: int = 30):
    since = utcnow() - timedelta(days=days)
    return load_judge_frame(since), load_canaries(since)


judge, canaries = data()
now = utcnow()
INK, S = ink(), series()

with SessionLocal() as s:
    j_alerts = s.execute(select(Alert).where(Alert.layer == "judge").order_by(Alert.created_at.desc())
                         .limit(50)).scalars().all()
open_alerts = [a for a in j_alerts if a.status == "open"]

overview, canary_tab, bias_tab, human_tab = st.tabs(["Overview", "Canaries", "Bias & consistency",
                                                     "Calibrate with human labels"])

# ---- overview -----------------------------------------------------------------
with overview:
    last7 = judge[judge.created_at >= now - timedelta(days=7)]
    runs = canary_runs(canaries)
    last_run = runs.iloc[-1] if len(runs) else None
    rec, rec_n = JUDGE_METRICS["judge_recall_on_bad"][1](last7)
    agr, agr_n = JUDGE_METRICS["judge_agreement"][1](last7)
    ff, ff_n = JUDGE_METRICS["judge_false_fail"][1](last7)
    flip, flip_n = JUDGE_METRICS["judge_flip_rate"][1](last7)
    err, err_n = JUDGE_METRICS["judge_error_rate"][1](last7)
    alerting = {a.metric for a in open_alerts}

    def pct(v):
        return "—" if v is None else f"{v:.0%}"

    tiles([
        ("Catches known-bad answers", pct(rec), f"n = {rec_n} · SLO ≥ 80%", "judge_recall_on_bad" in alerting),
        ("Latest canary run", pct(last_run.accuracy) if last_run is not None else "—",
         f"{int(last_run.checks)} checks · SLO ≥ 85%" if last_run is not None else "no runs yet",
         "canary_accuracy" in alerting),
        ("Agrees with ground truth", pct(agr), f"n = {agr_n:,}", False),
        ("Fails known-good answers", pct(ff), f"n = {ff_n:,}", False),
        ("Flip rate when re-judged", pct(flip), f"n = {flip_n} · SLO ≤ 15%", "judge_flip_rate" in alerting),
        ("Judge errors", pct(err), f"n = {err_n:,} · SLO ≤ 5%", "judge_error_rate" in alerting),
    ])
    st.caption(f"Last 7 days. Rubric `{RUBRIC_HASH}`. Ground truth = the deterministic price checks: when they say an "
               "answer has a wrong or ungrounded price, the judge should fail it too.")

    st.markdown(f"#### Layer 2 alerts · {len(open_alerts)} open")
    if not j_alerts:
        st.markdown("<div class='alert-row'>No judge alerts yet.</div>", unsafe_allow_html=True)
    for a in j_alerts[:8]:
        _, icon = SEVERITY.get(a.severity, ("warning", "🟡"))
        state = "" if a.status == "open" else " alert-resolved"
        resolved = f" · resolved {a.resolved_at:%b %d %H:%M}" if a.resolved_at else ""
        st.markdown(f"<div class='alert-row alert-{a.severity}{state}'>{icon} <b>{a.severity.upper()}</b> · "
                    f"{a.message}<div class='meta'>{a.rule} · opened {a.created_at:%b %d %H:%M} UTC{resolved}</div>"
                    "</div>", unsafe_allow_html=True)

    st.markdown("#### Does the judge still catch bad answers?")
    if not judge.empty:
        bad = judge[(judge.truth_bad == True) & judge.ground_pass.notna()].copy()  # noqa: E712
        bad["day"] = bad.created_at.dt.floor("1D")
        # few known-bad answers per day, so use the alert rule's own 72h window, and hide thin points
        daily = (bad.groupby("day").agg(hit=("ground_pass", lambda s: int((~s.astype(bool)).sum())),
                                        n=("ground_pass", "size"))
                 .asfreq("1D", fill_value=0).rolling(3, min_periods=1).sum().reset_index())
        daily["caught"] = (daily.hit / daily.n).where(daily.n >= 5)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily.day, y=daily.caught, mode="lines+markers", name="caught (72h)",
                                 line=dict(width=2, color=S["v1"]), marker=dict(size=8),
                                 customdata=daily.n, hovertemplate="%{y:.0%} of %{customdata} bad answers"
                                 "<extra>judge caught</extra>"))
        if len(runs):
            fig.add_trace(go.Scatter(x=runs.created_at, y=runs.accuracy, mode="lines+markers", name="canary accuracy",
                                     line=dict(width=2, color=S["v2"]), marker=dict(size=8, symbol="diamond"),
                                     hovertemplate="%{y:.0%}<extra>canary accuracy</extra>"))
        for inc in incident_windows():
            if inc.get("layer") == "judge":
                fig.add_vrect(x0=inc["start"], x1=inc["end"], fillcolor=INK["band"], opacity=0.6, line_width=0,
                              annotation_text="judge model update", annotation_position="top left",
                              annotation_font=dict(size=10, color=INK["muted"]))
        base_layout(fig, height=300, y_pct=True)
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Blue: over the last 72h, the share of answers the deterministic checks know are wrong that the judge also "
                   "failed. Orange: daily canary accuracy. In the shaded window a judge model update made the judge "
                   "wave bad answers through. Layer 1's judge scores *improved*; only Layer 2 noticed.")

# ---- canaries -----------------------------------------------------------------
with canary_tab:
    st.markdown("A fixed set of replies whose correct verdict is known by construction: correct answers, wrong "
                "prices, prices from memory, invented chains, the wrong product, and a complied prompt injection. "
                "They're re-judged daily, so a judge change shows up even when production traffic has no bad answers.")
    if runs.empty:
        st.info("No canary runs yet. Run `python -m app.evals.canaries --run`.")
    else:
        last_id = runs.iloc[-1].run_id
        latest = canaries[canaries.run_id == last_id]
        by_kind = latest.groupby(["kind", "dimension"]).agg(checks=("correct", "size"),
                                                           correct=("correct", "sum")).reset_index()
        st.markdown(f"**Latest run** · {runs.iloc[-1].created_at:%b %d %H:%M} UTC · judge `{runs.iloc[-1].judge_model}` "
                    f"· {runs.iloc[-1].accuracy:.0%} correct")
        st.dataframe(by_kind, hide_index=True, use_container_width=True)
        wrong = latest[~latest.correct.astype(bool)]
        if len(wrong):
            st.markdown("**Missed checks**")
            st.dataframe(wrong[["canary_id", "kind", "dimension", "expected", "got", "reasoning"]],
                         hide_index=True, use_container_width=True)
        st.markdown("**History**")
        st.dataframe(runs.assign(accuracy=runs.accuracy.round(3)).sort_values("created_at", ascending=False),
                     hide_index=True, use_container_width=True)

# ---- bias & consistency ------------------------------------------------------------
with bias_tab:
    last7 = judge[judge.created_at >= now - timedelta(days=7)]
    rho, rho_n = JUDGE_METRICS["judge_length_bias"][1](last7)
    gap, gap_n = JUDGE_METRICS["judge_language_gap"][1](last7)
    tiles([
        ("Length bias (Spearman ρ)", "—" if rho is None else f"{rho:+.2f}", f"n = {rho_n:,} known-good answers", False),
        ("Helpfulness gap he − en", "—" if gap is None else f"{gap:+.2f}", "points on the 1–5 scale", False),
        ("Flip rate when re-judged", "—" if flip is None else f"{flip:.0%}", f"n = {flip_n}", False),
    ])
    st.caption("Judges tend to reward longer answers, and can favour one language. Both are measured on answers "
               "the deterministic checks consider correct, so a real quality difference doesn't masquerade as bias.")
    good = last7[(last7.truth_bad != True) & last7.help_raw.notna()]  # noqa: E712
    if len(good) > 20:
        sample = good.sample(min(len(good), 1500), random_state=1)
        jitter = pd.Series([random.Random(i).uniform(-0.18, 0.18) for i in range(len(sample))], index=sample.index)
        fig = go.Figure()
        for lang, color in (("he", S["v1"]), ("en", S["v2"])):
            d = sample[sample.language == lang]
            fig.add_trace(go.Scatter(x=d.answer_len, y=d.help_raw.astype(float) + jitter[d.index], mode="markers",
                                     name={"he": "Hebrew", "en": "English"}[lang],
                                     marker=dict(size=8, color=color, opacity=0.55, line=dict(width=1, color=INK["surface"])),
                                     hovertemplate="%{x} chars · helpfulness %{y:.0f}<extra></extra>"))
        base_layout(fig, height=300, title="Helpfulness vs. answer length")
        fig.update_xaxes(title="answer length (chars)", showgrid=True, gridcolor=INK["grid"])
        fig.update_yaxes(title="judge helpfulness (1–5)", dtick=1)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

# ---- human calibration ------------------------------------------------------------
with human_tab:
    st.caption("The gold standard: your owner judgments on the Review page, compared with the judge's verdicts on the "
               "same answers. Cohen's κ corrects for chance agreement; below 0.6 (n ≥ 30) a Layer 2 alert opens.")
    res = agreement()
    items = [("Owner-judged traces", str(res["n"]), "aim for 40+", False)]
    for key, name in (("helpful", "Answer correct ↔ judge 'answers'"), ("grounded", "Method sound ↔ judge 'method'")):
        r = res.get(key)
        val = "—" if not r or r["kappa"] is None else f"κ = {r['kappa']:.2f}"
        sub = "" if not r else (f"{r['agreement']:.0%} agree · judge too lenient {r['judge_too_lenient']}× · "
                                f"too strict {r['judge_too_strict']}× · n={r['n']}")
        items.append((name, val, sub, bool(r and r["kappa"] is not None and r["n"] >= 30 and r["kappa"] < 0.6)))
    tiles(items)
    st.page_link("pages/10_Review.py", label="Judge traces on the Review page", icon="🧑‍⚖️")
