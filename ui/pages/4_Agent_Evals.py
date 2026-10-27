from datetime import timedelta

import httpx
import pandas as pd
import streamlit as st
from sqlalchemy import func, select

from app.config import settings
from app.db import Alert, GoldenItem, SessionLocal, utcnow
from app.evals.bench import latest_run
from app.evals.human import judgments_frame, override_rate
from app.monitor.aggregates import metric_value
from lib import SEVERITY, api, fmt, frames, setup, tiles, track, user_key

setup("Agent evals", "📈")
st.markdown("## 📈 Agent Evals")

DEFS = {
    "accuracy": "Share of golden-set questions the agent answered correctly in the latest test. Each answer is "
                "compared with the result of the question's reference SQL on today's data.",
    "tested": "How many golden-set questions the latest test ran: approved and draft items (rejected ones are "
              "left out).",
    "run_cost": "What the latest test spent on live LLM calls, in total and per answered question.",
    "run_at": "When the latest test ran (UTC), and the model and prompt version it tested.",
    "helpfulness": "The LLM judge's 1–5 rating of whether an answer actually addresses the question, averaged over "
                   "the answers it judged.",
    "grounding": "Share of answers where every number appears in the result of the query it cites, with no uncited "
                 "numbers in the headline.",
    "sql_health": "Share of answers whose underlying SQL queries all ran without errors.",
    "method": "Share of answers whose SQL avoids known analysis pitfalls, such as comparing averages over different "
              "products per chain.",
    "declines": "Share of answers that decline exactly when they should: unanswerable or out-of-scope questions are "
                "declined, answerable ones aren't.",
    "latency": "95th-percentile time from question to finished answer: 95% of answers were faster than this.",
    "override": "Share of human-reviewed answers that a reviewer judged wrong, unsound, or corrected on the Review "
                "page.",
    "cost": "Average LLM spend (agent + judge) per answer that completed, over the last 24 hours.",
}

me = user_key()
traces, evals = frames(days=1)
now = utcnow()
run = latest_run()

# ---- golden set test -----------------------------------------------------------------
status = api("GET", "/bench/status", params={"user_key": me}) or {}
with SessionLocal() as s:
    mix = dict(s.execute(select(GoldenItem.source, func.count()).where(GoldenItem.status != "rejected")
                         .group_by(GoldenItem.source)).all())
n_items = sum(mix.values())
spent, budget = status.get("spent_today") or 0.0, status.get("budget", settings.user_daily_budget_usd)

with st.container(border=True):
    st.markdown("#### 🏅 Test the agent against the golden set")
    blocked = None
    if status and not status.get("live_llm"):
        blocked = "No API key is configured, so the live agent can't run."
    elif spent >= budget:
        blocked = f"You've used today's \\${budget:.0f} of live LLM calls. You can run it again tomorrow (UTC)."
    elif status.get("running"):
        blocked = "A test is already running."
    info, action = st.columns([3, 1], vertical_alignment="center")
    info.caption(f"Runs the live agent ({settings.agent_model}, prompt {settings.prompt_version}) on all {n_items} "
                 "approved and draft questions, and scores each answer against reference SQL computed from today's "
                 f"data. Takes about 4 minutes and ~\\$0.20. You've used \\${spent:.2f} of your \\${budget:.0f} "
                 "today.")
    if action.button("▶ Run the test", type="primary", disabled=bool(blocked) or not status,
                     use_container_width=True):
        try:
            r = httpx.post(settings.api_url + "/bench/run", json={"user_key": me}, timeout=30)
            if r.status_code >= 400:
                st.error(str(r.json().get("detail", r.text)).replace("$", "\\$"))
            else:
                st.session_state.bench_was_running = True
                track("bench_run", page="Agent evals")
                st.rerun()
        except httpx.HTTPError as exc:
            st.error(f"API error: {exc}")
    if blocked and not status.get("running"):
        st.caption(blocked)

    @st.fragment(run_every=3 if status.get("running") else None)
    def progress() -> None:
        cur = api("GET", "/bench/status", params={"user_key": me}) or {}
        if cur.get("running"):
            st.session_state.bench_was_running = True
            st.progress(cur["done"] / max(cur["total"], 1),
                        text=f"Testing… {cur['done']}/{cur['total']} answered · {cur['passed']} correct so far")
        elif st.session_state.pop("bench_was_running", False):
            st.rerun()  # the run just finished: redraw the whole page with its results
        if cur.get("last_error"):
            st.error(f"The last run failed: {cur['last_error']}")

    progress()

    if run is None:
        st.info("No test has run yet.")
    else:
        o, summ = run["summary"]["overall"], run["summary"]
        drafts = run.get("include_drafts", not run.get("reviewed_only", True))
        tiles([
            ("Accuracy", f"{o['rate']:.0%}", f"95% CI {o['ci'][0]:.0%}–{o['ci'][1]:.0%}", False, DEFS["accuracy"]),
            ("Questions tested", f"{o['n']}", "approved + drafts" if drafts else "approved only", False,
             DEFS["tested"]),
            ("Run cost", f"${summ['cost_usd']:.2f}",
             f"${summ['cost_per_completed'] or 0:.4f} per answer", False, DEFS["run_cost"]),
            ("Run at", run["at"][5:16].replace("T", " "),
             f"{run['model']}{'' if run.get('live') else ' (simulated)'} · prompt {run['version']}", False,
             DEFS["run_at"]),
        ])
        by_type = pd.DataFrame([{"question type": k, "questions": v["n"], "accuracy": v["rate"]}
                                for k, v in summ["qtype"].items()]).sort_values("accuracy")
        st.dataframe(by_type, hide_index=True, use_container_width=True,
                       column_config={"accuracy": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                                    max_value=1)})
        fails = [r for r in run["rows"] if not r["passed"]]
        with st.expander(f"Wrong answers ({len(fails)})", expanded=False):
            for r in fails:
                st.markdown(f"**{r['id']}** · {r['qtype']} · {r['question']}  \n"
                            f"→ {r.get('headline') or r.get('error') or 'no answer'}")
                st.caption(", ".join(f"{k}: {v}" for k, v in (r.get("details") or {}).items())[:300])
    labels = {"drafted": "generated drafts", "promoted": "owner judgments", "trace": "the Golden set page",
              "feedback": "chat feedback"}
    st.caption("Golden set: " + " · ".join(f"{n} from {labels.get(src, src)}" for src, n in
                                          sorted(mix.items(), key=lambda kv: -kv[1])))
    st.page_link("pages/3_Golden_Set.py", label="Review golden-set answers (source: golden)", icon="🏅")

# ---- layer 1 and alerts ------------------------------------------------------------------
with SessionLocal() as s:
    alerts = s.execute(select(Alert).order_by(Alert.created_at.desc()).limit(200)).scalars().all()
open_alerts = [a for a in alerts if a.status == "open"]


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
    day_t = traces[traces.created_at >= now - timedelta(days=1)]
    st.markdown("#### Success Criteria, last 3 hours")
    items = [("Accuracy", "—" if run is None else f"{run['summary']['overall']['rate']:.0%}",
              "latest golden-set test", False, DEFS["accuracy"])]
    for m, label, key in (("judge_helpfulness", "Helpfulness", "helpfulness"), ("grounding", "Grounding", "grounding"),
                          ("sql_health", "SQL health", "sql_health"), ("method_lint", "Sound method", "method"),
                          ("decline_check", "Correct declines", "declines"),
                          ("latency_p95_ms", "Latency p95", "latency")):
        v, n = metric_value(m, recent_e, recent_t)
        items.append((label, fmt(m, v), f"n = {n:,}", False, DEFS[key]))
    ovr, ovr_n = override_rate(judgments_frame())
    completed = day_t[day_t.error.isna()]
    cpt = (day_t.cost_usd.sum() / len(completed)) if len(completed) else None
    items += [("Human override rate", "—" if ovr is None else f"{ovr:.0%}", f"{ovr_n} reviewed answers", False,
               DEFS["override"]),
              ("Cost per completed task", "—" if cpt is None else f"${cpt:.4f}", "last 24h", False, DEFS["cost"])]
    tiles(items)
with right:
    st.markdown(f"#### Alerts · {len(open_alerts)} open")
    if not open_alerts:
        st.markdown("<div class='alert-row'>✅ No open alerts</div>", unsafe_allow_html=True)
    for a in open_alerts:
        st.markdown(alert_html(a), unsafe_allow_html=True)
    with st.expander(f"History ({len(alerts) - len(open_alerts)} resolved)"):
        for a in [a for a in alerts if a.status != "open"][:40]:
            st.markdown(alert_html(a), unsafe_allow_html=True)
