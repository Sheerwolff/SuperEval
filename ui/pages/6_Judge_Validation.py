import httpx
import pandas as pd
import streamlit as st

from app.config import settings
from app.evals.human import agreement, judge_feedback_summary, judgments_frame, missed_failures
from app.evals.judge_validation import gold_set, latest_calibration, production_matrix
from components import owner_unlock, role
from lib import api, setup, tiles, user_key

setup("Judge validation", "⚖️")
st.markdown("## ⚖️ Judge Validation")
st.caption("Compare judge verdicts against a human-annotated gold set.")

DEFS = {
    "recall": "Of the answers a human judged wrong, the share the judge also failed. The judge's main job.",
    "precision": "Of the answers the judge failed, the share a human also judged wrong. Low means false alarms.",
    "agreement": "Share of gold-set answers where the judge and the human reached the same verdict.",
    "kappa": "Agreement corrected for chance: 1 is perfect, 0 is no better than chance.",
    "gold": "Answers the owner judged correct or wrong on the Golden set page (latest judgment per answer).",
    "chat": "Share of chat users who said they agree with the eval verdicts on an answer.",
    "missed": "Answers people judged wrong on the Golden set page where no automated check (rules or judge) failed.",
    "suite_recall": "Of the answers people judged wrong, the share at least one automated rule check also failed.",
}


def pct(v):
    return "—" if v is None else f"{v:.0%}"


owner_unlock()
is_owner = role() == "owner"
gold = gold_set()
cal = latest_calibration()

# ---- confusion matrix ---------------------------------------------------------------------
st.markdown("#### Confusion matrix")
if cal:
    m = cal["summary"]["with"]
    st.caption(f"Latest calibration ({cal['created_at']:%b %d %H:%M} UTC): the judge with ground-truth evidence, as it "
               "now runs in Chat. Wrong answers are the positive class.")
else:
    m = production_matrix()
    st.caption("The live judge's stored verdicts on the gold set (only the answers it sampled). Calibrate to re-judge "
               "the whole set. Wrong answers are the positive class.")
tiles([
    ("Recall", pct(m["recall"]), f"{m['tp']} of {m['tp'] + m['fn']} wrong answers caught", False, DEFS["recall"]),
    ("Precision", pct(m["precision"]), f"{m['fp']} false alarms", False, DEFS["precision"]),
    ("Agreement", pct(m["agreement"]), f"n = {m['n']}", False, DEFS["agreement"]),
    ("Cohen's κ", "—" if m["kappa"] is None else f"{m['kappa']:.2f}", "chance-corrected", False, DEFS["kappa"]),
    ("Gold set", str(len(gold)), "owner-annotated answers", False, DEFS["gold"]),
])
st.dataframe(pd.DataFrame({"judge: failed": [m["tp"], m["fp"]], "judge: passed": [m["fn"], m["tn"]]},
                          index=["human: wrong", "human: correct"]), use_container_width=True)
if not m["n"]:
    st.info("No gold set to compare yet. Judge answers on the Golden set page as owner (correct / wrong), then calibrate.")

# ---- ablation ----------------------------------------------------------------------------
st.markdown("#### Ablation: without vs with ground-truth evidence")
st.caption("The same gold set judged twice: with the plain rubric, and with database records retrieved independently "
           "of the agent (a reviewed golden reference query, or the canonical chain price index).")
if cal:
    w, wo = cal["summary"]["with"], cal["summary"]["without"]

    def fmt(key, v):
        if v is None:
            return "—"
        return f"{v:.2f}" if key == "kappa" else (f"{v:.0%}" if key in ("recall", "precision", "agreement") else str(v))

    def diff(key):
        if w[key] is None or wo[key] is None:
            return "—"
        d = w[key] - wo[key]
        return f"{d:+.2f}" if key == "kappa" else (f"{d:+.0%}" if key in ("recall", "precision", "agreement")
                                                    else f"{d:+d}")

    keys = [("recall", "Recall"), ("precision", "Precision"), ("agreement", "Agreement"), ("kappa", "Cohen's κ"),
            ("tp", "Wrong answers caught"), ("fn", "Wrong answers missed"), ("fp", "False alarms"),
            ("tn", "Correct answers passed")]
    st.dataframe(pd.DataFrame({"without evidence": [fmt(k, wo[k]) for k, _ in keys],
                               "with evidence": [fmt(k, w[k]) for k, _ in keys],
                               "diff": [diff(k) for k, _ in keys]}, index=[label for _, label in keys]),
                 use_container_width=True)
    changed = [r for r in cal["rows"] if r.get("without") != r.get("with")]
    with st.expander(f"Verdicts that changed with evidence ({len(changed)})"):
        for r in changed:
            st.markdown(f"**{r['question']}**  \nhuman: {'wrong' if r['human_wrong'] else 'correct'} · "
                        f"without: {'failed' if r['without'] else 'passed'} → with: "
                        f"{'failed' if r['with'] else 'passed'} · evidence: {r.get('evidence') or '—'}")
            st.caption(r.get("with_reasoning") or "")
    st.caption(f"Judge {cal['judge_model']} · evidence set: {len(cal['evidence'])} approved golden questions · "
               f"cost \\${cal['cost_usd']:.2f}")
else:
    st.info("No calibration yet.")

status = api("GET", "/judge/calibrate/status") or {}
if is_owner:
    run, note = st.columns([1, 3], vertical_alignment="center")
    note.caption(f"Updates the evidence the judge is fed (today's approved golden questions), then re-judges the "
                 f"{len(gold)} gold-set answers without and with it, using the live judge ({settings.judge_model}). "
                 "Charged to your daily budget.")
    if run.button("🎯 Calibrate", type="primary", use_container_width=True,
                  disabled=bool(status.get("running")) or not gold):
        r = httpx.post(settings.api_url + "/judge/calibrate", json={"user_key": user_key()},
                       headers={"X-Admin-Key": settings.admin_key or ""}, timeout=30)
        if r.status_code >= 400:
            st.error(str(r.json().get("detail", r.text)).replace("$", "\\$"))
        else:
            st.session_state.cal_was_running = True
            st.rerun()
else:
    st.caption("Only the owner can calibrate the judge.")


@st.fragment(run_every=3 if status.get("running") else None)
def cal_progress() -> None:
    cur = api("GET", "/judge/calibrate/status") or {}
    if cur.get("running"):
        st.session_state.cal_was_running = True
        st.progress(cur["done"] / max(cur["total"], 1), text=f"Calibrating… {cur['done']}/{cur['total']} judge calls")
    elif st.session_state.pop("cal_was_running", False):
        st.rerun()
    if cur.get("last_error"):
        st.error(f"The last calibration failed: {cur['last_error']}")


cal_progress()

# ---- human feedback ----------------------------------------------------------------------
st.markdown("#### Human feedback")
fb = judge_feedback_summary()
agr = agreement(judgments_frame())
miss = missed_failures()
tiles([
    ("Chat agreement rate", pct(fb["agree_rate"]), f"{fb['n']} answers rated · {fb['n_disagree']} disagreed", False,
     DEFS["chat"]),
    ("Failures the suite missed", str(len(miss)), "judged wrong by people, no check failed", len(miss) > 0,
     DEFS["missed"]),
    ("Suite catches human-flagged failures", pct(agr.get("eval_recall_on_human_flags")),
     f"{agr.get('both_flag', 0)} caught · {agr.get('human_only', 0)} missed by the rule checks", False,
     DEFS["suite_recall"]),
])
if len(miss):
    st.markdown("**Failures the suite should have caught**")
    st.dataframe(miss.drop(columns=["trace_id"]), hide_index=True, use_container_width=True,
                 column_config={"when": st.column_config.DatetimeColumn(format="MMM D HH:mm"),
                                "question": st.column_config.TextColumn(width="large")})
if not fb["disputed"].empty:
    top = fb["disputed"].head(5)
    st.caption("Most disputed checks in Chat: " + " · ".join(
        f"{r.evaluator} {r.total}× (✓ {r.said_pass} · ✗ {r.said_fail})" for r in top.itertuples()))
