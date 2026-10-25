import random

import streamlit as st
from sqlalchemy import select

from app.db import EvalResult, HumanLabel, SessionLocal, Trace
from app.evals.calibration import agreement
from lib import api, setup

setup("Calibrate the judge", "🎯")
st.markdown("## 🎯 Calibrate the judge")
st.caption("An LLM judge is a model too, so it needs evaluating. Label a few judged traces yourself and see how "
           "often the judge agrees with you (Cohen's κ corrects for chance agreement).")

res = agreement()
c1, c2, c3 = st.columns(3)
c1.markdown(f"<div class='tile'><div class='label'>Labelled traces</div><div class='value'>{res['n']}</div>"
            "<div class='sub'>aim for 40+</div></div>", unsafe_allow_html=True)
for col, key, name in [(c2, "helpful", "Helpfulness"), (c3, "grounded", "Groundedness")]:
    r = res.get(key)
    val = "—" if not r or r["kappa"] is None else f"κ = {r['kappa']:.2f}"
    sub = "" if not r else (f"{r['agreement']:.0%} raw agreement · judge too lenient {r['judge_too_lenient']}×, "
                            f"too strict {r['judge_too_strict']}×")
    col.markdown(f"<div class='tile'><div class='label'>{name}</div><div class='value'>{val}</div>"
                 f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)
st.caption("Rough guide: κ > 0.6 means substantial agreement. Below that, fix the rubric before you trust the judge's trends.")

st.divider()
with SessionLocal() as s:
    labelled = {tid for (tid,) in s.execute(select(HumanLabel.trace_id)).all()}
    judged = [tid for (tid,) in s.execute(
        select(EvalResult.trace_id).where(EvalResult.evaluator == "judge_helpfulness").distinct().limit(3000)).all()]
pool = [t for t in judged if t not in labelled]
if not pool:
    st.success("Nothing left to label.")
    st.stop()
if "label_id" not in st.session_state or st.session_state.label_id not in pool:
    st.session_state.label_id = random.choice(pool)

with SessionLocal() as s:
    t = s.get(Trace, st.session_state.label_id)
    question, answer, tools = t.question, t.answer, t.tool_calls

st.markdown("#### Label this reply")
st.markdown(f"**Q:** {question}")
st.markdown(f"**A:** {answer}")
with st.expander("What the agent saw (tool results)"):
    st.json(tools, expanded=False)
st.caption("Judge it before you look at the judge, so you aren't anchored by its verdict.")
with st.form("label"):
    helpful = st.radio("Helpful? Does it answer what was asked, usefully?", ["yes", "no"], horizontal=True)
    grounded = st.radio("Grounded? Is every fact supported by the tool results?", ["yes", "no"], horizontal=True)
    note = st.text_input("Note (optional)")
    if st.form_submit_button("Save label", type="primary"):
        api("POST", "/labels", json={"trace_id": st.session_state.label_id, "helpful": helpful == "yes",
                                     "grounded": grounded == "yes", "note": note or None})
        del st.session_state["label_id"]
        st.rerun()
if st.button("Skip"):
    del st.session_state["label_id"]
    st.rerun()
