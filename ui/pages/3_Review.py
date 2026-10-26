import pandas as pd
import streamlit as st
from sqlalchemy import select

from app.analyst.semantic import get_semantic
from app.db import EvalResult, GoldenItem, HumanJudgment, SessionLocal, Trace
from app.evals import golden
from app.evals.human import (
    FAILURE_TAGS,
    agreement,
    approve_into_golden,
    curate_golden,
    draft_from_trace,
    golden_for_trace,
    judgments_frame,
    override_rate,
)
from components import owner_unlock, role
from lib import eval_badges, setup, tiles
from render import render_chart, render_queries

setup("Review", "🧑‍⚖️")
st.markdown("## 🧑‍⚖️ Review traces")
st.caption("Pick an answer, then: ① read the question and answer, ② read the trace behind it, ③ judge it (what the "
           "agent got wrong, and what a good answer looks like), ④ approve it into the golden set. Owner and visitor "
           "judgments are kept apart; together they measure the **human override rate**. Pick source **golden** to "
           "review golden-set test answers.")
owner_unlock()
me = role()
if toast := st.session_state.pop("review_toast", None):
    st.toast(toast)
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = uuid.uuid4().hex[:12]

# ---- headline metrics ----------------------------------------------------------------
owner_df, visitor_df = judgments_frame("owner"), judgments_frame("visitor")
o_rate, o_n = override_rate(owner_df)
v_rate, v_n = override_rate(visitor_df)
agr = agreement(pd.concat([owner_df, visitor_df]).drop_duplicates("trace_id", keep="first")
                if len(owner_df) + len(visitor_df) else owner_df)


def pct(v):
    return "—" if v is None else f"{v:.0%}"


tiles([("Human override rate · owner", pct(o_rate), f"n = {o_n} reviewed", False),
       ("Human override rate · visitors", pct(v_rate), f"n = {v_n} reviewed", False),
       ("Evals catch what humans flag", pct(agr.get("eval_recall_on_human_flags")),
        f"{agr.get('both_flag', 0)} caught · {agr.get('human_only', 0)} missed", False),
       ("Eval-only flags", str(agr.get("auto_only", 0)), "automated fail, human said fine", False)])
with st.expander("Agreement matrix and per-evaluator recall"):
    if agr.get("n"):
        st.dataframe(pd.DataFrame({"automated: flagged": [agr["both_flag"], agr["auto_only"]],
                                   "automated: passed": [agr["human_only"], agr["neither"]]},
                                  index=["human: wrong/corrected", "human: fine"]), use_container_width=True)
        if agr["per_evaluator_recall"]:
            st.caption("Share of human-flagged traces each evaluator also failed:")
            st.dataframe(pd.Series(agr["per_evaluator_recall"], name="recall").to_frame(), use_container_width=True)
    else:
        st.caption("No judgments yet.")

# ---- trace list ---------------------------------------------------------------------
FAILING = ["any", "any check", "grounding", "method_lint", "canonical_check", "sql_health", "completion",
           "chart_lint", "decline_check", "assumption_check", "language_match", "judge_method", "judge_groundedness",
           "judge_helpfulness", "golden_match"]
TOPICS = ["any", "comparison", "lookup", "spread", "promo", "geo", "ambiguous", "unanswerable", "injection"]
st.divider()
f1, f2, f3 = st.columns(3)
source = f1.selectbox("Source", ["user", "golden", "any", "synthetic", "backfill", "offline"])
failing = f2.selectbox("Failing check", FAILING, help="'any check' = failed at least one automated check")
topic = f3.selectbox("Question type", TOPICS)
only_unreviewed = st.toggle("Unreviewed only", value=True)
with SessionLocal() as s:
    q = select(Trace).where(Trace.analysis.is_not(None) | Trace.error.is_not(None)).order_by(
        Trace.created_at.desc()).limit(300)
    if source != "any":
        q = q.where(Trace.source == source)
    if topic != "any":
        q = q.where(Trace.topic == topic)
    if failing != "any":
        fails = select(EvalResult.trace_id).where(EvalResult.passed.is_(False))
        q = q.where(Trace.id.in_(fails if failing == "any check" else fails.where(EvalResult.evaluator == failing)))
    traces = s.execute(q).scalars().all()
    reviewed = {tid for (tid,) in s.execute(select(HumanJudgment.trace_id).where(HumanJudgment.role == me))}
    rows = [{"time": t.created_at.strftime("%m-%d %H:%M"), "source": t.source, "type": t.topic,
             "question": t.question,
             "failed": ", ".join(e.evaluator for e in t.evals if e.passed is False) or "—", "id": t.id}
            for t in traces if not only_unreviewed or t.id not in reviewed]
if not rows:
    st.info("Nothing to review with these filters. Ask the analyst something on the Chat page first.")
    st.stop()
table = pd.DataFrame(rows)
st.caption(f"{len(table)} most recent matching traces")
pick = st.dataframe(table.drop(columns=["id"]), hide_index=True, use_container_width=True, height=280,
                    on_select="rerun", selection_mode="single-row")
idx = pick.selection.rows[0] if pick.selection.rows else 0
tid = table.iloc[idx]["id"]

with SessionLocal() as s:
    t = s.get(Trace, tid)
    tr = {"id": t.id, "question": t.question, "answer": t.answer, "analysis": t.analysis, "tool_calls": t.tool_calls,
          "error": t.error, "version": t.agent_version, "model": t.model}
    meta = (f"{t.created_at:%Y-%m-%d %H:%M} UTC · {t.source} · {t.agent_version} · {t.model} · {t.steps} steps · "
            f"{t.latency_ms / 1000:.1f}s · {t.input_tokens + t.output_tokens:,} tokens · ${t.cost_usd:.4f} · "
            f"trace `{t.id[:8]}`")
    evals = [{"evaluator": e.evaluator, "kind": e.kind, "passed": e.passed, "details": e.details} for e in t.evals]
    prior = s.execute(select(HumanJudgment).where(HumanJudgment.trace_id == tid)
                      .order_by(HumanJudgment.created_at.desc())).scalars().all()
    prior = [{"id": j.id, "role": j.role, "correct": j.correct, "method_ok": j.method_ok, "tags": j.tags,
              "corrected_answer": j.corrected_answer, "note": j.note, "promoted_to": j.promoted_to} for j in prior]
    gid, matched = golden_for_trace(tid, session=s)
    gitem = s.get(GoldenItem, gid) if gid else None
    gold = golden.as_dict(gitem) if gitem else None
    in_golden = s.execute(select(GoldenItem.id).where(GoldenItem.source_trace_id == tid)).scalar()

st.divider()

# ---- ① question & answer ---------------------------------------------------------------
st.markdown("### ① Question & answer")
st.markdown(f"#### {tr['question']}")
st.markdown(tr["answer"] or f"*(no answer: {tr['error']})*")
render_chart(tr, key="_review")

# ---- ② the trace -------------------------------------------------------------------------
st.markdown("### ② The trace")
st.caption(meta.replace("$", "\\$"))
q_col, v_col = st.columns([3, 2], gap="large")
with q_col:
    render_queries(tr, expanded=True)
with v_col:
    st.markdown("**Automated verdicts**")
    st.markdown(eval_badges(evals) or "—", unsafe_allow_html=True)
    for e in evals:
        if e["passed"] is False:
            with st.expander(f"✗ {e['evaluator']}", expanded=True):
                if e["details"].get("message"):
                    st.markdown(e["details"]["message"])
                st.json(e["details"], expanded=False)
    judge = next((e for e in evals if e["kind"] == "judge" and e["details"].get("reasoning")), None)
    if judge:
        st.caption(f"Judge ({judge['details'].get('judge_model')}): {judge['details']['reasoning']}")
if gold:
    with st.container(border=True):
        st.markdown(f"**🏅 Golden question {gold['id']}** · status **{gold['status']}** · {gold['qtype']} · "
                    f"answer type {gold['answer_type']} · scorer: "
                    + ("✓ matches the reference" if matched else "✗ doesn't match the reference"))
        if gold.get("notes"):
            st.caption(gold["notes"])
        if gold.get("reference_sql"):
            st.code(gold["reference_sql"], language="sql")
            ref = get_semantic().run(gold["reference_sql"], "ref")
            if ref.error:
                st.error(ref.error)
            else:
                st.caption(f"Today's reference result · {ref.row_count} rows")
                st.dataframe(pd.DataFrame(ref.rows[:10], columns=ref.columns), hide_index=True,
                             use_container_width=True)
        else:
            st.caption("No reference SQL: this question is scored on whether the agent declines or states an "
                       "assumption.")

# ---- ③ your judgment -------------------------------------------------------------------------
st.markdown(f"### ③ Your judgment · as **{me}**")
for j in prior:
    st.caption(f"Earlier, {j['role']}: correct = {j['correct']}" + (f" · {', '.join(j['tags'])}" if j["tags"] else "")
               + (f" · went wrong: {j['note']}" if j["note"] else "")
               + (f" · good answer: {j['corrected_answer']}" if j["corrected_answer"] else "")
               + (f" · in the golden set as {j['promoted_to']}" if j["promoted_to"] else ""))
correct = st.radio("Is the answer correct?", ["yes", "no", "unsure"], index=None, horizontal=True,
                   key=f"correct_{tid}")
wrong_col, good_col = st.columns(2, gap="large")
with wrong_col:
    st.markdown("**What did the agent get wrong?**")
    tags = st.multiselect("Problems", FAILURE_TAGS, key=f"tags_{tid}",
                          help="'bad method' also records that the method was unsound")
    wrong = st.text_area("Describe it", key=f"wrong_{tid}", height=100,
                         placeholder="e.g. compared averages over different products per chain")
with good_col:
    st.markdown("**What should a good answer have looked like?**")
    good = st.text_area("A good answer", key=f"good_{tid}", height=100,
                        placeholder="e.g. Rami Levy is cheapest in Haifa, about 8% below the city median")
    sql = st.text_area("Correct SQL (becomes the golden reference)", key=f"sql_{tid}", height=100) \
        if me == "owner" else ""


def save_judgment() -> int | None:
    if correct is None:
        st.warning("First say whether the answer is correct (step ③).")
        return None
    with SessionLocal() as s:
        j = HumanJudgment(trace_id=tid, role=me, reviewer=st.session_state.session_id, correct=correct,
                          method_ok=None if correct == "unsure" else "bad method" not in tags, tags=tags,
                          corrected_answer=good.strip() or None, corrected_sql=(sql or "").strip() or None,
                          note=wrong.strip() or None)
        s.add(j)
        s.commit()
        return j.id


# ---- ④ approve and add to the golden set ---------------------------------------------------------
st.markdown("### ④ Approve and add to the golden set")
if me == "owner":
    if gold:
        st.caption("This answer comes from a golden-set test. Approving saves your judgment and curates the question: "
                   "agreeing with the scorer approves it, correct SQL replaces its reference.")
        bad_question = st.checkbox("This golden question is bad: reject it", key=f"bad_{tid}")
    else:
        st.caption(("Already in the golden set as **" + in_golden + "**. " if in_golden else "")
                   + "Approving saves your judgment and puts this question in the golden set, approved when the "
                   "reference can be trusted: you gave correct SQL, judged the answer correct, or it's about declining "
                   "or stating an assumption. Otherwise it's added as a draft until you add the SQL.")
        bad_question = False
    approve, only = st.columns(2)
    if approve.button("✅ Approve and add to the golden set", type="primary", use_container_width=True,
                      key=f"approve_{tid}"):
        jid = save_judgment()
        if jid:
            msg = curate_golden(tid, correct, sql, bad_question) if gold else approve_into_golden(jid)
            st.session_state.review_toast = f"Judgment saved. {msg}."
            st.rerun()
else:
    st.caption("The owner approves what goes into the golden set. You can suggest this question for it: it's added as "
               "a draft, with your good answer attached.")
    approve, only = st.columns(2)
    if not gold and approve.button("🏅 Save and suggest for the golden set", type="primary",
                                   use_container_width=True, key=f"suggest_{tid}"):
        if save_judgment():
            new = draft_from_trace(tid, "trace", "Suggested on the Review page."
                                   + (f" A good answer: {good.strip()}" if good.strip() else ""))
            st.session_state.review_toast = f"Judgment saved, and {new} suggested for the golden set. Thanks!"
            st.rerun()
if only.button("Save judgment only", use_container_width=True, key=f"save_{tid}"):
    if save_judgment():
        st.session_state.review_toast = "Judgment saved. Thanks!"
        st.rerun()
