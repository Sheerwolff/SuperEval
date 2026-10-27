import json
import re

import httpx
import streamlit as st

from app.config import settings
from app.evals.human import save_judge_feedback
from components import role
from lib import eval_badges, session_id, setup, track, user_key
from render import render_chart, render_queries

setup("Chat with the agent", "💬")

if "messages" not in st.session_state:
    st.session_state.messages = []

EXAMPLES = [
    "Which chain is really cheapest in Haifa?",
    "האם רמי לוי באמת זול יותר משופרסל בתל אביב?",
    "Which products have the biggest price gap between chains in Jerusalem?",
    "How did milk prices change over the last year?",
]


def is_hebrew(text: str) -> bool:
    return len(re.findall(r"[֐-׿]", text)) > len(re.findall(r"[A-Za-z]", text))


def bubble(text: str) -> None:
    css = "rtl" if is_hebrew(text) else ""
    st.markdown(f"<div class='{css}'>\n\n{text}\n\n</div>", unsafe_allow_html=True)


def short_sql(sql: str, n: int = 70) -> str:
    one = " ".join(sql.split())
    return one if len(one) <= n else one[:n] + "…"


def stream_chat(question: str, history: list[dict], status) -> dict | None:
    payload = {"question": question, "session_id": session_id(), "user_key": user_key(),
               "history": history}
    steps = 0
    try:
        with httpx.stream("POST", settings.api_url + "/chat/stream", json=payload, timeout=150) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev["type"] == "tool_start":
                    status.update(label=f"🧮 Running {ev.get('id', 'query')}…")
                elif ev["type"] == "tool_end":
                    steps += 1
                    status.markdown(f"🧮 **{ev.get('id')}** `{short_sql(ev['args']['sql'])}` → {ev.get('summary', '')}")
                elif ev["type"] == "answer":
                    status.update(label="✍️ Writing the answer…")
                elif ev["type"] == "done":
                    status.update(label=f"Done · {steps} quer{'ies' if steps != 1 else 'y'} · "
                                        f"{ev['latency_ms'] / 1000:.1f}s", state="complete", expanded=False)
                    return ev
                elif ev["type"] == "error":
                    status.update(label="Something went wrong", state="error")
                    st.error(ev["message"])
                    return None
    except httpx.HTTPError as exc:
        status.update(label="API unreachable", state="error")
        st.error(f"API error: {exc}")
    return None


def follow_ups(trace: dict) -> list[str]:
    he = trace.get("language") == "he"
    if he:
        return ["ואיך זה בירושלים?", "באילו קטגוריות הפער הכי גדול?"]
    return ["And in Jerusalem?", "Which categories show the biggest gap?"]


def eval_label(e: dict) -> str:
    name = e["evaluator"].replace("judge_", "judge · ").replace("_", " ")
    return f"{'✓' if e['passed'] else '✗'} {name}"


def judge_feedback(msg: dict, verdicts: list[dict]) -> None:
    """Agree / disagree with the verdicts above; disagreeing asks which checks look wrong."""
    key = f"fb_{msg['id']}"
    if st.session_state.get(key) == "saved":
        st.caption("Thanks, your feedback on these verdicts was saved.")
        return
    st.markdown("**Do you agree with these verdicts?**")
    a, b = st.columns(2)
    if a.button("👍 Yes", key=f"{key}_yes", use_container_width=True):
        save_judge_feedback(msg["id"], role(), session_id(), agree=True)
        track("chat_feedback", page="Chat", agree=True, trace_id=msg["id"])
        st.session_state[key] = "saved"
        st.rerun()
    if b.button("👎 No", key=f"{key}_no", use_container_width=True):
        st.session_state[key] = "disagree"
    if st.session_state.get(key) == "disagree":
        labels = {e["evaluator"]: eval_label(e) for e in verdicts}
        with st.form(f"{key}_form"):
            disputed = st.multiselect("Which verdicts look wrong?", list(labels), format_func=labels.get)
            note = st.text_input("Why? (optional)")
            if st.form_submit_button("Send feedback", type="primary"):
                save_judge_feedback(msg["id"], role(), session_id(), agree=False,
                                    disputed=disputed, note=note)
                track("chat_feedback", page="Chat", agree=False, trace_id=msg["id"], disputed=len(disputed))
                st.session_state[key] = "saved"
                st.rerun()


def render_card(msg: dict) -> None:
    evals = msg["evals"]
    det = [e for e in evals if e["kind"] == "deterministic"]
    judge = [e for e in evals if e["kind"] == "judge"]
    failed = [e for e in evals if e["passed"] is False]
    status = ("✅ all checks passed" if not failed else f"⚠️ {len(failed)} check(s) failed") if evals else \
        "evaluators run here"
    st.markdown(f"**Eval scorecard** · {status}")
    if det:
        st.markdown(eval_badges(det), unsafe_allow_html=True)
    if judge:
        st.markdown(eval_badges(judge), unsafe_allow_html=True)
        reasoning = judge[0]["details"].get("reasoning")
        if reasoning:
            st.caption(f"Judge ({judge[0]['details'].get('judge_model')}): {reasoning}")
    for e in failed:
        msg_ = e["details"].get("message")
        if msg_:
            st.warning(msg_)
    verdicts = [e for e in evals if e["passed"] is not None]
    if verdicts:
        judge_feedback(msg, verdicts)
    st.caption(f"{msg['model']} · {msg['steps']} steps · {msg['latency_ms'] / 1000:.1f}s · "
               f"{msg['input_tokens'] + msg['output_tokens']:,} tokens · ${msg['cost_usd']:.4f} · "
               f"trace `{msg['id'][:8]}`")
    if msg.get("mode") == "sim":
        reason = {"no_api_key": "no API key configured",
                  "user_budget_exhausted": f"you've used today's ${settings.user_daily_budget_usd:.0f} of live "
                                           "LLM calls"}.get(msg.get("mode_reason"), "")
        st.info(f"Served by the simulated analyst ({reason}). Evals run the same way.")


# ---- page --------------------------------------------------------------------
st.markdown("## 💬 Chat with the agent")
st.caption("Ask about Israeli grocery prices. Every answer is evaluated live.")

clicked = st.session_state.pop("queued_prompt", None)
if not st.session_state.messages:
    for col, ex in zip(st.columns(2), EXAMPLES[:2], strict=True):
        if col.button(ex, use_container_width=True):
            clicked = ex
    for col, ex in zip(st.columns(2), EXAMPLES[2:], strict=True):
        if col.button(ex, use_container_width=True):
            clicked = ex

chat_col, card_col = st.columns([3, 2], gap="large")

with chat_col:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            bubble(m["content"])
            if m.get("trace"):
                render_chart(m["trace"])
                render_queries(m["trace"])

    prompt = st.chat_input("Ask an analytical question…") or clicked
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            bubble(prompt)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.chat_message("assistant"):
            status = st.status("Thinking…", expanded=True)
            res = stream_chat(prompt, history, status)
            track("chat_question", page="Chat", trace_id=res["id"] if res else None, answered=bool(res),
                  lang="he" if is_hebrew(prompt) else "en", suggested=bool(clicked), turn=len(history) // 2 + 1)
            if res:
                bubble(res["answer"])
                render_chart(res)
                render_queries(res)
                st.session_state.messages.append({"role": "assistant", "content": res["answer"], "trace": res})

    last = next((m for m in reversed(st.session_state.messages) if m.get("trace")), None)
    if last:
        chips = follow_ups(last["trace"])
        for i, (col, chip) in enumerate(zip(st.columns(len(chips)), chips, strict=True)):
            if col.button(chip, key=f"chip_{len(st.session_state.messages)}_{i}", use_container_width=True):
                st.session_state.queued_prompt = chip
                st.rerun()

with card_col:
    if last:
        render_card(last["trace"])
    else:
        st.markdown("**Eval scorecard**")
        st.caption("Send a question to see its answer evaluated.")
    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
