import json
import re
import uuid

import httpx
import streamlit as st

from app.config import settings
from lib import eval_badges, setup
from render import render_chart, render_queries

setup("Ask the analyst", "💬")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
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
    payload = {"question": question, "session_id": st.session_state.session_id, "history": history}
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
    st.caption(f"{msg['model']} · {msg['steps']} steps · {msg['latency_ms'] / 1000:.1f}s · "
               f"{msg['input_tokens'] + msg['output_tokens']:,} tokens · ${msg['cost_usd']:.4f} · "
               f"trace `{msg['id'][:8]}`")
    if msg.get("mode") == "sim":
        reason = {"no_api_key": "no API key configured", "rate_limited": "you hit the hourly rate limit",
                  "daily_budget_exhausted": "today's LLM budget is used up"}.get(msg.get("mode_reason"), "")
        st.info(f"Served by the simulated analyst ({reason}). Evals run the same way.")


# ---- page --------------------------------------------------------------------
st.markdown("## 💬 Ask the analyst")
st.caption("Ask an analytical question about grocery prices in Israel, in Hebrew or English. The analyst queries "
           "today's published prices from 8 chains in 25 cities, shows its queries and method, and every answer "
           "is evaluated on the spot.")

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
        st.caption("Send a question to see how its answer is evaluated: every number is re-checked against the query "
                   "it cites, the SQL method is linted, and an LLM judge reviews the analysis.")
    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
