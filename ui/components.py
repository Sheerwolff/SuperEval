"""Reusable UI pieces: the "Let's connect" card and the guided tour."""
from __future__ import annotations

import re
from urllib.parse import quote

import streamlit as st

from app.config import settings

ICONS = {
    "linkedin": '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M20.45 20.45h-3.56v-5.57'
                'c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 '
                '3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13zM7.12 '
                '20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 '
                '1.78-1.73V1.73C24 .77 23.2 0 22.22 0z"/></svg>',
    "email": '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 '
             '2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2zm0 4.24-8 5-8-5V6l8 5 8-5v2.24z"/></svg>',
    "whatsapp": '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M17.47 14.38c-.3-.15-1.76-.87-2.03'
                '-.97-.27-.1-.47-.15-.67.15-.2.3-.77.97-.94 1.17-.17.2-.35.22-.64.07-.3-.15-1.26-.46-2.4-1.48-.89-.79-1.49'
                '-1.77-1.66-2.07-.17-.3-.02-.46.13-.6.13-.14.3-.35.45-.52.15-.17.2-.3.3-.5.1-.2.05-.37-.03-.52-.07-.15-.67'
                '-1.61-.92-2.2-.24-.58-.49-.5-.67-.51h-.57c-.2 0-.52.07-.79.37-.27.3-1.04 1.02-1.04 2.48s1.07 2.88 1.21 '
                '3.07c.15.2 2.1 3.2 5.08 4.49.71.31 1.26.49 1.69.63.71.22 1.36.19 1.87.12.57-.09 1.76-.72 2.01-1.41.25-.7'
                '.25-1.29.17-1.41-.07-.12-.27-.2-.57-.35zM12.05 21.8h-.01a9.8 9.8 0 0 1-4.99-1.37l-.36-.21-3.71.97.99'
                '-3.62-.23-.37a9.8 9.8 0 0 1-1.5-5.21c0-5.42 4.41-9.83 9.84-9.83 2.62 0 5.09 1.03 6.94 2.88a9.77 9.77 0 0 '
                '1 2.88 6.95c0 5.42-4.41 9.83-9.83 9.83zm8.37-18.2A11.76 11.76 0 0 0 12.05.13C5.5.13.17 5.46.17 '
                '12.01c0 2.09.55 4.14 1.59 5.94L.07 24l6.2-1.63a11.84 11.84 0 0 0 5.67 1.45h.01c6.55 0 11.88-5.33 '
                '11.88-11.88 0-3.17-1.24-6.16-3.48-8.4z"/></svg>',
}


def _whatsapp_url() -> str | None:
    digits = re.sub(r"\D", "", settings.contact_whatsapp or "")
    if not digits:
        return None
    text = quote("Hi Sheer, I saw PriceWatch Evals and would love to talk.")
    return f"https://wa.me/{digits}?text={text}"


def _links() -> list[tuple[str, str, str]]:
    out = [("linkedin", "LinkedIn", settings.contact_linkedin),
           ("email", "Email", f"mailto:{settings.contact_email}?subject={quote('PriceWatch Evals')}")]
    wa = _whatsapp_url()
    if wa:
        out.append(("whatsapp", "WhatsApp", wa))
    return out


def connect_card(headline: str = "Let's connect", body: str | None = None) -> None:
    body = body or ("I build and monitor LLM systems in production: evals, judges, and the data underneath. "
                    "If this is the kind of work your team needs, I'd love to talk.")
    buttons = "".join(
        f"<a class='connect-btn connect-{key}' href='{url}' target='_blank' rel='noopener' "
        f"aria-label='{label}' title='{label}'>{ICONS[key]}<span>{label}</span></a>"
        for key, label, url in _links()
    )
    st.markdown(
        f"<div class='connect-card'><div class='connect-text'><div class='connect-kicker'>{headline}</div>"
        f"<div class='connect-name'>{settings.contact_name}</div><p>{body}</p></div>"
        f"<div class='connect-actions'>{buttons}</div></div>",
        unsafe_allow_html=True,
    )


def connect_compact() -> None:
    icons = "".join(
        f"<a class='connect-icon connect-{key}' href='{url}' target='_blank' rel='noopener' "
        f"aria-label='{label}' title='{label}'>{ICONS[key]}</a>"
        for key, label, url in _links()
    )
    st.markdown(f"<div class='connect-compact'><span>Built by {settings.contact_name}</span>"
                f"<div>{icons}</div></div>", unsafe_allow_html=True)


# ---- guided tour --------------------------------------------------------------
TOUR = [
    {"key": "intro", "page": "Home.py", "title": "Welcome",
     "hint": "A live LLM agent, its evals, and the monitors that watch both. About three minutes."},
    {"key": "chat", "page": "pages/1_Chat.py", "title": "Ask the agent",
     "hint": "Ask about any grocery item in Hebrew or English, or tap an example. Watch every answer get "
             "scored: exact price checks against the database, plus an LLM judge."},
    {"key": "break", "page": "pages/3_Live_Monitor.py", "title": "Break production",
     "hint": "Deploy v2, a plausible 'faster & cheaper' prompt, then send 40 requests. Critical alerts "
             "should open within seconds. Roll back when you've seen them."},
    {"key": "investigate", "page": "pages/4_Trace_Explorer.py", "title": "Investigate",
     "hint": "Filtered to answers that failed price accuracy. Open one to see the quoted number that "
             "isn't in any tool result, and the judge's reasoning."},
    {"key": "judge", "page": "pages/6_Judge_Monitor.py", "title": "Who watches the judge?",
     "hint": "Layer 2. Nine days ago a judge model update made the judge lenient, so Layer 1 looked "
             "*better*. Canaries and ground-truth recall caught it."},
    {"key": "wrap", "page": "Home.py", "title": "That's the tour", "hint": ""},
]


def tour_active() -> bool:
    return st.session_state.get("tour_step") is not None


def tour_start() -> None:
    st.session_state.tour_step = 1
    st.session_state.tour_done = set()
    st.switch_page(TOUR[1]["page"])


def tour_mark(key: str) -> None:
    """Record that the visitor did a step's action (sent a chat, deployed v2, ...)."""
    if tour_active():
        st.session_state.setdefault("tour_done", set()).add(key)


def tour_banner(key: str) -> None:
    """Shown at the top of each page while the tour is running."""
    if not tour_active():
        return
    idx = next((i for i, s in enumerate(TOUR) if s["key"] == key), None)
    if idx is None:
        return
    st.session_state.tour_step = idx
    step = TOUR[idx]
    done = key in st.session_state.get("tour_done", set())
    total = len(TOUR) - 1
    with st.container(border=True):
        a, b, c = st.columns([6, 1.3, 1.1], vertical_alignment="center")
        a.markdown(f"<div class='tour-kicker'>Tour · step {idx} of {total - 1}"
                   f"{' · ✓ done' if done else ''}</div><div class='tour-title'>{step['title']}</div>"
                   f"<div class='tour-hint'>{step['hint']}</div>", unsafe_allow_html=True)
        nxt = TOUR[idx + 1]
        if b.button("Next →" if idx + 1 < total else "Finish →", type="primary" if done else "secondary",
                    use_container_width=True, key=f"tour_next_{key}"):
            st.session_state.tour_step = idx + 1
            if nxt["key"] == "investigate":
                st.session_state["trace_filter_failing"] = "price_accuracy"
            st.switch_page(nxt["page"])
        if c.button("Exit tour", use_container_width=True, key=f"tour_exit_{key}"):
            st.session_state.tour_step = None
            st.rerun()
