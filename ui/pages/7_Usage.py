from datetime import timedelta

import plotly.graph_objects as go
import streamlit as st

from app.db import utcnow
from app.monitor.events import daily_activity, events_frame, page_counts, usage_summary, visitor_table
from components import owner_unlock, role
from lib import base_layout, series, setup, tiles

setup("Usage", "👥")
st.markdown("## 👥 Usage")
st.caption("Who uses this app and what they do. Visitors are counted by a hash of their IP address (no cookies, no raw "
           "IPs stored), so people sharing a network count as one and one person on two networks as two.")
owner_unlock()

DEFS = {
    "visitors": "Distinct visitors (IP hashes) who opened any page in the period.",
    "sessions": "Distinct browser sessions. A reload or a new tab starts a new session.",
    "views": "Pages opened, counted once per page per session.",
    "askers": "Visitors who asked the chat at least one question, and what share of all visitors that is.",
    "questions": "Chat questions asked, including the example buttons and follow-up chips.",
    "returning": "Visitors seen on more than one day in the period.",
}

c1, c2 = st.columns([1, 1])
days = c1.selectbox("Period", [7, 30, 90], index=1, format_func=lambda d: f"Last {d} days")
with_owner = c2.toggle("Include owner activity", value=False,
                       help="Visits and actions from sessions where the admin key was entered.")
since = utcnow() - timedelta(days=days)
df = events_frame(since, include_owner=with_owner)

if df.empty:
    st.info("No visits recorded in this period yet. Tracking started when this page was deployed, so earlier "
            "visits aren't included; chat questions before then are only in the traces.")
    st.stop()

s = usage_summary(df)
conv = "—" if s["conversion"] is None else f"{s['conversion']:.0%} of visitors"
tiles([
    ("Visitors", f"{s['visitors']:,}", f"{s['sessions']:,} sessions", False, DEFS["visitors"]),
    ("Page views", f"{s['page_views']:,}", "once per page per session", False, DEFS["views"]),
    ("Asked a question", f"{s['chat_visitors']:,}", conv, False, DEFS["askers"]),
    ("Questions", f"{s['questions']:,}", f"{s['feedback']:,} feedback votes", False, DEFS["questions"]),
    ("Returning", f"{s['returning']:,}", "seen on 2+ days", False, DEFS["returning"]),
])

# ---- daily activity ------------------------------------------------------------------
daily = daily_activity(df)
fig = go.Figure()
pal = series()
fig.add_trace(go.Bar(x=daily.day, y=daily.visitors, name="Visitors", marker_color=pal["v1"]))
fig.add_trace(go.Bar(x=daily.day, y=daily.questions, name="Questions", marker_color=pal["v2"]))
fig.update_layout(barmode="group")
base_layout(fig, height=260, title="Daily visitors and questions (UTC)")
fig.update_yaxes(dtick=1 if daily[["visitors", "questions"]].to_numpy().max() <= 5 else None)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---- funnel + pages -------------------------------------------------------------------
left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Funnel (distinct visitors)**")
    opened_chat = df[(df.event == "page_view") & (df.page == "Chat with the agent")].user_key.nunique()
    steps = [
        ("Opened any page", s["visitors"]),
        ("Opened the chat", opened_chat),
        ("Asked a question", s["chat_visitors"]),
        ("Gave feedback on a verdict", df[df.event == "chat_feedback"].user_key.nunique()),
    ]
    top = max(steps[0][1], 1)
    for label, n in steps:
        st.progress(min(n / top, 1.0), text=f"{label}: {n} ({n / top:.0%})")
with right:
    st.markdown("**Pages**")
    pages = page_counts(df)
    st.dataframe(pages.rename(columns={"page": "Page", "views": "Views", "visitors": "Visitors"}),
                 hide_index=True, use_container_width=True)

# ---- per-visitor (owner only) ----------------------------------------------------------
st.markdown("**Per visitor**")
if role() != "owner":
    st.caption("🔒 The per-visitor table is visible to the owner only.")
else:
    st.caption("Visitor = first 6 characters of the IP hash. Join to the traces with "
               "`events.props->>'trace_id' = traces.id` to read what each visitor asked.")
    st.dataframe(visitor_table(df), hide_index=True, use_container_width=True)
