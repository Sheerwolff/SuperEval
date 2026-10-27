"""Per-visitor event tracking: log what visitors do, summarise it for the Usage page.

    log_event("chat_question", user_key=..., session_id=..., page="Chat", props={"trace_id": ...})

Logging never raises: analytics must not break the app (a failed insert is swallowed).
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import select

from app.db import Event, SessionLocal

log = logging.getLogger(__name__)

EVENTS = ("page_view", "chat_question", "chat_feedback", "bench_run", "calibrate_run")


def log_event(event: str, *, user_key: str | None, session_id: str | None, page: str | None = None,
              role: str = "visitor", props: dict | None = None) -> bool:
    """Record one event. Returns False (and logs a warning) if the insert failed."""
    try:
        with SessionLocal() as s:
            s.add(Event(event=event, user_key=user_key, session_id=session_id, page=page, role=role,
                        props=props or {}))
            s.commit()
        return True
    except Exception as exc:  # noqa: BLE001 - analytics must never take the app down
        log.warning("could not log event %s: %s", event, exc)
        return False


def events_frame(since: datetime | None = None, include_owner: bool = False) -> pd.DataFrame:
    """Events as a DataFrame (created_at, event, user_key, session_id, page, role, props)."""
    q = select(Event.created_at, Event.event, Event.user_key, Event.session_id, Event.page, Event.role, Event.props)
    if since is not None:
        q = q.where(Event.created_at >= since)
    if not include_owner:
        q = q.where(Event.role != "owner")
    with SessionLocal() as s:
        rows = s.execute(q.order_by(Event.created_at)).all()
    return pd.DataFrame(rows, columns=["created_at", "event", "user_key", "session_id", "page", "role", "props"])


def usage_summary(df: pd.DataFrame) -> dict:
    """Headline numbers from an events frame.

    visitors = distinct user_key (an IP hash, so people behind one network count once and one person on two
    networks counts twice); sessions = distinct browser sessions; returning = visitors active on 2+ days."""
    if df.empty:
        return {"visitors": 0, "sessions": 0, "page_views": 0, "chat_visitors": 0, "questions": 0,
                "returning": 0, "conversion": None, "feedback": 0}
    views = df[df.event == "page_view"]
    asked = df[df.event == "chat_question"]
    visitors = df.user_key.dropna().nunique()
    days = df.assign(day=pd.to_datetime(df.created_at).dt.date).groupby("user_key").day.nunique()
    chat_visitors = asked.user_key.dropna().nunique()
    return {
        "visitors": int(visitors),
        "sessions": int(df.session_id.dropna().nunique()),
        "page_views": int(len(views)),
        "chat_visitors": int(chat_visitors),
        "questions": int(len(asked)),
        "returning": int((days > 1).sum()),
        "conversion": (chat_visitors / visitors) if visitors else None,
        "feedback": int((df.event == "chat_feedback").sum()),
    }


def daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day: visitors, sessions, questions."""
    if df.empty:
        return pd.DataFrame(columns=["day", "visitors", "sessions", "questions"])
    d = df.assign(day=pd.to_datetime(df.created_at).dt.normalize())
    out = d.groupby("day").agg(visitors=("user_key", "nunique"), sessions=("session_id", "nunique"))
    out["questions"] = d[d.event == "chat_question"].groupby("day").size()
    return out.fillna(0).astype(int).reset_index()


def page_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Views and distinct visitors per page, most viewed first."""
    v = df[df.event == "page_view"]
    if v.empty:
        return pd.DataFrame(columns=["page", "views", "visitors"])
    return (v.groupby("page").agg(views=("event", "size"), visitors=("user_key", "nunique"))
            .sort_values(["views", "visitors"], ascending=False).reset_index())


def visitor_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per visitor: first/last seen, active days, sessions, pages viewed, questions asked."""
    if df.empty:
        return pd.DataFrame(columns=["visitor", "first_seen", "last_seen", "days", "sessions", "pages", "questions"])
    d = df.dropna(subset=["user_key"]).assign(day=lambda x: pd.to_datetime(x.created_at).dt.date)
    g = d.groupby("user_key")
    out = pd.DataFrame({
        "first_seen": g.created_at.min(), "last_seen": g.created_at.max(), "days": g.day.nunique(),
        "sessions": g.session_id.nunique(),
        "pages": d[d.event == "page_view"].groupby("user_key").page.nunique(),
        "questions": d[d.event == "chat_question"].groupby("user_key").size(),
    }).fillna({"pages": 0, "questions": 0}).astype({"pages": int, "questions": int})
    out.insert(0, "visitor", [k[:6] for k in out.index])  # short, non-reversible label
    return out.sort_values("last_seen", ascending=False).reset_index(drop=True)
