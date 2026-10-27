"""Per-visitor event tracking: logging, summaries, and the UI wiring."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from app.db import Event, SessionLocal, init_db
from app.monitor import events
from app.monitor.events import daily_activity, events_frame, log_event, page_counts, usage_summary, visitor_table

UI = Path(__file__).resolve().parent.parent / "ui"


@pytest.fixture(autouse=True)
def _clean_events():
    init_db()
    with SessionLocal() as s:
        s.query(Event).delete()
        s.commit()
    yield


def _frame(rows):
    return pd.DataFrame(rows, columns=["created_at", "event", "user_key", "session_id", "page", "role", "props"])


def test_log_event_stores_a_row():
    assert log_event("chat_question", user_key="u1", session_id="s1", page="Chat", props={"trace_id": "t1"})
    with SessionLocal() as s:
        e = s.execute(select(Event)).scalar_one()
    assert (e.event, e.user_key, e.session_id, e.role, e.props) == ("chat_question", "u1", "s1", "visitor",
                                                                      {"trace_id": "t1"})


def test_log_event_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("database down")

    monkeypatch.setattr(events, "SessionLocal", boom)
    assert log_event("page_view", user_key="u1", session_id="s1") is False


def test_owner_events_are_excluded_by_default():
    log_event("page_view", user_key="me", session_id="s0", role="owner")
    log_event("page_view", user_key="u1", session_id="s1")
    assert set(events_frame().user_key) == {"u1"}
    assert set(events_frame(include_owner=True).user_key) == {"me", "u1"}


def test_usage_summary_counts_visitors_sessions_and_returning():
    d1, d2 = datetime(2026, 10, 26, 10), datetime(2026, 10, 27, 11)
    df = _frame([
        (d1, "page_view", "a", "s1", "Home", "visitor", {}),
        (d1, "page_view", "a", "s1", "Chat with the agent", "visitor", {}),
        (d1, "chat_question", "a", "s1", "Chat", "visitor", {}),
        (d2, "page_view", "a", "s2", "Home", "visitor", {}),  # a returns the next day
        (d2, "page_view", "b", "s3", "Home", "visitor", {}),  # b only browses
        (d2, "chat_feedback", "a", "s2", "Chat", "visitor", {}),
    ])
    s = usage_summary(df)
    assert (s["visitors"], s["sessions"], s["page_views"]) == (2, 3, 4)
    assert (s["chat_visitors"], s["questions"], s["returning"], s["feedback"]) == (1, 1, 1, 1)
    assert s["conversion"] == 0.5
    daily = daily_activity(df)
    assert daily.to_dict("records") == [
        {"day": pd.Timestamp("2026-10-26"), "visitors": 1, "sessions": 1, "questions": 1},
        {"day": pd.Timestamp("2026-10-27"), "visitors": 2, "sessions": 2, "questions": 0},
    ]
    assert page_counts(df).iloc[0].to_dict() == {"page": "Home", "views": 3, "visitors": 2}
    vt = visitor_table(df).set_index("visitor")
    assert vt.loc["a", "days"] == 2 and vt.loc["a", "questions"] == 1 and vt.loc["b", "questions"] == 0


def test_empty_frames_are_safe():
    empty = events_frame()
    assert usage_summary(empty)["visitors"] == 0 and usage_summary(empty)["conversion"] is None
    assert daily_activity(empty).empty and page_counts(empty).empty and visitor_table(empty).empty


def _app_test(script: str):
    from streamlit.testing.v1 import AppTest

    for p in (str(UI), str(UI.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return AppTest.from_file(str(UI / script), default_timeout=30)


def test_opening_a_page_logs_one_page_view_per_session():
    at = _app_test("pages/7_Usage.py").run()
    assert not at.exception
    at.run()  # a rerun in the same session must not log it again
    with SessionLocal() as s:
        rows = s.execute(select(Event)).scalars().all()
    assert [(r.event, r.page) for r in rows] == [("page_view", "Usage")]
    assert rows[0].user_key and rows[0].session_id


def test_usage_page_renders_with_data():
    log_event("page_view", user_key="a" * 32, session_id="s1", page="Chat with the agent")
    log_event("chat_question", user_key="a" * 32, session_id="s1", page="Chat")
    at = _app_test("pages/7_Usage.py").run()
    assert not at.exception
    assert "Asked a question" in " ".join(m.value for m in at.markdown)


def test_chat_page_still_starts_and_shares_one_session_id():
    at = _app_test("pages/1_Chat.py").run()
    assert not at.exception
    assert at.session_state["messages"] == [] and len(at.session_state["session_id"]) == 12
