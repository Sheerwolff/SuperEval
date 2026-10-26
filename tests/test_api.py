"""API smoke tests with the simulated analyst (no key, no network)."""
import json

from fastapi.testclient import TestClient

from app.api import main
from app.config import settings


def test_chat_stream_emits_query_steps_then_done(monkeypatch):
    monkeypatch.setattr(type(settings), "live_llm", property(lambda self: False))
    with TestClient(main.app) as client:
        with client.stream("POST", "/chat/stream", json={"question": "Which chain is cheapest in Haifa?"}) as r:
            events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "tool_start" and "tool_end" in kinds and kinds[-1] == "done"
    done = events[-1]
    assert done["mode"] == "sim" and done["analysis"]["claims"] and done["tool_calls"][0]["sql"]
    assert any("rows" in (e.get("summary") or "") for e in events if e["type"] == "tool_end")


def test_health_and_fault_injection(monkeypatch):
    monkeypatch.setattr(type(settings), "live_llm", property(lambda self: False))
    with TestClient(main.app) as client:
        assert client.get("/health").json()["ok"]
        r = client.post("/simulate", json={"n": 5, "fault": "degraded_method"}).json()
        assert r["generated"] == 5 and r["fault"] == "degraded_method"
        assert client.post("/simulate", json={"n": 5, "fault": "nope"}).status_code == 422


def test_bench_run_refusals(monkeypatch):
    from app.api.cost_guard import guard
    from app.db import SessionLocal, UserSpend

    with TestClient(main.app) as client:
        monkeypatch.setattr(type(settings), "live_llm", property(lambda self: False))
        assert client.post("/bench/run", json={"user_key": "test-nokey"}).status_code == 400
        monkeypatch.setattr(type(settings), "live_llm", property(lambda self: True))
        guard.charge("test-broke", 5.0)
        try:
            r = client.post("/bench/run", json={"user_key": "test-broke"})
            assert r.status_code == 429 and "$" in r.json()["detail"]
            st = client.get("/bench/status", params={"user_key": "test-broke"}).json()
            assert st["spent_today"] >= 5.0 and not st["running"]
        finally:
            with SessionLocal() as s:
                s.query(UserSpend).filter(UserSpend.user_key == "test-broke").delete()
                s.commit()
