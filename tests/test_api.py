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


def test_health_and_deploy_roundtrip():
    with TestClient(main.app) as client:
        assert client.get("/health").json()["ok"]
        assert client.post("/deployment", json={"version": "a1"}).json()["current"] == "a1"
        assert client.post("/deployment", json={"version": "nope"}).status_code == 400
