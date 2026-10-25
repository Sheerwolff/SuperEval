"""PriceWatch API: serves the analyst and exposes the monitoring data.

    uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import json
import queue
import random
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.analyst.pipeline import handle_analysis
from app.api.cost_guard import guard
from app.config import settings
from app.data.catalog import get_catalog
from app.db import Alert, SessionLocal, Trace, init_db
from app.jobs import sync_ingest_meta
from app.monitor.alerts import run_monitors

_last_monitor_run = 0.0
_monitor_lock = threading.Lock()


def maybe_run_monitors(min_interval: float = 20.0) -> None:
    """Run the monitors after traffic, at most every `min_interval` seconds, off the request path."""
    global _last_monitor_run
    if time.time() - _last_monitor_run < min_interval or not _monitor_lock.acquire(blocking=False):
        return

    def _run():
        global _last_monitor_run
        try:
            run_monitors()
        finally:
            _last_monitor_run = time.time()
            _monitor_lock.release()

    threading.Thread(target=_run, daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    sync_ingest_meta()
    yield


app = FastAPI(title="SuperEval API", version="2.0", lifespan=lifespan)

Fault = Literal["none", "degraded_method", "schema_broken", "lenient_judge", "slow_llm"]


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: str = Field(default="anon", max_length=64)
    history: list[dict] = Field(default_factory=list)


class SimulateIn(BaseModel):
    n: int = Field(default=40, ge=1, le=120)
    fault: Fault = "none"


def _trace_json(t: Trace, evals=None) -> dict:
    evals = evals if evals is not None else t.evals
    return {
        "id": t.id, "created_at": t.created_at.isoformat(), "source": t.source,
        "agent_version": t.agent_version, "model": t.model, "question": t.question, "answer": t.answer,
        "language": t.language, "topic": t.topic, "latency_ms": t.latency_ms, "cost_usd": t.cost_usd,
        "input_tokens": t.input_tokens, "output_tokens": t.output_tokens, "error": t.error,
        "tool_calls": t.tool_calls, "analysis": t.analysis, "steps": t.steps,
        "evals": [{"evaluator": e.evaluator, "kind": e.kind, "score": e.score, "passed": e.passed,
                   "details": e.details} for e in evals],
    }


def _answer(question: str, session_id: str, history: list[dict], mode: str, on_event=None):
    if mode == "sim":
        from app.analyst.sim import sim_analysis  # deterministic stand-in (no key / over budget)

        return sim_analysis(question, settings.prompt_version, session_id=session_id, on_event=on_event)
    return handle_analysis(question, settings.prompt_version, source="user", session_id=session_id,
                           history=history, on_event=on_event)


@app.get("/health")
def health():
    cat = get_catalog()
    return {"ok": True, "live_llm": settings.live_llm, "agent_model": settings.agent_model,
            "judge_model": settings.judge_model, "prompt_version": settings.prompt_version,
            "snapshot": cat.meta["fetched_at"], "chains": cat.chains, "cities": cat.cities}


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    key = f"{body.session_id}:{request.client.host if request.client else '-'}"
    mode, reason = guard.mode_for(key)
    history = [m for m in body.history if m.get("role") in ("user", "assistant")][-6:]
    trace, evals = _answer(body.question, body.session_id, history, mode)
    maybe_run_monitors()
    return {"mode": mode, "mode_reason": reason, **_trace_json(trace, evals)}


@app.post("/chat/stream")
def chat_stream(body: ChatIn, request: Request):
    """Server-sent events: `tool_start`/`tool_end` per query as the analyst works, `answer` when it
    writes the final answer, then `done` with the full trace and its evals (same payload as /chat)."""
    key = f"{body.session_id}:{request.client.host if request.client else '-'}"
    mode, reason = guard.mode_for(key)
    history = [m for m in body.history if m.get("role") in ("user", "assistant")][-6:]
    events: queue.Queue = queue.Queue()

    def work():
        try:
            trace, evals = _answer(body.question, body.session_id, history, mode, on_event=events.put)
            events.put({"type": "done", "mode": mode, "mode_reason": reason, **_trace_json(trace, evals)})
        except Exception as exc:  # surface failures to the client instead of hanging the stream
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            maybe_run_monitors()

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            ev = events.get()
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
            if ev["type"] in ("done", "error"):
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/simulate")
def simulate(body: SimulateIn):
    """A free burst of synthetic questions through the simulated analyst, optionally with a fault
    injected (model regression, schema break, lenient judge, slow LLM), then the monitors run."""
    from app.analyst.semantic import get_semantic
    from app.analyst.sim import sim_analysis
    from app.traffic.analyst_traffic import Asker

    asker = Asker(get_semantic(), random.Random())
    fault = None if body.fault == "none" else body.fault
    for _ in range(body.n):
        q = asker.ask()
        sim_analysis(q.text, settings.prompt_version, session_id="burst", source="synthetic", intent=q.intent,
                     seed=random.randrange(1 << 30), fault=fault)
    if fault == "lenient_judge":  # the judge's daily canary check runs against the faulty judge too
        from app.evals.canaries import run as run_canaries

        run_canaries(live=False, leniency=0.85, judge_model="sim-judge (updated)")
    return {"generated": body.n, "fault": body.fault, **run_monitors()}


@app.get("/traces")
def traces(limit: int = 50, source: str | None = None):
    with SessionLocal() as s:
        q = select(Trace).order_by(Trace.created_at.desc()).limit(min(limit, 200))
        if source:
            q = q.where(Trace.source == source)
        return [_trace_json(t) for t in s.execute(q).scalars().all()]


@app.get("/traces/{trace_id}")
def trace(trace_id: str):
    with SessionLocal() as s:
        t = s.get(Trace, trace_id)
        if not t:
            raise HTTPException(404)
        return _trace_json(t)


@app.get("/alerts")
def alerts(status: str | None = None, limit: int = 100):
    with SessionLocal() as s:
        q = select(Alert).order_by(Alert.created_at.desc()).limit(limit)
        if status:
            q = q.where(Alert.status == status)
        return [{"id": a.id, "created_at": a.created_at.isoformat(), "rule": a.rule, "severity": a.severity,
                 "layer": a.layer, "metric": a.metric, "value": a.value, "threshold": a.threshold,
                 "message": a.message, "status": a.status,
                 "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None}
                for a in s.execute(q).scalars().all()]


@app.post("/monitors/run")
def monitors():
    return run_monitors()
