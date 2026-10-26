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

from fastapi import FastAPI, Header, HTTPException, Request
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
    user_key: str | None = Field(default=None, max_length=64)  # who pays: the UI sends a hashed client IP
    history: list[dict] = Field(default_factory=list)


class BenchIn(BaseModel):
    user_key: str = Field(min_length=1, max_length=64)


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


def _user(body: ChatIn, request: Request) -> str:
    return body.user_key or f"{body.session_id}:{request.client.host if request.client else '-'}"


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
    key = _user(body, request)
    mode, reason = guard.mode_for(key)
    history = [m for m in body.history if m.get("role") in ("user", "assistant")][-6:]
    trace, evals = _answer(body.question, body.session_id, history, mode)
    if mode == "live":
        guard.charge(key, trace.cost_usd)
    maybe_run_monitors()
    return {"mode": mode, "mode_reason": reason, **_trace_json(trace, evals)}


@app.post("/chat/stream")
def chat_stream(body: ChatIn, request: Request):
    """Server-sent events: `tool_start`/`tool_end` per query as the analyst works, `answer` when it
    writes the final answer, then `done` with the full trace and its evals (same payload as /chat)."""
    key = _user(body, request)
    mode, reason = guard.mode_for(key)
    history = [m for m in body.history if m.get("role") in ("user", "assistant")][-6:]
    events: queue.Queue = queue.Queue()

    def work():
        try:
            trace, evals = _answer(body.question, body.session_id, history, mode, on_event=events.put)
            if mode == "live":
                guard.charge(key, trace.cost_usd)
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


# ---- golden-set test runs -------------------------------------------------------
_bench_lock = threading.Lock()
_bench: dict = {"running": False, "done": 0, "total": 0, "passed": 0, "started_at": None, "last_error": None}


@app.post("/bench/run")
def bench_run(body: BenchIn):
    """Start a live golden-set run (approved + draft items) in the background. Each question's cost is charged
    to the user, and the run stops once they reach their daily budget."""
    from app.evals import bench, golden

    if not settings.live_llm:
        raise HTTPException(400, "No API key configured, so the live agent can't run.")
    remaining = guard.remaining(body.user_key)
    if remaining <= 0:
        raise HTTPException(429, f"You've used today's ${guard.budget:.2f} of live LLM calls.")
    items = golden.load(("approved", "draft"))
    if not items:
        raise HTTPException(400, "The golden set is empty.")
    if not _bench_lock.acquire(blocking=False):
        raise HTTPException(409, "A golden-set run is already in progress.")
    _bench.update(running=True, done=0, total=len(items), passed=0, started_at=time.time(), last_error=None)

    def progress(row: dict) -> None:
        guard.charge(body.user_key, row["cost_usd"])
        _bench["done"] += 1
        _bench["passed"] += int(bool(row["passed"]))

    def work():
        try:
            bench.run_bench(items, settings.prompt_version, live=True, max_cost=remaining, on_progress=progress,
                            include_drafts=True, trigger="ui", user_key=body.user_key,
                            save_traces=True)
        except Exception as exc:  # report on the status endpoint instead of dying silently
            _bench["last_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            _bench["running"] = False
            _bench_lock.release()

    threading.Thread(target=work, daemon=True).start()
    return {"started": True, "total": len(items)}


@app.get("/bench/status")
def bench_status(user_key: str | None = None):
    spent = guard.spent_today(user_key) if user_key else None
    return {**_bench, "spent_today": spent, "budget": guard.budget, "live_llm": settings.live_llm}


# ---- judge calibration (Judge Validation page, owner only) -----------------------------
_cal_lock = threading.Lock()
_cal: dict = {"running": False, "done": 0, "total": 0, "started_at": None, "last_error": None}


@app.post("/judge/calibrate")
def judge_calibrate(body: BenchIn, x_admin_key: str | None = Header(default=None)):
    """Save a new evidence set for the live judge and re-judge the gold set without and with it, in the background.
    Owner only (ADMIN_KEY); each judge call is charged to the owner's daily budget."""
    import hmac

    from app.evals import judge_validation

    if not settings.admin_key or not hmac.compare_digest(x_admin_key or "", settings.admin_key):
        raise HTTPException(403, "Only the owner can calibrate the judge.")
    if not settings.live_llm:
        raise HTTPException(400, "No API key configured, so the live judge can't run.")
    remaining = guard.remaining(body.user_key)
    if remaining <= 0:
        raise HTTPException(429, f"You've used today's ${guard.budget:.2f} of live LLM calls.")
    n = len(judge_validation.gold_set())
    if not n:
        raise HTTPException(400, "No owner-annotated answers yet: judge some on the Golden set page first.")
    if not _cal_lock.acquire(blocking=False):
        raise HTTPException(409, "A calibration is already running.")
    _cal.update(running=True, done=0, total=2 * n, started_at=time.time(), last_error=None)

    def progress(cost: float) -> None:
        guard.charge(body.user_key, cost)
        _cal["done"] += 1

    def work():
        try:
            judge_validation.calibrate(on_progress=progress, max_cost=remaining)
        except Exception as exc:
            _cal["last_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            _cal["running"] = False
            _cal_lock.release()

    threading.Thread(target=work, daemon=True).start()
    return {"started": True, "gold": n}


@app.get("/judge/calibrate/status")
def judge_calibrate_status():
    return _cal


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
