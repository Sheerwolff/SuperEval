"""PriceWatch API: serves the agent and exposes the monitoring data.

    uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import json
import random
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agent.versions import load_versions
from app.api.cost_guard import guard
from app.config import settings
from app.data.catalog import get_catalog
from app.db import Alert, Deployment, HumanLabel, IngestRun, SessionLocal, Trace, current_version, init_db
from app.jobs import sync_ingest_meta
from app.monitor.alerts import run_monitors
from app.pipeline import handle_question

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


app = FastAPI(title="PriceWatch Evals API", version="1.0", lifespan=lifespan)


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: str = Field(default="anon", max_length=64)
    history: list[dict] = Field(default_factory=list)


class DeployIn(BaseModel):
    version: str
    note: str | None = None


class SimulateIn(BaseModel):
    n: int = Field(default=20, ge=1, le=60)


class LabelIn(BaseModel):
    trace_id: str
    helpful: bool
    grounded: bool
    note: str | None = None


def _trace_json(t: Trace, evals=None) -> dict:
    evals = evals if evals is not None else t.evals
    return {
        "id": t.id, "created_at": t.created_at.isoformat(), "source": t.source,
        "agent_version": t.agent_version, "model": t.model, "question": t.question, "answer": t.answer,
        "language": t.language, "topic": t.topic, "latency_ms": t.latency_ms, "cost_usd": t.cost_usd,
        "input_tokens": t.input_tokens, "output_tokens": t.output_tokens, "error": t.error,
        "tool_calls": t.tool_calls,
        "evals": [{"evaluator": e.evaluator, "kind": e.kind, "score": e.score, "passed": e.passed,
                   "details": e.details} for e in evals],
    }


@app.get("/health")
def health():
    cat = get_catalog()
    return {
        "ok": True, "live_llm": settings.live_llm, "agent_model": settings.agent_model,
        "judge_model": settings.judge_model, "snapshot": cat.meta["fetched_at"],
        "chains": cat.chains, "cities": cat.cities,
    }


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    key = f"{body.session_id}:{request.client.host if request.client else '-'}"
    mode, reason = guard.mode_for(key)
    with SessionLocal() as s:
        version = current_version(s)
    history = [m for m in body.history if m.get("role") in ("user", "assistant")][-6:]
    trace, evals = handle_question(body.question, version, source="user", session_id=body.session_id,
                                   history=history, force_sim=(mode == "sim"))
    maybe_run_monitors()
    return {"mode": mode, "mode_reason": reason, **_trace_json(trace, evals)}


@app.get("/deployment")
def get_deployment():
    with SessionLocal() as s:
        rows = s.execute(select(Deployment).order_by(Deployment.created_at.desc()).limit(20)).scalars().all()
        return {
            "current": current_version(s),
            "versions": {k: {"label": v.label, "description": v.description, "prompt_hash": v.prompt_hash}
                         for k, v in load_versions().items()},
            "history": [{"at": d.created_at.isoformat(), "version": d.agent_version, "note": d.note} for d in rows],
        }


@app.post("/deployment")
def deploy(body: DeployIn):
    if body.version not in load_versions():
        raise HTTPException(400, f"unknown version {body.version}")
    with SessionLocal() as s:
        s.add(Deployment(agent_version=body.version, note=body.note or "deployed from dashboard"))
        s.commit()
    return get_deployment()


@app.post("/simulate")
def simulate(body: SimulateIn):
    """Free traffic burst through the simulator on the currently deployed version."""
    from app.agent.agent import run_agent
    from app.pipeline import evaluate, to_trace
    from app.traffic.synthetic import Shopper

    with SessionLocal() as s:
        version = current_version(s)
    shopper = Shopper(get_catalog(), random.Random())
    with SessionLocal() as s:
        for _ in range(body.n):
            q = shopper.ask()
            res = run_agent(q.text, version, force_sim=True, intent=q.intent, seed=random.randrange(1 << 30))
            t = to_trace(res, "synthetic", "burst")
            s.add(t)
            s.flush()
            for e in evaluate(t, force_sim_judge=True):
                e.trace_id = t.id
                s.add(e)
        s.commit()
    result = run_monitors()
    return {"version": version, "generated": body.n, **result}


@app.get("/traces")
def traces(limit: int = 50, version: str | None = None, source: str | None = None):
    with SessionLocal() as s:
        q = select(Trace).order_by(Trace.created_at.desc()).limit(min(limit, 200))
        if version:
            q = q.where(Trace.agent_version == version)
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
        return [{
            "id": a.id, "created_at": a.created_at.isoformat(), "rule": a.rule, "severity": a.severity,
            "metric": a.metric, "agent_version": a.agent_version, "value": a.value, "threshold": a.threshold,
            "message": a.message, "status": a.status,
            "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
        } for a in s.execute(q).scalars().all()]


@app.post("/monitors/run")
def monitors():
    return run_monitors()


@app.post("/labels")
def add_label(body: LabelIn):
    with SessionLocal() as s:
        if not s.get(Trace, body.trace_id):
            raise HTTPException(404, "trace not found")
        s.add(HumanLabel(trace_id=body.trace_id, helpful=body.helpful, grounded=body.grounded, note=body.note))
        s.commit()
    return {"ok": True}


@app.get("/data/health")
def data_health():
    cat = get_catalog()
    with SessionLocal() as s:
        runs = s.execute(select(IngestRun).order_by(IngestRun.created_at.desc()).limit(200)).scalars().all()
    return {
        "meta": json.loads(json.dumps(cat.meta, default=str)),
        "runs": [{"at": r.created_at.isoformat(), "chain": r.chain, "ok": r.ok, "price_rows": r.price_rows,
                  "published_at": r.latest_price_update} for r in runs],
    }
