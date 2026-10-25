"""One request through the whole system: agent -> trace -> evals -> store."""
from __future__ import annotations

import random
from datetime import datetime

from app.agent.agent import AgentResult, run_agent
from app.config import settings
from app.db import EvalResult, SessionLocal, Trace
from app.evals.deterministic import run_deterministic
from app.evals.judge import run_judge


def to_trace(res: AgentResult, source: str, session_id: str | None, created_at: datetime | None = None) -> Trace:
    t = Trace(
        id=res.id, session_id=session_id, source=source, agent_version=res.agent_version, model=res.model,
        prompt_hash=res.prompt_hash, snapshot_date=res.snapshot_date, question=res.question,
        language=res.language, topic=res.topic, answer=res.answer, tool_calls=res.tool_calls,
        input_tokens=res.input_tokens, output_tokens=res.output_tokens, cost_usd=res.cost_usd,
        latency_ms=res.latency_ms, error=res.error,
    )
    if created_at is not None:
        t.created_at = created_at
    return t


def evaluate(trace: Trace, judge: bool | None = None, force_sim_judge: bool = False,
             judge_leniency: float = 0.0, judge_model: str = "sim-judge") -> list[EvalResult]:
    det = run_deterministic(trace)
    outcomes = list(det)
    if judge is None:
        judge = random.random() < settings.judge_sample_rate
    if judge:
        judged = run_judge(trace, det, force_sim=force_sim_judge, leniency=judge_leniency, judge_model=judge_model)
        outcomes += judged
        # the trace's cost is the full cost of serving it, evaluation included
        trace.cost_usd = (trace.cost_usd or 0.0) + (judged[0].details.get("cost_usd") or 0.0)
    return [
        EvalResult(evaluator=o.evaluator, kind=o.kind, score=o.score, passed=o.passed, details=o.details,
                   created_at=trace.created_at)
        for o in outcomes
    ]


def handle_question(
    question: str,
    version: str,
    source: str = "user",
    session_id: str | None = None,
    history: list[dict] | None = None,
    force_sim: bool = False,
    on_event=None,
) -> tuple[Trace, list[EvalResult]]:
    res = run_agent(question, version, history=history, force_sim=force_sim, on_event=on_event)
    trace = to_trace(res, source, session_id)
    with SessionLocal() as s:
        s.add(trace)
        s.flush()
        evals = evaluate(trace)
        for e in evals:
            e.trace_id = trace.id
        s.add_all(evals)
        s.commit()
    return trace, evals
