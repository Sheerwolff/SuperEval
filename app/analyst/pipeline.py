"""One analytical question through the system: analyst -> trace -> evals -> store."""
from __future__ import annotations

from datetime import datetime

from app.analyst.agent import AnalystResult, run_analyst
from app.db import EvalResult, SessionLocal, Trace


def to_trace(res: AnalystResult, source: str, session_id: str | None, created_at: datetime | None = None,
             snapshot_date: str | None = None, topic: str | None = None) -> Trace:
    t = Trace(
        id=res.id, session_id=session_id, source=source, agent_version=res.version, model=res.model,
        prompt_hash=res.prompt_hash, snapshot_date=snapshot_date, question=res.question, language=res.language,
        topic=topic, answer=res.markdown, tool_calls=res.queries, analysis=res.answer,
        input_tokens=res.input_tokens, output_tokens=res.output_tokens, cost_usd=res.cost_usd,
        latency_ms=res.latency_ms, error=res.error, steps=res.steps,
    )
    if created_at is not None:
        t.created_at = created_at
    return t


def evaluate(trace: Trace, judge: bool | None = None, **judge_kw) -> list[EvalResult]:
    """Layer 1 (+ sampled judge) evaluators for an analyst trace."""
    from app.evals.online import run_online_evals

    return run_online_evals(trace, judge=judge, **judge_kw)


def handle_analysis(question: str, version: str, source: str = "user", session_id: str | None = None,
                    history: list[dict] | None = None, on_event=None,
                    model: str | None = None) -> tuple[Trace, list[EvalResult]]:
    from app.analyst.semantic import get_semantic
    from app.analyst.topics import classify
    from app.data.catalog import get_catalog

    res = run_analyst(question, version, history=history, on_event=on_event, model=model)
    trace = to_trace(res, source, session_id, snapshot_date=get_catalog().snapshot_date,
                     topic=classify(question, get_semantic().cities))
    with SessionLocal() as s:
        s.add(trace)
        s.flush()
        evals = evaluate(trace)
        for e in evals:
            e.trace_id = trace.id
        s.add_all(evals)
        s.commit()
    return trace, evals
