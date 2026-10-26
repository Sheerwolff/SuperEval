"""Turn traces + eval rows into tidy frames and time series for rules and charts."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import select

from app.db import EvalResult, SessionLocal, Trace

# metric name -> (evaluator, aggregation). "rate" = share of applicable traces that passed.
# metric name -> (evaluator, aggregation). "rate" = share of applicable traces that passed.
METRICS = {
    "completion": ("completion", "rate"),
    "grounding": ("grounding", "rate"),
    "sql_health": ("sql_health", "rate"),
    "method_lint": ("method_lint", "rate"),
    "canonical_check": ("canonical_check", "rate"),
    "chart_lint": ("chart_lint", "rate"),
    "decline_check": ("decline_check", "rate"),
    "assumption_check": ("assumption_check", "rate"),
    "language_match": ("language_match", "rate"),
    "judge_method": ("judge_method", "mean"),
    "judge_helpfulness": ("judge_helpfulness", "mean"),
    "judge_groundedness": ("judge_groundedness", "rate"),
}
METRIC_LABELS = {
    "completion": "Task completion",
    "grounding": "Grounding",
    "sql_health": "SQL health",
    "method_lint": "Sound method (lint)",
    "canonical_check": "Agrees with canonical metric",
    "chart_lint": "Chart checks",
    "decline_check": "Correct declines",
    "assumption_check": "Assumptions stated",
    "language_match": "Language match",
    "judge_method": "Judge: method",
    "judge_helpfulness": "Judge: answers the question",
    "judge_groundedness": "Judge: grounded",
    "latency_p95_ms": "Latency p95",
    "cost_per_request": "Cost / request",
    "error_rate": "Error rate",
}


def load_traces(since: datetime | None = None, until: datetime | None = None, session=None) -> pd.DataFrame:
    q = select(
        Trace.id, Trace.created_at, Trace.source, Trace.agent_version, Trace.model, Trace.language,
        Trace.topic, Trace.latency_ms, Trace.cost_usd, Trace.input_tokens, Trace.output_tokens, Trace.error,
    ).where(Trace.source != "golden")  # golden-set test answers aren't production traffic
    if since is not None:
        q = q.where(Trace.created_at >= since)
    if until is not None:
        q = q.where(Trace.created_at < until)
    own = session is None
    session = session or SessionLocal()
    try:
        df = pd.DataFrame(session.execute(q).all())
    finally:
        if own:
            session.close()
    if df.empty:
        return pd.DataFrame(columns=["id", "created_at", "source", "agent_version", "model", "language", "topic",
                                     "latency_ms", "cost_usd", "input_tokens", "output_tokens", "error"])
    df["created_at"] = pd.to_datetime(df.created_at)
    return df


def load_evals(since: datetime | None = None, until: datetime | None = None, session=None) -> pd.DataFrame:
    q = select(
        EvalResult.trace_id, EvalResult.created_at, EvalResult.evaluator, EvalResult.kind,
        EvalResult.score, EvalResult.passed, Trace.agent_version, Trace.language, Trace.topic, Trace.source,
    ).join(Trace, Trace.id == EvalResult.trace_id).where(Trace.source != "golden")
    if since is not None:
        q = q.where(EvalResult.created_at >= since)
    if until is not None:
        q = q.where(EvalResult.created_at < until)
    own = session is None
    session = session or SessionLocal()
    try:
        df = pd.DataFrame(session.execute(q).all())
    finally:
        if own:
            session.close()
    if df.empty:
        return pd.DataFrame(columns=["trace_id", "created_at", "evaluator", "kind", "score", "passed",
                                     "agent_version", "language", "topic", "source"])
    df["created_at"] = pd.to_datetime(df.created_at)
    return df


def metric_value(metric: str, evals: pd.DataFrame, traces: pd.DataFrame) -> tuple[float | None, int]:
    """Value of one metric over the given rows, plus the sample size behind it."""
    if metric == "latency_p95_ms":
        return (float(traces.latency_ms.quantile(0.95)) if len(traces) else None), len(traces)
    if metric == "cost_per_request":
        return (float(traces.cost_usd.mean()) if len(traces) else None), len(traces)
    if metric == "error_rate":
        return (float(traces.error.notna().mean()) if len(traces) else None), len(traces)
    evaluator, agg = METRICS[metric]
    rows = evals[(evals.evaluator == evaluator) & evals.score.notna()]
    if rows.empty:
        return None, 0
    if agg == "rate":
        return float(rows.passed.astype(float).mean()), len(rows)
    return float(rows.score.mean()), len(rows)


def timeseries(metric: str, evals: pd.DataFrame, traces: pd.DataFrame, freq: str = "1h",
               by: str | None = None) -> pd.DataFrame:
    """Columns: bucket, [by], value, n."""
    out = []
    if metric in ("latency_p95_ms", "cost_per_request", "error_rate"):
        frame, key = traces, "created_at"
    else:
        evaluator, _ = METRICS[metric]
        frame, key = evals[(evals.evaluator == evaluator) & evals.score.notna()], "created_at"
    if frame.empty:
        return pd.DataFrame(columns=["bucket", by or "all", "value", "n"])
    frame = frame.assign(bucket=frame[key].dt.floor(freq))
    groups = ["bucket"] + ([by] if by else [])
    for keys, g in frame.groupby(groups):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if metric in ("latency_p95_ms", "cost_per_request", "error_rate"):
            val, n = metric_value(metric, evals.iloc[0:0], g)
        else:
            val, n = metric_value(metric, g, traces.iloc[0:0])
        row = {"bucket": keys[0], "value": val, "n": n}
        if by:
            row[by] = keys[1]
        out.append(row)
    return pd.DataFrame(out)
