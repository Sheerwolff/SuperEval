"""Operational store: traces, eval results, alerts, ingest runs.

SQLite locally, Postgres (Neon) in production; same SQLAlchemy models for both.
Price data itself lives in parquet + DuckDB (see app/data/catalog.py).
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from app.config import settings


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(16), default="user")  # user | synthetic | backfill | offline
    agent_version: Mapped[str] = mapped_column(String(32), index=True)
    model: Mapped[str] = mapped_column(String(64))
    prompt_hash: Mapped[str] = mapped_column(String(16))
    snapshot_date: Mapped[str | None] = mapped_column(String(32))
    question: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(4))
    topic: Mapped[str | None] = mapped_column(String(32))
    answer: Mapped[str] = mapped_column(Text)
    tool_calls: Mapped[list] = mapped_column(JSON, default=list)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text)
    analysis: Mapped[dict | None] = mapped_column(JSON)  # analyst's structured answer (app/analyst/answer.py)
    steps: Mapped[int] = mapped_column(Integer, default=0)
    canary: Mapped[bool] = mapped_column(Boolean, default=False)  # served by a canary deployment

    evals: Mapped[list[EvalResult]] = relationship(back_populates="trace", cascade="all, delete-orphan")


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    evaluator: Mapped[str] = mapped_column(String(48), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # deterministic | judge
    score: Mapped[float | None] = mapped_column(Float)  # 0..1, None = not applicable
    passed: Mapped[bool | None] = mapped_column(Boolean)
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    trace: Mapped[Trace] = relationship(back_populates="evals")


class HumanLabel(Base):
    """Labels you add on the Calibrate page; used to measure judge-human agreement."""

    __tablename__ = "human_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.id", ondelete="CASCADE"), index=True)
    labeler: Mapped[str] = mapped_column(String(64), default="owner")
    helpful: Mapped[bool] = mapped_column(Boolean)
    grounded: Mapped[bool] = mapped_column(Boolean)
    note: Mapped[str | None] = mapped_column(Text)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    rule: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(8))  # info | warn | critical
    metric: Mapped[str] = mapped_column(String(48))
    agent_version: Mapped[str | None] = mapped_column(String(32))
    value: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(12), default="open")  # open | resolved
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
    layer: Mapped[str] = mapped_column(String(8), default="agent")  # agent (layer 1) | judge (layer 2)


class HumanJudgment(Base):
    """A person's verdict on a trace (Review page). Visitors and the owner are kept apart:
    owner judgments drive judge calibration and can be promoted into the golden set."""

    __tablename__ = "human_judgments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(8))  # visitor | owner
    reviewer: Mapped[str | None] = mapped_column(String(64))  # session id, for de-duplication
    correct: Mapped[str] = mapped_column(String(8))  # yes | no | unsure
    method_ok: Mapped[bool | None] = mapped_column(Boolean)
    tags: Mapped[list] = mapped_column(JSON, default=list)  # failure taxonomy
    corrected_answer: Mapped[str | None] = mapped_column(Text)
    corrected_sql: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    promoted_to: Mapped[str | None] = mapped_column(String(16))  # golden item id


class JudgeFeedback(Base):
    """A chat user agreeing or disagreeing with the eval verdicts shown on an answer (Chat page).
    Unlike HumanJudgment, this rates the evaluators, not the answer."""

    __tablename__ = "judge_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(8))  # visitor | owner
    reviewer: Mapped[str | None] = mapped_column(String(64))  # session id, for de-duplication
    agree: Mapped[bool] = mapped_column(Boolean)
    disputed: Mapped[list] = mapped_column(JSON, default=list)  # evaluator names the user disagrees with
    note: Mapped[str | None] = mapped_column(Text)


class JudgeCalibration(Base):
    """One 'Calibrate' run on the Judge Validation page: the evidence set the live judge is fed from now on (approved
    golden questions), and the owner-annotated gold set re-judged without and with that evidence."""

    __tablename__ = "judge_calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    judge_model: Mapped[str] = mapped_column(String(64))
    evidence: Mapped[list] = mapped_column(JSON, default=list)  # [{id, question, reference_sql, answer_type}]
    summary: Mapped[dict] = mapped_column(JSON, default=dict)  # {"without": metrics, "with": metrics}
    rows: Mapped[list] = mapped_column(JSON, default=list)  # per gold item: human label + both verdicts
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


class UserSpend(Base):
    """Live LLM spend per user per day (chat + golden-set runs), for the $1/user/day cap."""

    __tablename__ = "user_spend"

    user_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    day: Mapped[str] = mapped_column(String(10), primary_key=True)  # UTC date, YYYY-MM-DD
    usd: Mapped[float] = mapped_column(Float, default=0.0)


class GoldenItem(Base):
    """One benchmark question. Drafts become CI-gating only once a human approves them
    (curated by owner judgments of its test answers on the Review page); items can also come from traces."""

    __tablename__ = "golden_items"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    question: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(4))
    qtype: Mapped[str] = mapped_column(String(24))  # lookup | comparison | ranking | spread | promo | geo | trap_*
    difficulty: Mapped[str] = mapped_column(String(8))  # easy | medium | hard
    answer_type: Mapped[str] = mapped_column(String(12))  # scalar | ranking | set | decline | assumption
    reference_sql: Mapped[str | None] = mapped_column(Text)
    answer_key: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(10), default="draft")  # draft | approved | rejected
    source: Mapped[str] = mapped_column(String(10), default="drafted")  # drafted | promoted
    source_trace_id: Mapped[str | None] = mapped_column(String(36))
    notes: Mapped[str | None] = mapped_column(Text)
    reviewed_by: Mapped[str | None] = mapped_column(String(16))


class JudgeCanaryResult(Base):
    """One canary item judged in one canary run (Layer 2: is the judge still right on known cases?)."""

    __tablename__ = "judge_canary_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    canary_id: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(24))  # what the canary tests, e.g. wrong_price
    dimension: Mapped[str] = mapped_column(String(16))  # groundedness | relevance | scope
    expected: Mapped[bool] = mapped_column(Boolean)
    got: Mapped[bool | None] = mapped_column(Boolean)
    correct: Mapped[bool] = mapped_column(Boolean)
    judge_model: Mapped[str] = mapped_column(String(64))
    reasoning: Mapped[str | None] = mapped_column(Text)


class Deployment(Base):
    """Which agent version is live. The 'deploy buggy version' toggle writes here."""

    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    agent_version: Mapped[str] = mapped_column(String(32))
    note: Mapped[str | None] = mapped_column(Text)


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    chain: Mapped[str] = mapped_column(String(32))
    ok: Mapped[bool] = mapped_column(Boolean)
    price_rows: Mapped[int] = mapped_column(Integer, default=0)
    latest_price_update: Mapped[str | None] = mapped_column(String(32))
    fetched_at: Mapped[str | None] = mapped_column(String(32))


def _engine():
    url = settings.database_url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


engine = _engine()
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db() -> None:
    if engine.url.drivername.startswith("sqlite"):
        from pathlib import Path

        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def current_version(session=None) -> str:
    """The production prompt version (one prompt; no deploy switching in the app)."""
    return settings.prompt_version
