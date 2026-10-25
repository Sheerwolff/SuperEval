"""Layer 2: monitoring the LLM judge itself.

Layer 1 trusts the judge's verdicts about the agent. A judge that silently goes lenient
(a provider model update, a rubric edit) would make the agent look *better* while it isn't,
so the judge gets its own signals:

- ground-truth recall: on answers the deterministic evals *know* are bad (wrong/ungrounded
  prices, invented chains), how often does the judge also fail them?
- agreement with ground truth overall, and false-fail rate on known-good answers
- canary accuracy: verdicts on a fixed set of known cases (canaries.py)
- self-consistency: flip rate when the same trace is judged twice
- bias: helpfulness vs answer length; helpfulness gap between Hebrew and English
- ops: parse/API error rate, latency, cost
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import func, select

from app.db import EvalResult, JudgeCanaryResult, SessionLocal, Trace

TRUTH_EVALS = ("price_accuracy", "grounded_prices", "no_uncovered_chains")
JUDGE_EVALS = ("judge_groundedness", "judge_helpfulness", "judge_relevance", "judge_error", "judge_repeat")
JUDGE_COLUMNS = ["trace_id", "created_at", "agent_version", "language", "answer_len", "judge_model",
                 "rubric_hash", "ground_pass", "help_raw", "relevance_pass", "judge_latency_ms", "error",
                 "truth_bad", "repeat_pass"]


def judge_frame_from(evals: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    """One row per judged trace. `evals` needs trace_id, created_at, evaluator, passed, details;
    `traces` needs id, agent_version, language, answer_len."""
    if evals.empty:
        return pd.DataFrame(columns=JUDGE_COLUMNS)
    truth = evals[evals.evaluator.isin(TRUTH_EVALS) & evals.passed.notna()]
    truth_bad = truth.assign(bad=~truth.passed.astype(bool)).groupby("trace_id").bad.any()

    j = evals[evals.evaluator.isin(JUDGE_EVALS)]
    if j.empty:
        return pd.DataFrame(columns=JUDGE_COLUMNS)

    def pick(name, col="passed"):
        sub = j[j.evaluator == name].drop_duplicates("trace_id", keep="last").set_index("trace_id")
        return sub[col] if col in sub else pd.Series(dtype=object)

    ground = j[j.evaluator.isin(["judge_groundedness", "judge_error"])].drop_duplicates("trace_id", keep="last")
    base = ground.set_index("trace_id")[["created_at", "details"]].copy()
    details = base.pop("details").apply(lambda d: d or {})
    base["judge_model"] = details.map(lambda d: d.get("judge_model"))
    base["rubric_hash"] = details.map(lambda d: d.get("rubric_hash"))
    base["judge_latency_ms"] = details.map(lambda d: d.get("latency_ms"))
    base["help_raw"] = pick("judge_helpfulness", "details").map(lambda d: (d or {}).get("raw"))
    base["ground_pass"] = pick("judge_groundedness")
    base["relevance_pass"] = pick("judge_relevance")
    base["error"] = base.index.isin(j[j.evaluator == "judge_error"].trace_id)
    base["repeat_pass"] = pick("judge_repeat")
    base["truth_bad"] = truth_bad.reindex(base.index)
    tr = traces.set_index("id")[["agent_version", "language", "answer_len"]]
    out = base.join(tr, how="left").reset_index().rename(columns={"index": "trace_id"})
    return out[JUDGE_COLUMNS]


def load_judge_frame(since: datetime, until: datetime | None = None, session=None) -> pd.DataFrame:
    own = session is None
    session = session or SessionLocal()
    try:
        q = select(EvalResult.trace_id, EvalResult.created_at, EvalResult.evaluator, EvalResult.passed,
                   EvalResult.details).where(EvalResult.created_at >= since,
                                             EvalResult.evaluator.in_(TRUTH_EVALS + JUDGE_EVALS))
        if until is not None:
            q = q.where(EvalResult.created_at < until)
        evals = pd.DataFrame(session.execute(q).all(),
                             columns=["trace_id", "created_at", "evaluator", "passed", "details"])
        tq = select(Trace.id, Trace.agent_version, Trace.language,
                    func.length(Trace.answer).label("answer_len")).where(Trace.created_at >= since)
        if until is not None:
            tq = tq.where(Trace.created_at < until)
        traces = pd.DataFrame(session.execute(tq).all(), columns=["id", "agent_version", "language", "answer_len"])
    finally:
        if own:
            session.close()
    evals["created_at"] = pd.to_datetime(evals.created_at)
    # deterministic rows don't need details; drop them to keep memory flat
    evals.loc[~evals.evaluator.isin(JUDGE_EVALS), "details"] = None
    return judge_frame_from(evals, traces)


def load_canaries(since: datetime, until: datetime | None = None, session=None) -> pd.DataFrame:
    own = session is None
    session = session or SessionLocal()
    try:
        q = select(JudgeCanaryResult.created_at, JudgeCanaryResult.run_id, JudgeCanaryResult.canary_id,
                   JudgeCanaryResult.kind, JudgeCanaryResult.dimension, JudgeCanaryResult.expected,
                   JudgeCanaryResult.got, JudgeCanaryResult.correct, JudgeCanaryResult.judge_model,
                   JudgeCanaryResult.reasoning).where(JudgeCanaryResult.created_at >= since)
        if until is not None:
            q = q.where(JudgeCanaryResult.created_at < until)
        df = pd.DataFrame(session.execute(q).all())
    finally:
        if own:
            session.close()
    if df.empty:
        return pd.DataFrame(columns=["created_at", "run_id", "canary_id", "kind", "dimension", "expected",
                                     "got", "correct", "judge_model", "reasoning"])
    df["created_at"] = pd.to_datetime(df.created_at)
    return df


# ---- metrics ------------------------------------------------------------------
def recall_on_bad(j: pd.DataFrame) -> tuple[float | None, int]:
    """Share of known-bad answers the judge also failed on groundedness."""
    bad = j[(j.truth_bad == True) & j.ground_pass.notna()]  # noqa: E712
    if bad.empty:
        return None, 0
    return float((~bad.ground_pass.astype(bool)).mean()), len(bad)


def agreement(j: pd.DataFrame) -> tuple[float | None, int]:
    sub = j[j.truth_bad.notna() & j.ground_pass.notna()]
    if sub.empty:
        return None, 0
    agree = sub.ground_pass.astype(bool) == ~sub.truth_bad.astype(bool)
    return float(agree.mean()), len(sub)


def false_fail_rate(j: pd.DataFrame) -> tuple[float | None, int]:
    good = j[(j.truth_bad == False) & j.ground_pass.notna()]  # noqa: E712
    if good.empty:
        return None, 0
    return float((~good.ground_pass.astype(bool)).mean()), len(good)


def flip_rate(j: pd.DataFrame) -> tuple[float | None, int]:
    sub = j[j.repeat_pass.notna() & j.ground_pass.notna()]
    if sub.empty:
        return None, 0
    return float((sub.repeat_pass.astype(bool) != sub.ground_pass.astype(bool)).mean()), len(sub)


def error_rate(j: pd.DataFrame) -> tuple[float | None, int]:
    if j.empty:
        return None, 0
    return float(j.error.astype(bool).mean()), len(j)


def length_bias(j: pd.DataFrame) -> tuple[float | None, int]:
    """Spearman correlation of helpfulness with answer length, on known-good answers.
    Judges tend to reward longer answers; a strong positive value is a red flag."""
    sub = j[(j.truth_bad != True) & j.help_raw.notna() & j.answer_len.notna()]  # noqa: E712
    if len(sub) < 20 or sub.help_raw.nunique() < 2:
        return None, len(sub)
    return float(sub.answer_len.rank().corr(sub.help_raw.astype(float).rank())), len(sub)


def language_gap(j: pd.DataFrame) -> tuple[float | None, int]:
    """Mean helpfulness (1-5) for Hebrew minus English, on known-good answers."""
    sub = j[(j.truth_bad != True) & j.help_raw.notna()]  # noqa: E712
    he, en = sub[sub.language == "he"].help_raw, sub[sub.language == "en"].help_raw
    if len(he) < 10 or len(en) < 10:
        return None, len(sub)
    return float(he.astype(float).mean() - en.astype(float).mean()), len(sub)


def canary_runs(c: pd.DataFrame) -> pd.DataFrame:
    if c.empty:
        return pd.DataFrame(columns=["run_id", "created_at", "accuracy", "checks", "judge_model"])
    return (c.groupby("run_id").agg(created_at=("created_at", "first"), accuracy=("correct", "mean"),
                                    checks=("correct", "size"), judge_model=("judge_model", "first"))
            .reset_index().sort_values("created_at"))


JUDGE_METRICS = {
    "judge_recall_on_bad": ("Catches known-bad answers", recall_on_bad),
    "judge_agreement": ("Agrees with ground truth", agreement),
    "judge_false_fail": ("Fails known-good answers", false_fail_rate),
    "judge_flip_rate": ("Flip rate when re-judged", flip_rate),
    "judge_error_rate": ("Judge error rate", error_rate),
    "judge_length_bias": ("Length bias (Spearman ρ)", length_bias),
    "judge_language_gap": ("Helpfulness gap he − en", language_gap),
}
