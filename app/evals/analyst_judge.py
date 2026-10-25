"""LLM judge for analyst answers: what deterministic checks can't fully see (does it answer the
question, is the method sound, are caveats honest, is it clear). A stronger model than the agent
(gpt-5 judging gpt-5-mini) on a sample of traffic; its own reliability is Layer 2.

Outcomes (names shared with judge_monitor.py):
  judge_groundedness  every stated fact is supported by the query results (bool)
  judge_method        method validity >= 4/5
  judge_helpfulness   answers the question >= 4/5 (raw 1-5 kept for bias metrics)
  judge_caveats       caveats/assumptions appropriate >= 4/5
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import zlib

from pydantic import BaseModel, Field

from app.analyst.semantic import SCHEMA_DOC
from app.config import cost_usd, settings
from app.evals.outcome import EvalOutcome

RUBRIC = """You review one answer from a data analyst agent about Israeli grocery prices. You get the
question, the SQL queries the agent ran with (truncated) results, and its structured answer.

Score each 1-5 using these anchors:
- answers_question: 5 = directly answers what was asked; 3 = partly / answers a nearby question;
  1 = doesn't answer. A correct decline of an unanswerable question (history, online prices, hours,
  uncovered places) is a 5; declining an answerable one is a 1.
- method_validity: 5 = compares like with like (same products across chains/cities, e.g.
  chain_price_index or explicit item_code matching), right filters and aggregation; 3 = defensible
  but loose; 1 = misleading, e.g. AVG(price) over each chain's different products (Simpson's trap),
  wrong denominator, wrong city or category filter, conclusions from a tiny sample.
- caveats_quality: 5 = states assumptions for ambiguous questions and real limits (small samples,
  snapshot only); 3 = generic; 1 = missing where it matters or overclaims.
- clarity: 5 = a lay reader gets the answer in one read; 1 = confusing.
- grounded: true only if every number and fact in the answer appears in, or follows directly from,
  the query results shown or the metric definitions below.
Write `reasoning` (max 3 sentences) BEFORE the scores.

The agent's documented metric definitions:
""" + SCHEMA_DOC[SCHEMA_DOC.index("Canonical metrics"):]
RUBRIC_HASH = hashlib.sha256(RUBRIC.encode()).hexdigest()[:10]


class AnalystVerdict(BaseModel):
    reasoning: str = Field(description="Max three sentences")
    answers_question: int = Field(ge=1, le=5)
    method_validity: int = Field(ge=1, le=5)
    caveats_quality: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    grounded: bool


def _payload(question: str, queries: list[dict], analysis: dict, limit: int = 9000) -> str:
    qs = [{"id": q["id"], "sql": q["sql"], "error": q.get("error"), "columns": q.get("columns"),
           "rows": (q.get("rows") or [])[:15], "row_count": q.get("row_count")} for q in queries]
    s = json.dumps({"question": question, "queries": qs, "answer": analysis}, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…(truncated)"


def judge_live(question: str, queries: list[dict], analysis: dict,
               model: str | None = None) -> tuple[AnalystVerdict, dict]:
    from openai import OpenAI

    model = model or settings.judge_model
    client = OpenAI(api_key=settings.openai_api_key, timeout=120)
    extra = {"reasoning_effort": "low"} if model.startswith(("gpt-5", "o")) else {"temperature": 0}
    start = time.perf_counter()
    resp = client.chat.completions.parse(
        model=model, response_format=AnalystVerdict,
        messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": _payload(question, queries,
                                                                                               analysis)}],
        **extra)
    u = resp.usage
    meta = {"judge_model": model, "rubric_hash": RUBRIC_HASH, "latency_ms": round((time.perf_counter() - start) * 1000),
            "input_tokens": u.prompt_tokens if u else 0, "output_tokens": u.completion_tokens if u else 0}
    meta["cost_usd"] = cost_usd(model, meta["input_tokens"], meta["output_tokens"])
    return resp.choices[0].message.parsed, meta


def judge_sim(seed_key: str, deterministic: list[EvalOutcome], leniency: float = 0.0,
              judge_model: str = "sim-judge") -> tuple[AnalystVerdict, dict]:
    """Simulated judge: follows the deterministic verdicts with realistic noise. `leniency` is the
    chance it waves a known-bad answer through (the seeded lenient-judge incident)."""
    rng = random.Random(zlib.crc32(seed_key.encode()))
    by = {o.evaluator: o for o in deterministic}

    def failed(name):
        return name in by and by[name].passed is False

    bad_ground = failed("grounding")
    bad_method = failed("method_lint") or failed("canonical_check")
    bad_scope = failed("decline_check")
    bad_assume = failed("assumption_check")
    lenient = (bad_ground or bad_method) and rng.random() < leniency
    grounded = True if lenient else not bad_ground
    if bad_ground and not lenient and rng.random() < 0.06:
        grounded = True  # judges miss things
    method = 5 if lenient else (2 if bad_method else rng.choice([4, 5, 5]))
    if bad_method and not lenient and rng.random() < 0.06:
        method = 4
    answers = 1 if bad_scope else (4 if lenient else (3 if (bad_ground or bad_method) else rng.choice([4, 5, 5])))
    caveats = 2 if bad_assume else rng.choice([3, 4, 4, 5])
    reasons = []
    if not grounded:
        reasons.append("Some numbers aren't in the query results.")
    if method <= 2:
        reasons.append("The comparison doesn't match products across chains/cities.")
    if bad_scope:
        reasons.append("The decline decision is wrong for this question.")
    if bad_assume:
        reasons.append("Ambiguous question without a stated assumption.")
    v = AnalystVerdict(reasoning=" ".join(reasons) or "Answers the question with a sound, grounded method.",
                       answers_question=answers, method_validity=method, caveats_quality=caveats,
                       clarity=rng.choice([4, 4, 5]), grounded=grounded)
    meta = {"judge_model": judge_model, "rubric_hash": RUBRIC_HASH, "latency_ms": round(4000 * rng.lognormvariate(0, .3)),
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    return v, meta


def outcomes(v: AnalystVerdict, meta: dict) -> list[EvalOutcome]:
    d = {"reasoning": v.reasoning, **meta}
    return [
        EvalOutcome("judge_groundedness", float(v.grounded), v.grounded, d, kind="judge"),
        EvalOutcome("judge_method", (v.method_validity - 1) / 4, v.method_validity >= 4,
                    {**d, "raw": v.method_validity}, kind="judge"),
        EvalOutcome("judge_helpfulness", (v.answers_question - 1) / 4, v.answers_question >= 4,
                    {**d, "raw": v.answers_question}, kind="judge"),
        EvalOutcome("judge_caveats", (v.caveats_quality - 1) / 4, v.caveats_quality >= 4,
                    {**d, "raw": v.caveats_quality}, kind="judge"),
    ]


def run_judge(trace, deterministic: list[EvalOutcome], force_sim: bool = False, leniency: float = 0.0,
              judge_model: str = "sim-judge") -> list[EvalOutcome]:
    use_live = settings.live_llm and not force_sim and trace.model != "sim"
    try:
        if use_live:
            v, meta = judge_live(trace.question, trace.tool_calls or [], trace.analysis or {})
        else:
            v, meta = judge_sim(trace.id, deterministic, leniency, judge_model)
    except Exception as exc:
        return [EvalOutcome("judge_error", None, False, {"error": f"{type(exc).__name__}: {exc}",
                                                         "rubric_hash": RUBRIC_HASH}, kind="judge")]
    return outcomes(v, meta)
