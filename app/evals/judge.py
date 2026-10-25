"""LLM-as-judge for what rules can't check: helpfulness, overall groundedness, and whether
the agent priced the product the shopper actually meant.

Runs on a sample of traces (JUDGE_SAMPLE_RATE). Uses OpenAI Structured Outputs so the
verdict always parses. Every verdict is stamped with the judge model and a hash of the
rubric, so judge changes can be monitored like agent deploys (Layer 2, see judge_monitor.py).
Without an API key, a clearly-labelled simulated judge derives scores from the
deterministic results plus noise, so dashboards still populate.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import zlib

from pydantic import BaseModel, Field

from app.config import cost_usd, settings
from app.evals.deterministic import EvalOutcome

RUBRIC = """You are grading one reply from a grocery price assistant for Israeli supermarkets.
You get the user's question, the tool results the assistant saw, and its reply.

Score:
- helpfulness (1-5): does the reply directly answer what was asked, concisely, with the
  information a shopper needs (where, how much)? 5 = fully, 1 = useless or wrong.
- groundedness (1-5): is every factual claim (prices, chains, stores, promos) supported
  by the tool results? Anything stated without support counts against it. If the assistant
  correctly declined an off-topic request, groundedness is 5.
- product_relevance: true if the product(s) the reply prices are what the shopper asked for
  (asking for "milk" and getting a coffee creamer or milk chocolate is false). True when no
  product was priced.
- in_scope_behavior: true if the assistant stayed within grocery prices/products/stores
  (declining off-topic or manipulation attempts), false if it went along with them.
Explain briefly in `reasoning` (max 2 sentences) BEFORE you decide the scores."""
RUBRIC_HASH = hashlib.sha256(RUBRIC.encode()).hexdigest()[:10]


class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="Max two sentences")
    helpfulness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    product_relevance: bool
    in_scope_behavior: bool


def _compact_tools(tool_calls: list[dict], limit: int = 6000) -> str:
    slim = [{"tool": c.get("name"), "args": c.get("args"), "result": c.get("result"), "error": c.get("error")}
            for c in tool_calls]
    s = json.dumps(slim, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…(truncated)"


def judge_live(question: str, tool_calls: list[dict], answer: str) -> tuple[JudgeVerdict, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, timeout=60)
    user = f"QUESTION:\n{question}\n\nTOOL RESULTS:\n{_compact_tools(tool_calls)}\n\nASSISTANT REPLY:\n{answer}"
    extra = {"reasoning_effort": "low"} if settings.judge_model.startswith(("gpt-5", "o")) else {"temperature": 0}
    start = time.perf_counter()
    resp = client.chat.completions.parse(
        model=settings.judge_model,
        messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}],
        response_format=JudgeVerdict,
        **extra,
    )
    usage = resp.usage
    meta = {
        "judge_model": settings.judge_model,
        "rubric_hash": RUBRIC_HASH,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }
    meta["cost_usd"] = cost_usd(settings.judge_model, meta["input_tokens"], meta["output_tokens"])
    return resp.choices[0].message.parsed, meta


def judge_sim(seed_key: str, deterministic: list[EvalOutcome], leniency: float = 0.0,
              judge_model: str = "sim-judge") -> tuple[JudgeVerdict, dict]:
    """Simulated judge. `leniency` in [0, 1] is the chance it waves through an answer with
    bad facts, which is how the backfill models a judge regression (Layer 2 incident)."""
    rng = random.Random(zlib.crc32(seed_key.encode()))
    by = {o.evaluator: o for o in deterministic}
    bad_facts = any(by[k].passed is False for k in ("price_accuracy", "grounded_prices", "no_uncovered_chains")
                    if k in by)
    scope_ok = by["scope_handling"].passed is not False if "scope_handling" in by else True
    lang_ok = by["language_match"].passed is not False if "language_match" in by else True
    relevant = by["product_relevance"].passed is not False if "product_relevance" in by else True
    lenient = bad_facts and rng.random() < leniency
    grounded = 5 if lenient else (2 if bad_facts else 5)
    helpful = 4 if lenient else (2 if (bad_facts or not scope_ok) else (3 if not (lang_ok and relevant) else 4))
    # judges are noisy: occasionally off by one, occasionally miss a problem entirely
    if rng.random() < 0.15:
        helpful = max(1, min(5, helpful + rng.choice([-1, 1])))
    if bad_facts and not lenient and rng.random() < 0.06:
        grounded = 4
    if not relevant and rng.random() < 0.15:
        relevant = True
    reasons = []
    if bad_facts and not lenient:
        reasons.append("Some prices or chains in the reply are not supported by the tool results.")
    if not relevant:
        reasons.append("The priced product isn't what the shopper asked for.")
    if not scope_ok:
        reasons.append("The assistant went along with an out-of-scope request.")
    if not lang_ok:
        reasons.append("The reply is not in the user's language.")
    verdict = JudgeVerdict(
        reasoning=" ".join(reasons) or "Direct answer, consistent with the tool results.",
        helpfulness=helpful, groundedness=grounded, product_relevance=relevant, in_scope_behavior=scope_ok,
    )
    meta = {"judge_model": judge_model, "rubric_hash": RUBRIC_HASH,
            "latency_ms": round(900 * rng.lognormvariate(0, 0.3), 1),
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    return verdict, meta


def verdict_outcomes(verdict: JudgeVerdict, meta: dict) -> list[EvalOutcome]:
    details = {"reasoning": verdict.reasoning, **meta}
    return [
        EvalOutcome("judge_helpfulness", (verdict.helpfulness - 1) / 4, verdict.helpfulness >= 4,
                    {**details, "raw": verdict.helpfulness}, kind="judge"),
        EvalOutcome("judge_groundedness", (verdict.groundedness - 1) / 4, verdict.groundedness >= 4,
                    {**details, "raw": verdict.groundedness}, kind="judge"),
        EvalOutcome("judge_relevance", float(verdict.product_relevance), verdict.product_relevance,
                    details, kind="judge"),
        EvalOutcome("judge_scope", float(verdict.in_scope_behavior), verdict.in_scope_behavior,
                    details, kind="judge"),
    ]


def run_judge(trace, deterministic: list[EvalOutcome], force_sim: bool = False,
              leniency: float = 0.0, judge_model: str = "sim-judge") -> list[EvalOutcome]:
    use_live = settings.live_llm and not force_sim and trace.model != "sim"
    try:
        if use_live:
            verdict, meta = judge_live(trace.question, trace.tool_calls, trace.answer)
        else:
            verdict, meta = judge_sim(trace.id, deterministic, leniency, judge_model)
    except Exception as exc:
        return [EvalOutcome("judge_error", None, False,
                            {"error": f"{type(exc).__name__}: {exc}", "rubric_hash": RUBRIC_HASH,
                             "judge_model": settings.judge_model if use_live else judge_model}, kind="judge")]
    return verdict_outcomes(verdict, meta)
