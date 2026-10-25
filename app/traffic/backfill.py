"""Seed 30 days of production history for the analyst with six realistic incidents, then replay the
monitors hour by hour so the alerts table tells the story.

    python -m app.traffic.backfill --days 30 --reset

All backfilled traces come from the simulator (model="sim", source="backfill"); the simulated judge
scores every trace here (production samples 30%) so Layer 2 has data. The incidents:

  1. Model regression  day -21, 10:00-18:00  provider update: comparisons average unmatched products.
                                             Grounding stays green; method lint + canonical fire.
  2. Schema change     day -14, 13:00-16:00  a table rename breaks every query (completion, SQL health)
  3. Lenient judge     day -9 .. -6          judge update waves bad answers through; only Layer 2 sees it
  4. Stale feed        day -12 .. -10        one chain's prices stop updating (freshness)
  5. Latency spike     day -5, 14:00-20:00   upstream LLM slowness, 3x latency
  6. Input drift       day -3                a surge of ambiguous "which chain is cheapest?" questions
"""
from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta

import pandas as pd

from app.analyst.pipeline import to_trace
from app.analyst.semantic import get_semantic
from app.analyst.sim import run_sim
from app.analyst.topics import classify
from app.config import settings
from app.data.catalog import get_catalog
from app.db import Alert, Base, EvalResult, IngestRun, SessionLocal, engine, init_db, utcnow
from app.evals import canaries
from app.evals.analyst_judge import judge_sim
from app.evals.judge_monitor import judge_frame_from, load_canaries
from app.evals.online import run_online_evals
from app.monitor.alerts import freshness_as_of, run_monitors
from app.monitor.rules import Context
from app.traffic.analyst_traffic import DEFAULT_MIX, Asker

INCIDENTS = [
    {"day": -21, "start": 10, "end": 18, "kind": "model_regression",
     "title": "Provider model update: comparisons averaged unmatched products"},
    {"day": -14, "start": 13, "end": 16, "kind": "schema_change", "title": "Table rename broke every query"},
    {"day": -9, "start": 9, "end": 9 + 72, "kind": "judge_lenient", "layer": "judge",
     "title": "Judge model update made the judge lenient"},
    {"day": -12, "start": 6, "end": 6 + 60, "kind": "stale_feed", "chain": "Victory",
     "title": "Victory price feed stopped updating"},
    {"day": -5, "start": 14, "end": 20, "kind": "latency", "title": "Upstream LLM latency degradation (3x)"},
    {"day": -3, "start": 0, "end": 24, "kind": "drift", "title": "Surge of ambiguous questions"},
]
JUDGE_MODEL = "gpt-5 (sim)"
JUDGE_MODEL_UPDATED = "gpt-5-2026-10 (sim)"
JUDGE_LENIENCY = 0.85
REPEAT_RATE = 0.1


def hourly_volume(ts: datetime) -> float:
    """Busy weekday daytime, quiet nights, Friday evening and Saturday."""
    dow, h = ts.weekday(), ts.hour
    daily = 0.5 + 0.5 * math.sin((h - 7) / 24 * 2 * math.pi) if 7 <= h <= 23 else 0.08
    if dow == 4:
        daily *= 1.2 if h < 14 else 0.25
    if dow == 5:
        daily *= 0.2 if h < 19 else 0.8
    return 11 * daily


def _in(ts: datetime, today: datetime, inc: dict) -> bool:
    d0 = today + timedelta(days=inc["day"])
    return d0 + timedelta(hours=inc["start"]) <= ts < d0 + timedelta(hours=inc["end"])


def _poisson(lam: float, rng: random.Random) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def run(days: int = 30, reset: bool = False, seed: int = 42) -> dict:
    init_db()
    if reset:
        Base.metadata.drop_all(engine)
        init_db()
    rng = random.Random(seed)
    sem = get_semantic()
    asker = Asker(sem, rng)
    now = utcnow().replace(minute=0, second=0, microsecond=0)
    today = now.replace(hour=0)
    start = today - timedelta(days=days)
    regress, schema, lenient, stale, slow, drift = INCIDENTS
    version = settings.prompt_version

    with SessionLocal() as s:  # daily ingests at 06:00
        chains = get_catalog().chains
        for d in range(days + 1):
            at = start + timedelta(days=d, hours=6)
            if at > now:
                break
            for chain in chains:
                failed = chain == stale["chain"] and _in(at, today, stale)
                s.add(IngestRun(created_at=at, chain=chain, ok=not failed, price_rows=0 if failed else 1,
                                latest_price_update=None if failed else (at - timedelta(hours=2, minutes=30)).isoformat(),
                                fetched_at=at.isoformat()))
        s.commit()

    traces, evals = [], []
    ts = start
    while ts < now:
        bad_judge = _in(ts, today, lenient)
        mix = DEFAULT_MIX
        if _in(ts, today, drift):
            mix = {**DEFAULT_MIX, "ambiguous": 0.50, "comparison": 0.15}
        for _ in range(_poisson(hourly_volume(ts), rng)):
            q = asker.ask(mix)
            res = run_sim(q.text, version, intent=q.intent, seed=rng.randrange(1 << 30),
                          latency_multiplier=3.0 if _in(ts, today, slow) else 1.0,
                          schema_broken=_in(ts, today, schema), degraded_method=_in(ts, today, regress))
            t = to_trace(res, "backfill", f"sim-{rng.randrange(500)}", created_at=ts + timedelta(seconds=rng.randrange(3600)),
                         snapshot_date=get_catalog().snapshot_date, topic=classify(q.text, sem.cities))
            ev = run_online_evals(t, judge=True, force_sim=True, leniency=JUDGE_LENIENCY if bad_judge else 0.0,
                                  judge_model=JUDGE_MODEL_UPDATED if bad_judge else JUDGE_MODEL)
            if t.analysis is not None and rng.random() < REPEAT_RATE:
                from app.evals.online import run_deterministic

                v, meta = judge_sim(t.id + "#repeat", run_deterministic(t),
                                    JUDGE_LENIENCY if bad_judge else 0.0,
                                    JUDGE_MODEL_UPDATED if bad_judge else JUDGE_MODEL)
                ev.append(EvalResult(evaluator="judge_repeat", kind="judge_meta", created_at=t.created_at,
                                     score=float(v.grounded and v.method_validity >= 4),
                                     passed=v.grounded and v.method_validity >= 4, details=meta))
            for e in ev:
                e.trace_id = t.id
            traces.append(t)
            evals.extend(ev)
        ts += timedelta(hours=1)

    with SessionLocal() as s:
        s.add_all(traces)
        s.flush()
        s.add_all(evals)
        s.commit()
        for d in range(days + 1):  # daily judge canaries at 06:30
            at = start + timedelta(days=d, hours=6, minutes=30)
            if at > now:
                break
            bad = _in(at, today, lenient)
            canaries.run(live=False, at=at, leniency=JUDGE_LENIENCY if bad else 0.0,
                         judge_model=JUDGE_MODEL_UPDATED if bad else JUDGE_MODEL, session=s)

    # replay the monitors hourly on in-memory frames, so alerts open and resolve at the right times
    tr_df = pd.DataFrame([{"id": t.id, "created_at": t.created_at, "source": t.source, "agent_version": t.agent_version,
                           "model": t.model, "language": t.language, "topic": t.topic, "latency_ms": t.latency_ms,
                           "cost_usd": t.cost_usd, "input_tokens": t.input_tokens, "output_tokens": t.output_tokens,
                           "error": t.error, "answer_len": len(t.answer or "")} for t in traces])
    by_id = tr_df.set_index("id")
    ev_df = pd.DataFrame([{"trace_id": e.trace_id, "created_at": e.created_at, "evaluator": e.evaluator,
                           "kind": e.kind, "score": e.score, "passed": e.passed,
                           "details": e.details if e.kind != "deterministic" else None} for e in evals])
    for col in ("agent_version", "language", "topic", "source"):
        ev_df[col] = ev_df.trace_id.map(by_id[col])
    tr_df["created_at"] = pd.to_datetime(tr_df.created_at)
    ev_df["created_at"] = pd.to_datetime(ev_df.created_at)
    judge_df = judge_frame_from(ev_df, tr_df)
    judge_df["created_at"] = pd.to_datetime(judge_df.created_at)
    canary_df = load_canaries(start)

    ts = start + timedelta(days=1)
    with SessionLocal() as s:
        while ts <= now:
            lo = ts - timedelta(days=8)
            ctx = Context(
                now=ts, traces=tr_df[(tr_df.created_at >= lo) & (tr_df.created_at < ts)],
                evals=ev_df[(ev_df.created_at >= lo) & (ev_df.created_at < ts)],
                freshness=freshness_as_of(s, ts), spend_today=0.0, budget=0.0, freshness_sla_hours=36,
                judge=judge_df[(judge_df.created_at >= lo) & (judge_df.created_at < ts)],
                canaries=canary_df[(canary_df.created_at >= ts - timedelta(days=2)) & (canary_df.created_at < ts)],
            )
            run_monitors(ts, notify=False, ctx=ctx)
            ts += timedelta(hours=1)
    with SessionLocal() as s:
        n_alerts = s.query(Alert).count()
    return {"traces": len(traces), "evals": len(evals), "alerts": n_alerts, "from": str(start), "to": str(now)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(run(args.days, args.reset, args.seed))


if __name__ == "__main__":
    main()
