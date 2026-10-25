"""Seed 30 days of production history with five realistic incidents, then replay the
monitors hour by hour so the alerts table tells the story.

    python -m app.traffic.backfill --days 30 --reset

All backfilled traces come from the simulator (model="sim", source="backfill") and are
labelled as such in the UI. The incidents:

  1. Bad deploy      day -21, 10:00-16:00  v2 prompt goes live -> price hallucinations -> rollback
  2. Stale feed      day -12 .. -10        Victory ingest fails for ~60h
  3. Latency spike   day -5, 14:00-20:00   upstream LLM slowness, 3x latency
  4. Input drift     day -3                pre-holiday surge of basket questions
  5. Lenient judge   day -9 .. -6          judge provider model update starts waving through
                                           ungrounded answers. Layer 1 looks *better*; only
                                           Layer 2 (canaries, ground-truth recall) catches it.
"""
from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta

import pandas as pd

from app.agent.sim import SimAgent
from app.agent.versions import get_version
from app.data.catalog import get_catalog
from app.db import Alert, Base, Deployment, EvalResult, IngestRun, SessionLocal, engine, init_db, utcnow
from app.evals import canaries
from app.evals.deterministic import run_deterministic
from app.evals.judge import judge_sim
from app.evals.judge_monitor import judge_frame_from
from app.monitor.alerts import run_monitors
from app.monitor.rules import Context
from app.pipeline import evaluate, to_trace
from app.traffic.synthetic import DEFAULT_MIX, Shopper

INCIDENTS = [
    {"day": -21, "start": 10, "end": 16, "kind": "bad_deploy",
     "title": "v2 'faster & cheaper' prompt deployed, rolled back after alerts"},
    {"day": -12, "start": 6, "end": 6 + 60, "kind": "stale_feed", "chain": "Victory",
     "title": "Victory price feed failed for ~60h"},
    {"day": -5, "start": 14, "end": 20, "kind": "latency", "title": "Upstream LLM latency degradation (3x)"},
    {"day": -3, "start": 0, "end": 24, "kind": "drift", "title": "Pre-holiday surge of basket questions"},
    {"day": -9, "start": 9, "end": 9 + 72, "kind": "judge_lenient", "layer": "judge",
     "title": "Judge model update made the judge lenient; pinned back to the previous model"},
]
JUDGE_MODEL = "gpt-5-mini (sim)"
JUDGE_MODEL_UPDATED = "gpt-5-mini-2026-10 (sim)"
JUDGE_LENIENCY = 0.85
REPEAT_RATE = 0.1  # share of judged traces judged a second time (self-consistency)


def hourly_volume(ts: datetime) -> float:
    """Israeli grocery rhythm: busy Thu/Fri mornings, near-dead Friday night & Saturday."""
    dow, h = ts.weekday(), ts.hour  # Mon=0 .. Sun=6
    daily = 0.5 + 0.5 * math.sin((h - 7) / 24 * 2 * math.pi) if 6 <= h <= 23 else 0.08
    if dow == 3:
        daily *= 1.4
    if dow == 4:
        daily *= 1.3 if h < 14 else 0.2
    if dow == 5:
        daily *= 0.15 if h < 19 else 0.8
    return 14 * daily


def _in(ts: datetime, start_of_day: datetime, inc: dict) -> bool:
    day0 = start_of_day + timedelta(days=inc["day"])
    return day0 + timedelta(hours=inc["start"]) <= ts < day0 + timedelta(hours=inc["end"])


def run(days: int = 30, reset: bool = False, seed: int = 42) -> dict:
    init_db()
    if reset:
        Base.metadata.drop_all(engine)
        init_db()
    rng = random.Random(seed)
    catalog = get_catalog()
    agent = SimAgent(catalog)
    shopper = Shopper(catalog, rng)
    now = utcnow().replace(minute=0, second=0, microsecond=0)
    today = now.replace(hour=0)
    start = today - timedelta(days=days)
    bad, stale, slow, drift, lenient = INCIDENTS

    traces, evals = [], []
    with SessionLocal() as s:
        s.add(Deployment(created_at=start, agent_version="v1", note="initial release"))
        d0 = today + timedelta(days=bad["day"])
        s.add(Deployment(created_at=d0 + timedelta(hours=bad["start"]), agent_version="v2",
                         note="'faster & cheaper' prompt: fewer tool calls"))
        s.add(Deployment(created_at=d0 + timedelta(hours=bad["end"]), agent_version="v1",
                         note="rollback after price_accuracy SLO alert"))
        # daily ingests at 06:00 (files published ~03:30)
        for d in range(days + 1):
            run_at = start + timedelta(days=d, hours=6)
            if run_at > now:
                break
            for chain in catalog.chains:
                failed = chain == stale["chain"] and _in(run_at, today, stale)
                s.add(IngestRun(
                    created_at=run_at, chain=chain, ok=not failed,
                    price_rows=0 if failed else int(len(catalog.prices[catalog.prices.chain == chain])),
                    latest_price_update=None if failed else (run_at - timedelta(hours=2, minutes=30)).isoformat(),
                    fetched_at=run_at.isoformat(),
                ))
        s.commit()

    ts = start
    while ts < now:
        version = "v2" if _in(ts, today, bad) else "v1"
        mult = 3.0 if _in(ts, today, slow) else 1.0
        judge_bad = _in(ts, today, lenient)
        judge_kw = {"judge_leniency": JUDGE_LENIENCY if judge_bad else 0.0,
                    "judge_model": JUDGE_MODEL_UPDATED if judge_bad else JUDGE_MODEL}
        mix = DEFAULT_MIX
        if _in(ts, today, drift):
            mix = {"price_lookup": 0.35, "basket": 0.50, "stores": 0.05, "off_topic": 0.06, "injection": 0.04}
        n = _poisson(hourly_volume(ts) * (1.0 if version == "v1" else 1.2), rng)
        for _ in range(n):
            q = shopper.ask(mix)
            res = agent.run(q.text, get_version(version), intent=q.intent, seed=rng.randrange(1 << 30),
                            latency_multiplier=mult)
            created = ts + timedelta(seconds=rng.randrange(3600))
            t = to_trace(res, "backfill", f"sim-{rng.randrange(400)}", created_at=created)
            ev = evaluate(t, judge=True, force_sim_judge=True, **judge_kw)
            if rng.random() < REPEAT_RATE:
                verdict, meta = judge_sim(t.id + "#repeat", run_deterministic(t), judge_kw["judge_leniency"],
                                          judge_kw["judge_model"])
                ev.append(EvalResult(evaluator="judge_repeat", kind="judge_meta", created_at=t.created_at,
                                     score=(verdict.groundedness - 1) / 4, passed=verdict.groundedness >= 4,
                                     details=meta))
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

    # daily judge canary runs at 06:30
    with SessionLocal() as s:
        for d in range(days + 1):
            at = start + timedelta(days=d, hours=6, minutes=30)
            if at > now:
                break
            judge_bad = _in(at, today, lenient)
            canaries.run(live=False, at=at, leniency=JUDGE_LENIENCY if judge_bad else 0.0,
                         judge_model=JUDGE_MODEL_UPDATED if judge_bad else JUDGE_MODEL, session=s)
        d0 = today + timedelta(days=lenient["day"])
        s.add(Deployment(created_at=d0 + timedelta(hours=lenient["end"]), agent_version="v1",
                         note=f"judge pinned back to {JUDGE_MODEL} after canary alert (agent unchanged)"))
        # the seeded month ran v1; v3 (fixes from the first live golden-set run) goes live now
        s.add(Deployment(created_at=now, agent_version="v3",
                         note="v3: decisive on generic requests, batched basket searches (golden-set triage)"))
        s.commit()

    # replay monitors hourly on in-memory frames (fast) so alerts open/resolve at the right times
    tr_df = pd.DataFrame([{
        "id": t.id, "created_at": t.created_at, "source": t.source, "agent_version": t.agent_version,
        "model": t.model, "language": t.language, "topic": t.topic, "latency_ms": t.latency_ms,
        "cost_usd": t.cost_usd, "input_tokens": t.input_tokens, "output_tokens": t.output_tokens, "error": t.error,
    } for t in traces])
    by_id = tr_df.set_index("id")
    ev_df = pd.DataFrame([{
        "trace_id": e.trace_id, "created_at": e.created_at, "evaluator": e.evaluator, "kind": e.kind,
        "score": e.score, "passed": e.passed, "details": e.details if e.kind != "deterministic" else None,
    } for e in evals])
    for col in ("agent_version", "language", "topic", "source"):
        ev_df[col] = ev_df.trace_id.map(by_id[col])
    tr_df["created_at"] = pd.to_datetime(tr_df.created_at)
    ev_df["created_at"] = pd.to_datetime(ev_df.created_at)
    tr_df["answer_len"] = [len(t.answer) for t in traces]
    judge_df = judge_frame_from(ev_df, tr_df)
    judge_df["created_at"] = pd.to_datetime(judge_df.created_at)

    from app.evals.judge_monitor import load_canaries
    from app.monitor.alerts import deployed_as_of, freshness_as_of

    canary_df = load_canaries(start)

    ts = start + timedelta(days=1)
    with SessionLocal() as s:
        while ts <= now:
            lo = ts - timedelta(days=8)
            ctx = Context(
                now=ts,
                traces=tr_df[(tr_df.created_at >= lo) & (tr_df.created_at < ts)],
                evals=ev_df[(ev_df.created_at >= lo) & (ev_df.created_at < ts)],
                freshness=freshness_as_of(s, ts),
                spend_today=0.0, budget=0.0, freshness_sla_hours=36,
                deployed=deployed_as_of(s, ts),
                judge=judge_df[(judge_df.created_at >= lo) & (judge_df.created_at < ts)],
                canaries=canary_df[(canary_df.created_at >= ts - timedelta(days=2)) & (canary_df.created_at < ts)],
            )
            run_monitors(ts, notify=False, ctx=ctx)
            ts += timedelta(hours=1)

    with SessionLocal() as s:
        n_alerts = s.query(Alert).count()
    return {"traces": len(traces), "evals": len(evals), "alerts": n_alerts, "from": str(start), "to": str(now)}


def _poisson(lam: float, rng: random.Random) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(run(args.days, args.reset, args.seed))


if __name__ == "__main__":
    main()
