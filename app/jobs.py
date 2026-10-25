"""Scheduled jobs (run by GitHub Actions against the production database).

    python -m app.jobs record-ingest          # after app.data.ingest: log the snapshot as an ingest run
    python -m app.jobs traffic --n 20         # synthetic shoppers on the deployed version, then monitors
    python -m app.jobs traffic --n 5 --live   # same, through the real OpenAI agent (costs tokens)
"""
from __future__ import annotations

import argparse
import random

from sqlalchemy import select

from app.data.catalog import get_catalog
from app.db import IngestRun, SessionLocal, current_version, init_db


def sync_ingest_meta() -> bool:
    """Record the snapshot on disk as an ingest run (once per fetched_at). Returns True if new."""
    meta = get_catalog().meta
    with SessionLocal() as s:
        if s.execute(select(IngestRun).where(IngestRun.fetched_at == meta["fetched_at"])).first():
            return False
        for chain, m in meta["chains"].items():
            s.add(IngestRun(
                chain=chain, ok=bool(m.get("ok")), price_rows=m.get("price_rows", 0),
                latest_price_update=m.get("published_at") or m.get("latest_price_update"),
                fetched_at=meta["fetched_at"],
            ))
        s.commit()
    return True


def synthetic_traffic(n: int, live: bool = False) -> dict:
    from app.monitor.alerts import run_monitors
    from app.pipeline import handle_question
    from app.traffic.synthetic import Shopper

    shopper = Shopper(get_catalog(), random.Random())
    with SessionLocal() as s:
        version = current_version(s)
    for _ in range(n):
        q = shopper.ask()
        if live:
            handle_question(q.text, version, source="synthetic", session_id="cron")
        else:
            from app.agent.agent import run_agent
            from app.pipeline import evaluate, to_trace

            res = run_agent(q.text, version, force_sim=True, intent=q.intent, seed=random.randrange(1 << 30))
            t = to_trace(res, "synthetic", "cron")
            with SessionLocal() as s:
                s.add(t)
                s.flush()
                for e in evaluate(t, force_sim_judge=True):
                    e.trace_id = t.id
                    s.add(e)
                s.commit()
    return {"version": version, "n": n, **run_monitors()}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record-ingest")
    t = sub.add_parser("traffic")
    t.add_argument("--n", type=int, default=20)
    t.add_argument("--live", action="store_true")
    args = ap.parse_args()
    init_db()
    if args.cmd == "record-ingest":
        print({"recorded": sync_ingest_meta()})
    else:
        print(synthetic_traffic(args.n, args.live))


if __name__ == "__main__":
    main()
