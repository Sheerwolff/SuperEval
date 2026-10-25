"""Scheduled jobs (run by GitHub Actions against the production database).

    python -m app.jobs record-ingest          # after app.data.ingest: log the snapshot as an ingest run
    python -m app.jobs traffic --n 20         # synthetic questions through the simulated analyst, then monitors
    python -m app.jobs traffic --n 2 --live   # same, through the real OpenAI analyst (costs tokens)
"""
from __future__ import annotations

import argparse
import random

from sqlalchemy import select

from app.config import settings
from app.data.catalog import get_catalog
from app.db import IngestRun, SessionLocal, init_db


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
    from app.analyst.pipeline import handle_analysis
    from app.analyst.semantic import get_semantic
    from app.analyst.sim import sim_analysis
    from app.monitor.alerts import run_monitors
    from app.traffic.analyst_traffic import Asker

    asker = Asker(get_semantic(), random.Random())
    version = settings.prompt_version
    for _ in range(n):
        q = asker.ask()
        if live:
            handle_analysis(q.text, version, source="synthetic", session_id="cron")
        else:
            sim_analysis(q.text, version, session_id="cron", source="synthetic", intent=q.intent,
                         seed=random.randrange(1 << 30))
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
