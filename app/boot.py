"""Container startup: make sure the database is reachable and seeded, without ever mistaking an error for 'empty'.

    python -m app.boot

A remote database (Neon) can drop or be slow to wake for a few seconds. Treating that as an empty database used to
start a 30-day seed inside the container, which outlived Cloud Run's startup timeout (and would have duplicated
the data). So connection errors are retried, seeding only happens when the tables are confirmed empty, and a
database that stays unreachable fails the start with a clear message.
"""
from __future__ import annotations

import subprocess
import sys
import time

ATTEMPTS = 5


def counts() -> tuple[int, int]:
    from app.db import GoldenItem, SessionLocal, Trace, init_db

    init_db()
    with SessionLocal() as s:
        return s.query(Trace).count(), s.query(GoldenItem).count()


def main() -> int:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            traces, golden = counts()
            break
        except Exception as exc:  # connection drops, a database waking up, DNS hiccups
            print(f"boot: database not ready (attempt {attempt}/{ATTEMPTS}): {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:200]}", flush=True)
            if attempt == ATTEMPTS:
                print("boot: database unreachable; not seeding. Check DATABASE_URL.", file=sys.stderr, flush=True)
                return 1
            time.sleep(min(2 ** attempt, 15))
    print(f"boot: database ready ({traces} traces, {golden} golden items)", flush=True)
    if not traces:
        print("boot: empty database, seeding 30 days of history. For a remote database prefer "
              "scripts/seed_db.py: row-by-row seeding over the network can outlast the startup timeout.", flush=True)
        subprocess.run([sys.executable, "-m", "app.traffic.backfill", "--days", "30"], check=True)
    if not golden:
        subprocess.run([sys.executable, "-m", "app.evals.golden", "import"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
