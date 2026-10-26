"""Fill a (remote) database in bulk: seed a fresh 30-day history locally and copy it, or copy another database.

    python scripts/seed_db.py                              # seed DATABASE_URL (from .env or the environment)
    python scripts/seed_db.py --to NEW_DATABASE_URL        # seed the URL held in another variable
    python scripts/seed_db.py --from DATABASE_URL --to NEW_DATABASE_URL --replace   # move a database

Seeding row by row across the network is slow (Neon from Europe took >15 min), and inside Cloud Run it can run past
the startup timeout. So the history is generated into a throwaway SQLite file (~2 min) and copied in chunks (~1 min).
Arguments name environment variables, never URLs, so connection strings stay out of shells and logs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _url(var: str) -> str:
    from dotenv import dotenv_values

    url = os.environ.get(var) or dotenv_values(ROOT / ".env").get(var)
    if not url:
        sys.exit(f"{var} isn't set (in the environment or .env)")
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return url.replace(prefix, "postgresql+psycopg://", 1)
    return url


def _seed_sqlite(days: int) -> str:
    path = Path(tempfile.mkdtemp(prefix="supereval-seed-")) / "seed.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{path.as_posix()}"}
    for cmd in (["-m", "app.traffic.backfill", "--days", str(days)], ["-m", "app.evals.golden", "import"]):
        subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    return env["DATABASE_URL"]


def copy(src_url: str, dst_url: str, replace: bool) -> None:
    from sqlalchemy import Integer, create_engine, func, select, text

    from app.db import Base, Trace

    src, dst = create_engine(src_url), create_engine(dst_url, pool_pre_ping=True)
    tables = Base.metadata.sorted_tables  # parents before children
    Base.metadata.create_all(dst)
    with dst.connect() as c:
        existing = c.execute(select(func.count()).select_from(Trace.__table__)).scalar()
    if existing and not replace:
        sys.exit(f"the target already has {existing} traces; pass --replace to overwrite it")
    with dst.begin() as c:
        if dst.dialect.name == "postgresql":
            c.execute(text(f"TRUNCATE {', '.join(t.name for t in tables)} RESTART IDENTITY CASCADE"))
        else:
            for t in reversed(tables):
                c.execute(t.delete())
    for t in tables:
        with src.connect() as s:
            rows = [dict(r._mapping) for r in s.execute(select(t))]
        with dst.begin() as c:
            for i in range(0, len(rows), 2000):
                c.execute(t.insert(), rows[i:i + 2000])
        print(f"  {t.name}: {len(rows)}")
    if dst.dialect.name == "postgresql":  # explicit ids were copied: move each id sequence past them
        with dst.begin() as c:
            for t in tables:
                pk = list(t.primary_key.columns)
                if len(pk) == 1 and isinstance(pk[0].type, Integer) and pk[0].autoincrement in (True, "auto"):
                    c.execute(text(f"SELECT setval(pg_get_serial_sequence('{t.name}', '{pk[0].name}'), "
                                   f"COALESCE((SELECT MAX({pk[0].name}) FROM {t.name}), 0) + 1, false)"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", help="variable holding the database to copy (default: a fresh seed)")
    ap.add_argument("--to", default="DATABASE_URL", help="variable holding the database to fill")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--replace", action="store_true", help="overwrite a target that already has data")
    args = ap.parse_args()
    start = time.time()
    dst = _url(args.to)
    if args.src:
        src = _url(args.src)
        print(f"copying {args.src} -> {args.to}")
    else:
        print(f"seeding {args.days} days locally…")
        src = _seed_sqlite(args.days)
        print(f"copying the seed -> {args.to}")
    copy(src, dst, args.replace)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
