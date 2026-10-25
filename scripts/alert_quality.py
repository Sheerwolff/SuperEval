"""Backfill with several seeds and score the alerting: every seeded incident should be caught
(recall), and every alert should belong to an incident (precision).

    python scripts/alert_quality.py --seeds 1 2 3 4 5
Each seed uses its own throwaway SQLite file.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECK = r"""
from datetime import timedelta
from app.db import SessionLocal, Alert, utcnow
from app.traffic.backfill import INCIDENTS
today = utcnow().replace(minute=0, second=0, microsecond=0).replace(hour=0)
wins = []
for inc in INCIDENTS:
    d0 = today + timedelta(days=inc["day"])
    wins.append((inc["kind"], d0 + timedelta(hours=inc["start"]), d0 + timedelta(hours=inc["end"] + 6)))
caught, fp = {}, []
with SessionLocal() as s:
    for a in s.query(Alert).order_by(Alert.created_at):
        hit = [k for k, lo, hi in wins if lo <= a.created_at <= hi]
        for k in hit:
            caught.setdefault(k, []).append(f"{a.layer}:{a.rule}")
        if not hit:
            fp.append(f"{a.created_at:%m-%d %H:%M} {a.rule} {a.agent_version or ''}")
by = "; ".join(f"{k} <- {sorted(set(v))}" for k, v in caught.items())
print(f"caught {len(caught)}/{len(wins)} incidents, {len(fp)} false positive(s) {fp} | {by}")
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    for seed in args.seeds:
        db = Path(tempfile.gettempdir()) / f"pricewatch-seed-{seed}.db"
        env = {**os.environ, "DATABASE_URL": f"sqlite:///{db.as_posix()}", "PYTHONPATH": str(ROOT)}
        subprocess.run([sys.executable, "-m", "app.traffic.backfill", "--reset", "--seed", str(seed)],
                       cwd=ROOT, env=env, check=True, capture_output=True)
        out = subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, env=env, check=True,
                             capture_output=True, text=True).stdout.strip()
        print(f"seed {seed}: {out}", flush=True)
        db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
