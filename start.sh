#!/usr/bin/env sh
# One container, two processes: the API on an internal port, the dashboard on the public one.
set -e
export PYTHONPATH=/app
PORT="${PORT:-7860}"

# latest prices from the daily ingest (GitHub Release asset); falls back to the committed sample
python -m app.data.fetch_snapshot || true

# first boot on an empty database: seed 30 days of history so the dashboard isn't blank
python -c "from app.db import init_db, SessionLocal, Trace; init_db(); s=SessionLocal(); exit(0 if s.query(Trace).count() else 1)" \
  || python -m app.traffic.backfill --days 30

uvicorn app.api.main:app --host 127.0.0.1 --port 8000 &
exec python -m streamlit run ui/Home.py --server.port "$PORT" --server.address 0.0.0.0
