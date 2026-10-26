#!/usr/bin/env sh
# One container, two processes: the API on an internal port, the dashboard on the public one.
set -e
export PYTHONPATH=/app
PORT="${PORT:-8080}"

# latest prices from the daily ingest (GitHub Release asset); falls back to the committed sample
python -m app.data.fetch_snapshot || true

# reachable and seeded before serving: retries a database that's waking up, seeds only a confirmed-empty one
# (prefer scripts/seed_db.py for a remote database), and fails the start if the database stays unreachable
python -m app.boot

uvicorn app.api.main:app --host 127.0.0.1 --port 8000 &
exec python -m streamlit run ui/Home.py --server.port "$PORT" --server.address 0.0.0.0
