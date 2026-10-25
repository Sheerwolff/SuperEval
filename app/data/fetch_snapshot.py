"""Download the latest price snapshot published by the daily ingest job.

    python -m app.data.fetch_snapshot

The ingest job uploads `snapshot.tar.gz` and `meta.json` to a GitHub Release tagged
`snapshot-latest` (overwritten daily, so no history accumulates). This script checks the
remote meta.json and only downloads when it's newer than what's on disk. With no
SNAPSHOT_REPO configured, or no network, the app keeps using what it has (or the
committed sample in data/sample/).
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from app.config import ROOT

TARGET = ROOT / "data" / "snapshot"
TAG = "snapshot-latest"


def _url(repo: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{TAG}/{asset}"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pricewatch-evals"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(repo: str | None = None, target: Path = TARGET) -> str:
    repo = repo or os.getenv("SNAPSHOT_REPO")
    if not repo:
        return "skipped: SNAPSHOT_REPO not set"
    try:
        remote = json.loads(_get(_url(repo, "meta.json")))
    except Exception as exc:
        return f"skipped: could not read remote meta.json ({exc})"
    local_meta = target / "meta.json"
    if local_meta.exists():
        local = json.loads(local_meta.read_text(encoding="utf-8"))
        if local.get("fetched_at", "") >= remote.get("fetched_at", ""):
            return f"up to date ({local.get('fetched_at')})"
    try:
        blob = _get(_url(repo, "snapshot.tar.gz"), timeout=300)
    except Exception as exc:
        return f"failed to download snapshot ({exc}); keeping current data"
    tmp = Path(tempfile.mkdtemp(prefix="snapshot-"))
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        tar.extractall(tmp, filter="data")
    if not (tmp / "meta.json").exists():
        return "failed: archive has no meta.json; keeping current data"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(tmp), str(target))
    return f"downloaded snapshot {remote.get('fetched_at')}"


if __name__ == "__main__":
    print(fetch(sys.argv[1] if len(sys.argv) > 1 else None))
