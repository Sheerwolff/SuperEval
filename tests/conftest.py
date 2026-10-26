"""Tests always run against a throwaway SQLite database, never the DATABASE_URL in .env (which may be the
production Neon database). Set before any app module loads settings; load_dotenv never overrides it."""
import os
import tempfile
from pathlib import Path

os.environ["DATABASE_URL"] = f"sqlite:///{(Path(tempfile.mkdtemp(prefix='supereval-tests-')) / 'test.db').as_posix()}"

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def golden_set():
    """The committed golden set, as on a freshly deployed app (CI starts from the same empty database)."""
    from app.evals import golden

    golden.import_file()
