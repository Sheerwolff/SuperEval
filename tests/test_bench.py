import uuid

from app.api.cost_guard import CostGuard
from app.config import settings
from app.db import SessionLocal, UserSpend, init_db
from app.evals import bench, golden


def test_run_bench_reports_progress_and_summary():
    items = golden.load(("approved", "draft"))[:3]
    seen = []
    rec = bench.run_bench(items, "a2", live=False, on_progress=seen.append, record=False)
    assert [r["id"] for r in seen] == [i["id"] for i in items]
    assert rec["summary"]["overall"]["n"] == 3 and rec["model"] == "sim" and not rec["live"]


def test_cost_guard_caps_each_user_per_day(monkeypatch):
    init_db()
    monkeypatch.setattr(type(settings), "live_llm", property(lambda self: True))
    g = CostGuard(1.0)
    a, b = f"test-{uuid.uuid4().hex[:8]}", f"test-{uuid.uuid4().hex[:8]}"
    try:
        g.charge(a, 0.6)
        assert g.mode_for(a) == ("live", None) and abs(g.remaining(a) - 0.4) < 1e-9
        g.charge(a, 0.5)
        assert g.mode_for(a) == ("sim", "user_budget_exhausted")
        assert g.mode_for(b) == ("live", None)  # another user is unaffected
    finally:
        with SessionLocal() as s:
            s.query(UserSpend).filter(UserSpend.user_key.in_([a, b])).delete()
            s.commit()
