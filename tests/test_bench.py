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


def test_golden_traces_are_saved_reviewable_and_curated():
    from app.db import EvalResult, GoldenItem, Trace
    from app.evals.human import curate_golden, golden_for_trace
    from app.monitor.aggregates import load_traces

    init_db()
    items = golden.load(("approved", "draft"))[:2]
    with SessionLocal() as s:  # remember statuses/references so the test leaves the set as it found it
        saved = {i["id"]: (s.get(GoldenItem, i["id"]).status, s.get(GoldenItem, i["id"]).reference_sql,
                           s.get(GoldenItem, i["id"]).notes) for i in items}
    rec = bench.run_bench(items, "a2", live=False, record=False, save_traces=True)
    run_session = f"golden:{rec['at']}"
    try:
        with SessionLocal() as s:
            tids = [t for (t,) in s.query(Trace.id).filter(Trace.session_id == run_session, Trace.source == "golden")]
        assert len(tids) == 2
        assert not load_traces().id.isin(tids).any()  # golden answers stay out of the monitoring metrics
        gid, matched = golden_for_trace(tids[0])
        assert gid in saved and matched is not None
        verdict = "yes" if matched else "no"  # agree with the scorer -> approved
        assert "approved" in curate_golden(tids[0], verdict)
        assert "draft" in curate_golden(tids[0], "no" if matched else "yes")  # disagree, no SQL -> stays draft
        assert "your SQL" in curate_golden(tids[0], "no", corrected_sql="SELECT 1 AS v")
        assert "rejected" in curate_golden(tids[0], "yes", bad_question=True)
        with SessionLocal() as s:
            assert s.get(GoldenItem, gid).status == "rejected" and s.get(GoldenItem, gid).reference_sql == "SELECT 1 AS v"
    finally:
        with SessionLocal() as s:
            for gid, (status, sql, notes) in saved.items():
                g = s.get(GoldenItem, gid)
                g.status, g.reference_sql, g.notes = status, sql, notes
            s.query(EvalResult).filter(EvalResult.trace_id.in_(tids)).delete(synchronize_session=False)
            s.query(Trace).filter(Trace.id.in_(tids)).delete(synchronize_session=False)
            s.commit()
