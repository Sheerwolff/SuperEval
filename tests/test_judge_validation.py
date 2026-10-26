import uuid

import pytest

from app.db import EvalResult, HumanJudgment, JudgeCalibration, SessionLocal, Trace, init_db
from app.evals import evidence, judge_validation
from app.evals.analyst_judge import AnalystVerdict


def test_confusion_is_recall_first_on_wrong_answers():
    m = judge_validation.confusion([(True, True), (True, False), (False, True), (False, False), (False, False)])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["agreement"] == 0.6
    assert judge_validation.confusion([])["recall"] is None


def test_evidence_retrieval():
    items = [{"id": "gX", "question": "Which chain is cheapest in Haifa?",
              "reference_sql": "SELECT * FROM chain_price_index('Haifa', NULL)", "answer_type": "ranking"},
             {"id": "gY", "question": "How did milk prices change over the last year?", "reference_sql": None,
              "answer_type": "decline"}]
    g = evidence.retrieve("which chain is cheapest in Haifa", items)
    assert g["source"].startswith("golden gX") and g["queries"][0]["rows"]
    assert "declines" in evidence.retrieve("How did milk prices change over the last year?", items)["expected"]
    canon = evidence.retrieve("Is Rami Levy cheaper than Shufersal in Jerusalem?", [])
    assert canon["source"].startswith("canonical") and "Jerusalem" in canon["queries"][0]["sql"]
    assert evidence.retrieve("How much is a liter of milk?", []) is None


@pytest.fixture()
def gold():
    init_db()
    ids = []
    with SessionLocal() as s:
        for q, wrong in [("Which chain is cheapest in Haifa?", True), ("Which chain is cheapest in Tel Aviv?", False)]:
            tid = str(uuid.uuid4())
            s.add(Trace(id=tid, agent_version="a2", model="m", prompt_hash="h", question=q, language="en", answer="a",
                        tool_calls=[], analysis={"headline": "a", "claims": [], "declined": False}))
            s.flush()
            s.add(HumanJudgment(trace_id=tid, role="owner", correct="no" if wrong else "yes", tags=[]))
            ids.append(tid)
        s.commit()
    before = {c for (c,) in SessionLocal().query(JudgeCalibration.id)}
    yield ids
    with SessionLocal() as s:
        s.query(HumanJudgment).filter(HumanJudgment.trace_id.in_(ids)).delete()
        s.query(EvalResult).filter(EvalResult.trace_id.in_(ids)).delete()
        s.query(Trace).filter(Trace.id.in_(ids)).delete()
        s.query(JudgeCalibration).filter(~JudgeCalibration.id.in_(before)).delete(synchronize_session=False)
        s.commit()
    evidence._cache.update(at=0.0, items=None)


def test_calibrate_ablation_with_stub_judge(gold, monkeypatch):
    wrong_tid, _ = gold
    real = judge_validation.gold_set
    monkeypatch.setattr(judge_validation, "gold_set", lambda session=None: [g for g in real() if g["trace_id"] in gold])

    def stub(question, queries, analysis, ev):
        # without evidence the stub passes everything; with evidence it fails the Haifa answer
        fail = ev is not None and "Haifa" in question
        v = AnalystVerdict(reasoning="stub", answers_question=2 if fail else 5, method_validity=5, caveats_quality=5,
                           clarity=5, grounded=not fail)
        return v, {"cost_usd": 0.01}

    calls = []
    res = judge_validation.calibrate(judge_fn=stub, on_progress=calls.append)
    w, wo = res["summary"]["with"], res["summary"]["without"]
    assert res["n"] == 2 and wo["recall"] == 0.0 and w["recall"] == 1.0 and w["fp"] == 0
    assert len(calls) == 4 and abs(res["cost_usd"] - 0.04) < 1e-9  # both answers get canonical evidence
    latest = judge_validation.latest_calibration()
    assert latest["id"] == res["id"] and any(r["trace_id"] == wrong_tid and r["with"] for r in latest["rows"])



def test_calibrate_endpoint_is_owner_only(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import main
    from app.config import settings

    monkeypatch.setitem(settings.__dict__, "admin_key", "secret-test-key")  # frozen dataclass: patch the instance
    monkeypatch.setattr(type(settings), "live_llm", property(lambda self: False))
    with TestClient(main.app) as client:
        assert client.post("/judge/calibrate", json={"user_key": "t"}).status_code == 403
        assert client.post("/judge/calibrate", json={"user_key": "t"},
                           headers={"X-Admin-Key": "wrong"}).status_code == 403
        ok = client.post("/judge/calibrate", json={"user_key": "t"}, headers={"X-Admin-Key": "secret-test-key"})
        assert ok.status_code == 400 and "API key" in ok.json()["detail"]  # past the owner check
