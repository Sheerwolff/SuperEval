import uuid

import pytest

from app.db import EvalResult, GoldenItem, HumanJudgment, SessionLocal, Trace, init_db
from app.evals.human import agreement, judgments_frame, override_rate, promote


@pytest.fixture()
def traces():
    init_db()
    ids = []
    with SessionLocal() as s:
        for q, failed in [("Which chain is cheapest in Haifa?", True),
                              ("How many chains are covered in Haifa?", False)]:
            tid = str(uuid.uuid4())
            s.add(Trace(id=tid, agent_version="a1", model="m", prompt_hash="h", question=q, language="en",
                        answer="a", tool_calls=[{"id": "q1", "sql": "SELECT * FROM chain_price_index('Haifa', NULL)",
                                                 "columns": ["chain"], "rows": [["Rami Levy"]], "error": None}],
                        analysis={"headline": "a", "claims": [], "declined": False}))
            s.add(EvalResult(trace_id=tid, evaluator="method_lint", kind="deterministic", score=0.0 if failed else 1.0,
                             passed=not failed, details={}))
            ids.append(tid)
        s.commit()
    yield ids
    with SessionLocal() as s:
        s.query(HumanJudgment).filter(HumanJudgment.trace_id.in_(ids)).delete()
        s.query(GoldenItem).filter(GoldenItem.source_trace_id.in_(ids)).delete()
        s.query(EvalResult).filter(EvalResult.trace_id.in_(ids)).delete()
        s.query(Trace).filter(Trace.id.in_(ids)).delete()
        s.commit()


def test_override_rate_agreement_and_promotion(traces):
    bad, good = traces
    with SessionLocal() as s:
        s.add(HumanJudgment(trace_id=bad, role="owner", correct="no", method_ok=False, tags=["bad method"]))
        s.add(HumanJudgment(trace_id=good, role="owner", correct="yes", method_ok=True, tags=[]))
        s.add(HumanJudgment(trace_id=good, role="visitor", correct="yes", method_ok=None, tags=[]))
        s.commit()
        jid = s.query(HumanJudgment).filter_by(trace_id=bad, role="owner").one().id
    owner = judgments_frame("owner")
    owner = owner[owner.trace_id.isin(traces)]
    rate, n = override_rate(owner)
    assert n == 2 and rate == 0.5
    agr = agreement(owner)
    assert agr["both_flag"] == 1 and agr["neither"] == 1 and agr["eval_recall_on_human_flags"] == 1.0
    gid = promote(jid)
    with SessionLocal() as s:
        g = s.get(GoldenItem, gid)
        assert g.status == "draft" and g.source == "promoted" and g.source_trace_id == bad
        assert g.answer_type == "ranking" and "chain_price_index" in g.reference_sql
    assert promote(jid) == gid  # idempotent
    with SessionLocal() as s:
        vid = s.query(HumanJudgment).filter_by(trace_id=good, role="visitor").one().id
    with pytest.raises(ValueError):
        promote(vid)  # visitors can't promote
