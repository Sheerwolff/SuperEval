import uuid

import pytest

from app.db import EvalResult, GoldenItem, HumanJudgment, JudgeFeedback, SessionLocal, Trace, init_db
from app.evals.human import (
    agreement,
    draft_from_trace,
    judge_feedback_summary,
    judgments_frame,
    override_rate,
    promote,
    save_judge_feedback,
)


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
        s.query(JudgeFeedback).filter(JudgeFeedback.trace_id.in_(ids)).delete()
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


def test_judge_feedback(traces):
    bad, good = traces
    agree_id = save_judge_feedback(good, "visitor", "sess1", agree=True, disputed=["method_lint"], note="  ")
    disagree_id = save_judge_feedback(bad, "visitor", "sess1", agree=False, disputed=["method_lint"],
                                      note="the method looks fine to me")
    with SessionLocal() as s:
        ok, no = s.get(JudgeFeedback, agree_id), s.get(JudgeFeedback, disagree_id)
    assert ok.agree and ok.disputed == [] and ok.note is None  # agreeing disputes nothing
    assert not no.agree and no.disputed == ["method_lint"] and no.note == "the method looks fine to me"


def test_judge_feedback_summary(traces):
    bad, good = traces  # method_lint failed on `bad`, passed on `good`
    with SessionLocal() as s:
        since = s.query(Trace).filter(Trace.id == bad).one().created_at
    save_judge_feedback(good, "visitor", "a", agree=True)
    save_judge_feedback(good, "visitor", "b", agree=False, disputed=["method_lint"], note="lint missed it")
    save_judge_feedback(bad, "owner", None, agree=False, disputed=["method_lint", "judge_answers"])
    out = judge_feedback_summary(since=since)
    assert out["n"] == 3 and out["n_disagree"] == 2 and abs(out["agree_rate"] - 1 / 3) < 1e-9
    d = out["disputed"].set_index("evaluator")
    assert d.loc["method_lint"].tolist() == [1, 1, 2]  # disputed once as a pass (lenient), once as a fail (strict)
    assert d.loc["judge_answers"].tolist() == [0, 0, 1]  # no stored result for it: counted, but unsplit
    assert out["disputed"].evaluator.iloc[0] == "method_lint"
    assert out["recent"].note.tolist() == [None, "lint missed it"]  # newest first


def test_golden_drafts_from_traces_and_feedback(traces):
    bad, good = traces
    gid = draft_from_trace(good, "trace", "from the explorer")
    assert draft_from_trace(good, "trace", "again") == gid  # one item per trace
    with SessionLocal() as s:
        g = s.get(GoldenItem, gid)
        assert g.status == "draft" and g.source == "trace" and "chain_price_index" in g.reference_sql
    save_judge_feedback(bad, "visitor", "x", agree=True)
    with SessionLocal() as s:
        assert not s.query(GoldenItem).filter_by(source_trace_id=bad).count()  # agreeing adds nothing
    save_judge_feedback(bad, "visitor", "x", agree=False, disputed=["method_lint"], note="fine to me")
    with SessionLocal() as s:
        g = s.query(GoldenItem).filter_by(source_trace_id=bad).one()
        assert g.source == "feedback" and "method_lint" in g.notes and "fine to me" in g.notes


def test_approve_into_golden(traces):
    from app.evals.human import approve_into_golden

    bad, good = traces
    with SessionLocal() as s:
        ok = HumanJudgment(trace_id=good, role="owner", correct="yes", method_ok=True, tags=[])
        wrong = HumanJudgment(trace_id=bad, role="owner", correct="no", method_ok=False, tags=["bad method"])
        s.add_all([ok, wrong])
        s.commit()
        ok_id, wrong_id = ok.id, wrong.id
    assert "approved" in approve_into_golden(ok_id)  # judged correct: the agent's own SQL is a trusted reference
    assert "draft" in approve_into_golden(wrong_id)  # judged wrong, no SQL: can't trust the reference yet
    with SessionLocal() as s:
        fixed = HumanJudgment(trace_id=bad, role="owner", correct="no", method_ok=False, tags=["bad method"],
                              corrected_sql="SELECT * FROM chain_price_index('Haifa', NULL)")
        s.add(fixed)
        s.commit()
        fixed_id = fixed.id
    assert "approved" in approve_into_golden(fixed_id)  # same item, now with the owner's SQL
    with SessionLocal() as s:
        items = s.query(GoldenItem).filter(GoldenItem.source_trace_id.in_([bad, good])).all()
        assert len(items) == 2 and all(g.status == "approved" for g in items)
        assert s.get(HumanJudgment, fixed_id).promoted_to == s.query(GoldenItem).filter_by(source_trace_id=bad).one().id
