"""Ground-truth evidence for the LLM judge: database records retrieved independently of the agent, so the judge can
check an answer against the truth instead of only against the agent's own queries.

  golden      the question matches an approved golden question: its reviewed reference SQL, run on today's data
  canonical   a chain-price comparison: the canonical chain price index for the city (the semantic layer's metric)
  none        nothing reliable to retrieve: the judge uses the plain rubric

The golden part comes from the evidence set saved by the latest calibration on the Judge Validation page
(app/evals/judge_validation.py), so 'Calibrate' is what updates what the judge is fed.
"""
from __future__ import annotations

import re
import time
from difflib import SequenceMatcher

from app.agent.textutil import CHAIN_HE, CITY_HE

MATCH_RATIO = 0.9
CHAIN_WORDS = re.compile(r"\bchains?\b|supermarket|רשת|רשתות|סופר", re.I)
COMPARE_WORDS = re.compile(r"cheap|expens|pric(ier|ey)|compare|vs\.?|versus|than|זול|יקר|השוו", re.I)
_cache: dict = {"at": 0.0, "items": None}


def _norm(q: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", q.lower()).split())


def evidence_set(max_age: float = 60.0) -> list[dict]:
    """The approved golden questions saved by the latest calibration (cached briefly for the live judge)."""
    if _cache["items"] is not None and time.time() - _cache["at"] < max_age:
        return _cache["items"]
    from sqlalchemy import select

    from app.db import JudgeCalibration, SessionLocal

    with SessionLocal() as s:
        latest = s.execute(select(JudgeCalibration).order_by(JudgeCalibration.created_at.desc())).scalars().first()
        items = list(latest.evidence) if latest else []
    _cache.update(at=time.time(), items=items)
    return items


def match_golden(question: str, items: list[dict]) -> dict | None:
    q = _norm(question)
    best, score = None, 0.0
    for it in items:
        r = SequenceMatcher(None, q, _norm(it["question"])).ratio()
        if r > score:
            best, score = it, r
    return best if score >= MATCH_RATIO else None


def _city(question: str, cities: list[str]) -> str | None:
    low = question.lower()
    return next((c for c in cities if c.lower() in low or (CITY_HE.get(c) and CITY_HE[c] in question)), None)


def _result(sem, sql: str, qid: str) -> dict:
    r = sem.run(sql, qid).to_dict()
    return {"sql": sql, "columns": r.get("columns"), "rows": (r.get("rows") or [])[:15],
            "row_count": r.get("row_count"), "error": r.get("error")}


def retrieve(question: str, items: list[dict] | None = None) -> dict | None:
    """Evidence for one question, or None when nothing reliable can be retrieved."""
    from app.analyst.semantic import get_semantic, sql_str

    sem = get_semantic()
    items = evidence_set() if items is None else items
    g = match_golden(question, items)
    if g:
        if not g.get("reference_sql"):  # decline / assumption questions have no reference query
            expected = {"decline": "the data can't answer this question; a correct answer declines",
                        "assumption": "the question is ambiguous; a correct answer states the reading it chose"}
            return {"source": f"golden {g['id']} (reviewed)", "expected": expected.get(g.get("answer_type"), "")}
        return {"source": f"golden {g['id']} (reviewed reference query, today's data)",
                "queries": [_result(sem, g["reference_sql"], "ref")]}
    names_chain = CHAIN_WORDS.search(question) or any(c.lower() in question.lower() or he in question
                                                      for c, he in CHAIN_HE.items())
    if names_chain and COMPARE_WORDS.search(question):
        city = _city(question, sem.cities)
        sql = f"SELECT * FROM chain_price_index({sql_str(city)}, NULL)"
        return {"source": f"canonical chain price index ({city or 'all cities'})", "queries": [_result(sem, sql, "ref")]}
    return None
