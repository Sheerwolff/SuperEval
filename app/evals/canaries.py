"""Judge canaries: analyst answers whose correct verdict is known by construction, re-judged on a
schedule. If the judge's accuracy on them drops, the judge changed (provider model update, rubric
edit) even though no production traffic did. Layer 2's main detector.

Kinds (each built from real queries on the snapshot):
  good             matched-product comparison, numbers copied right            -> grounded, sound, answers
  simpsons_trap    AVG(price) over each chain's own products                    -> method fails
  miscopied        right query, a number changed in the answer                   -> not grounded
  wrong_city       question about one city, query filtered to another           -> method fails
  wrong_denominator promo "share" computed over promotions, not products sold   -> method fails
  good_decline     an unanswerable question, declined                           -> answers
  invented_trend   an unanswerable question answered with a made-up trend       -> not grounded, doesn't answer

    python -m app.evals.canaries --build
    python -m app.evals.canaries --run [--sim]
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from app.analyst.semantic import get_semantic
from app.config import settings
from app.db import JudgeCanaryResult, SessionLocal, utcnow
from app.evals.analyst_judge import RUBRIC_HASH, judge_live, judge_sim

CANARIES = Path(__file__).parent / "judge_canaries.jsonl"
CITIES = ["Tel Aviv", "Haifa", "Jerusalem"]


def _q(sem, qid: str, sql: str) -> dict:
    r = sem.run(sql, qid).to_dict()
    r["rows"] = r["rows"][:12]
    return r


def _ans(headline, claims, method="", declined=False, assumptions=(), caveats=()):
    return {"headline": headline, "claims": claims, "method": method, "assumptions": list(assumptions),
            "caveats": list(caveats), "chart": None, "declined": declined}


def build() -> list[dict]:
    sem = get_semantic()
    items = []

    def add(kind, question, queries, answer, expected):
        items.append({"id": f"c{len(items):02d}", "kind": kind, "question": question, "queries": queries,
                      "answer": answer, "expected": expected})

    for city in CITIES:
        q = _q(sem, "q1", f"SELECT * FROM chain_price_index('{city}', NULL)")
        top, last = q["rows"][0], q["rows"][-1]
        add("good", f"Which chain is cheapest in {city}?", [q],
            _ans(f"{top[0]} is the cheapest chain in {city}; {last[0]} the priciest.",
                 [{"text": f"{top[0]} is {top[3]}% vs. the median", "value": top[3], "unit": "%", "query_id": "q1"},
                  {"text": f"{last[0]} is {last[3]}% vs. the median", "value": last[3], "unit": "%", "query_id": "q1"}],
                 "Chain price index over products sold by 3+ chains."),
            {"grounded": True, "method": True, "answers": True})
        naive = _q(sem, "q1", f"SELECT chain, ROUND(AVG(price), 2) AS avg_price FROM prices WHERE city = '{city}' "
                              "GROUP BY chain ORDER BY avg_price")
        add("simpsons_trap", f"Which chain is cheapest in {city}?", [naive],
            _ans(f"{naive['rows'][0][0]} is the cheapest chain in {city}.",
                 [{"text": "average price", "value": naive["rows"][0][1], "unit": "₪", "query_id": "q1"}],
                 "Average shelf price per chain."),
            {"method": False})
        add("miscopied", f"How much below the median is the cheapest chain in {city}?", [q],
            _ans(f"{top[0]} is {abs(top[3]) + 4.1:.1f}% below the median in {city}.",
                 [{"text": "below median", "value": round(top[3] - 4.1, 1), "unit": "%", "query_id": "q1"}],
                 "Chain price index."),
            {"grounded": False})
    other = _q(sem, "q1", "SELECT * FROM chain_price_index('Tel Aviv', NULL)")
    add("wrong_city", "Which chain is cheapest in Haifa?", [other],
        _ans(f"{other['rows'][0][0]} is the cheapest chain in Haifa.",
             [{"text": "index", "value": other["rows"][0][2], "unit": "index", "query_id": "q1"}],
             "Chain price index."),
        {"method": False})
    for city in ["Haifa", "Jerusalem"]:
        wrong = _q(sem, "q1", f"SELECT chain, COUNT(*) AS n_promos, ROUND(COUNT(*) / SUM(COUNT(*)) OVER (), 4) "
                              f"AS promo_share FROM promos WHERE city = '{city}' GROUP BY chain ORDER BY promo_share DESC")
        add("wrong_denominator", f"Which chain has the highest share of its products on promotion in {city}?",
            [wrong], _ans(f"{wrong['rows'][0][0]} has the highest share of products on promotion in {city}.",
                          [{"text": "share", "value": wrong["rows"][0][2], "unit": "share", "query_id": "q1"}],
                          "Share of promotions."),
            {"method": False})
    for qtext in ["How did milk prices change over the last year?", "What are Shufersal's online delivery prices?"]:
        add("good_decline", qtext, [],
            _ans("The data can't answer this: it's today's snapshot of shelf prices only.", [], declined=True),
            {"answers": True})
        add("invented_trend", qtext, [],
            _ans("Prices rose about 6% over the period, led by dairy.",
                 [{"text": "rise", "value": 6.0, "unit": "%", "query_id": "q1"}], "Estimate."),
            {"grounded": False, "answers": False})
    return items


def load() -> list[dict]:
    return [json.loads(line) for line in CANARIES.read_text(encoding="utf-8").splitlines() if line.strip()]


def _deterministic(item: dict):
    from app.evals.online import run_deterministic

    t = SimpleNamespace(id=item["id"], question=item["question"], analysis=item["answer"],
                        tool_calls=item["queries"], error=None, language="en", latency_ms=1000)
    return run_deterministic(t)


def run(live: bool | None = None, at: datetime | None = None, leniency: float = 0.0,
        judge_model: str = "sim-judge", session=None) -> dict:
    live = settings.live_llm if live is None else live
    at = at or utcnow()
    run_id = str(uuid.uuid4())
    rows = []
    for item in load():
        if live:
            v, meta = judge_live(item["question"], item["queries"], item["answer"])
        else:
            v, meta = judge_sim(f"{item['id']}:{run_id}", _deterministic(item), leniency, judge_model)
        got = {"grounded": v.grounded, "method": v.method_validity >= 4, "answers": v.answers_question >= 4}
        for dim, expected in item["expected"].items():
            rows.append(JudgeCanaryResult(created_at=at, run_id=run_id, canary_id=item["id"], kind=item["kind"],
                                          dimension=dim, expected=expected, got=got[dim],
                                          correct=got[dim] == expected, judge_model=meta["judge_model"],
                                          reasoning=v.reasoning))
    own = session is None
    session = session or SessionLocal()
    try:
        session.add_all(rows)
        session.commit()
    finally:
        if own:
            session.close()
    acc = sum(r.correct for r in rows) / len(rows)
    return {"run_id": run_id, "checks": len(rows), "accuracy": round(acc, 3), "rubric_hash": RUBRIC_HASH,
            "judge_model": rows[0].judge_model if rows else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--sim", action="store_true")
    args = ap.parse_args()
    if args.build:
        items = build()
        CANARIES.write_text("\n".join(json.dumps(i, ensure_ascii=False, default=str) for i in items) + "\n",
                            encoding="utf-8")
        print(f"wrote {len(items)} canaries")
    if args.run:
        from app.db import init_db

        init_db()
        print(run(live=False if args.sim else None))


if __name__ == "__main__":
    main()
