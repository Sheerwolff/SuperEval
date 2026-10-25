"""Deterministic stand-in for the analyst when there's no API key or the daily budget is spent.

It recognises a few question shapes (chain comparison, price spread, promotions), runs the
canonical SQL for them on the real data, and writes a structured answer. Anything else gets an
honest "demo mode" reply. Traces are tagged model="sim".
"""
from __future__ import annotations

import re
import time

from app.agent.textutil import CITY_HE, detect_language
from app.analyst.agent import AnalystResult, load_versions
from app.analyst.answer import render_markdown
from app.analyst.semantic import get_semantic


def _city(question: str, cities: list[str]) -> str | None:
    low = question.lower()
    for c in cities:
        if c.lower() in low or (CITY_HE.get(c) and CITY_HE[c] in question):
            return c
    return None


def plan(question: str) -> tuple[str, str] | None:
    """(kind, sql) for questions the simulator can answer, else None."""
    sem = get_semantic()
    city = _city(question, sem.cities)
    c = f"'{city}'" if city else "NULL"
    q = question.lower()
    if re.search(r"promo|deal|sale|מבצע", q):
        return "promo", f"SELECT * FROM promo_share({c})"
    if re.search(r"gap|spread|differ|פער", q):
        return "spread", f"SELECT * FROM price_spread({c}, NULL) LIMIT 10"
    if re.search(r"cheap|expens|zol|זול|יקר|compare|השוו", q):
        return "index", f"SELECT * FROM chain_price_index({c}, NULL)"
    return None


def run_sim(question: str, version: str) -> AnalystResult:
    v = load_versions()[version]
    he = detect_language(question) == "he"
    res = AnalystResult(question=question, version=version, model="sim", prompt_hash=v.prompt_hash,
                        language="he" if he else "en")
    start = time.perf_counter()
    p = plan(question)
    if p is None:
        res.answer = {
            "headline": ("מצב הדגמה: הסוכן החי אינו זמין כרגע, ומצב ההדגמה עונה רק על השוואות בין רשתות, פערי מחירים ומבצעים."
                         if he else "Demo mode: the live analyst is unavailable right now, and demo mode only answers "
                                    "chain comparisons, price gaps and promotion questions."),
            "claims": [], "method": "", "assumptions": [], "caveats": [], "chart": None, "declined": True}
    else:
        kind, sql = p
        q = get_semantic().run(sql, "q1")
        res.queries.append(q.to_dict())
        df = q.frame()
        city = _city(question, get_semantic().cities)
        where = f" in {city}" if city else ""
        if kind == "index" and len(df):
            top, last = df.iloc[0], df.iloc[-1]
            res.answer = {
                "headline": f"{top.chain} is the cheapest chain{where} on matched products; {last.chain} the priciest.",
                "claims": [
                    {"text": f"{top.chain} is {abs(top.pct_vs_median)}% below the median.", "value": float(top.pct_vs_median),
                     "unit": "%", "query_id": "q1"},
                    {"text": f"{last.chain} is {last.pct_vs_median}% above the median.", "value": float(last.pct_vs_median),
                     "unit": "%", "query_id": "q1"}],
                "method": "Chain price index: each chain's price for products sold by 3+ chains, relative to the median.",
                "assumptions": [] if city else ["All covered cities combined."],
                "caveats": ["Snapshot of today's published prices."],
                "chart": {"type": "bar", "query_id": "q1", "x": "chain", "y": "pct_vs_median", "series": None,
                          "title": f"% vs. median price{where}"},
                "declined": False}
        elif kind == "spread" and len(df):
            top = df.iloc[0]
            res.answer = {
                "headline": f"The biggest gap{where} is {top.name_en}: {top.spread_pct}% between chains.",
                "claims": [{"text": f"{top.name_en} costs ₪{top.min_price} at {top.cheapest_chain}.",
                            "value": float(top.min_price), "unit": "₪", "query_id": "q1"},
                           {"text": f"…and ₪{top.max_price} at {top.priciest_chain}.", "value": float(top.max_price),
                            "unit": "₪", "query_id": "q1"}],
                "method": "Cheapest store per chain for each product, then max/min across chains.",
                "assumptions": [] if city else ["All covered cities combined."], "caveats": [],
                "chart": {"type": "horizontal_bar", "query_id": "q1", "x": "name_en", "y": "spread_pct", "series": None,
                          "title": f"Largest price gaps{where} (%)"},
                "declined": False}
        elif kind == "promo" and len(df):
            top = df.iloc[0]
            res.answer = {
                "headline": f"{top.chain} has the highest share of products on promotion{where}.",
                "claims": [{"text": f"{top.chain}: {top.promo_share:.0%} of products on promotion.",
                            "value": float(top.promo_share), "unit": "share", "query_id": "q1"}],
                "method": "Share of each chain's products with an active promotion.",
                "assumptions": [] if city else ["All covered cities combined."], "caveats": [],
                "chart": {"type": "bar", "query_id": "q1", "x": "chain", "y": "promo_share", "series": None,
                          "title": f"Share of products on promotion{where}"},
                "declined": False}
        else:
            res.error = "no_answer: empty result"
    res.markdown = render_markdown(res.answer, he=he) if res.answer else "No answer."
    res.steps = 2 if res.queries else 1
    res.input_tokens, res.output_tokens = 3500, 400
    res.latency_ms = (time.perf_counter() - start) * 1000 + 6000
    return res


def sim_analysis(question: str, version: str, session_id: str | None = None, on_event=None,
                 source: str = "user"):
    from app.agent.textutil import classify_topic
    from app.analyst.pipeline import evaluate, to_trace
    from app.data.catalog import get_catalog
    from app.db import SessionLocal

    res = run_sim(question, version)
    if on_event:
        for q in res.queries:
            on_event({"type": "tool_start", "name": "run_sql", "args": {"sql": q["sql"]}, "id": q["id"]})
            on_event({"type": "tool_end", "name": "run_sql", "id": q["id"], "args": {"sql": q["sql"]},
                      "summary": q["error"] or f"{q['row_count']} rows"})
    trace = to_trace(res, source, session_id, snapshot_date=get_catalog().snapshot_date,
                     topic=classify_topic(question))
    with SessionLocal() as s:
        s.add(trace)
        s.flush()
        evals = evaluate(trace)
        for e in evals:
            e.trace_id = trace.id
        s.add_all(evals)
        s.commit()
    return trace, evals
