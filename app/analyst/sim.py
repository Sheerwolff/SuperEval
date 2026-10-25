"""Deterministic stand-in for the analyst: the 30-day backfill, synthetic bursts, CI, and the demo
when there's no API key or the daily budget is spent.

It runs *real* SQL on the real snapshot for each question's intent and writes a structured answer,
then injects each prompt version's known failure modes at realistic rates:
  normal              matched-product method; rare slips (mis-copied number, missing assumption)
Fault knobs (seeded incidents and the Live Monitor's "inject a fault" buttons):
  degraded_method     a model regression: AVG(price) over each chain's own products (the Simpson's
                      trap), no assumptions, sometimes answers unanswerable questions. Every number is
                      still real and grounded, which is why grounding alone can't catch it
  schema_broken       every query fails (a renamed table)
  latency_multiplier  slow upstream LLM
Traces are tagged model="sim".
"""
from __future__ import annotations

import random
import time
import zlib

from app.agent.textutil import CHAIN_HE, CITY_HE, detect_language
from app.analyst.agent import AnalystResult, load_versions
from app.analyst.answer import render_markdown
from app.analyst.semantic import QueryResult, get_semantic, sql_str


def _city(question: str, cities: list[str]) -> str | None:
    low = question.lower()
    for c in cities:
        if c.lower() in low or (CITY_HE.get(c) and CITY_HE[c] in question):
            return c
    return None


def intent_from_text(question: str) -> dict:
    """Rough intent for free text typed in the demo (synthetic traffic passes its intent directly)."""
    from app.analyst.topics import classify

    sem = get_semantic()
    kind = classify(question, sem.cities)
    city = _city(question, sem.cities)
    cat = next((c for c in sem.categories if c.split(" ")[0] in question.lower()), None)
    intent = {"kind": kind, "city": city, "category": cat, "lang": detect_language(question)}
    if kind == "geo":
        others = [c for c in sem.cities if c != city and c.lower() in question.lower()]
        intent["other"] = others[0] if others else None
    return intent


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)


class Sim:
    def __init__(self, version: str, rng: random.Random, schema_broken: bool = False, degraded_method: bool = False):
        self.v = version
        self.bad = degraded_method
        self.rng = rng
        self.sem = get_semantic()
        self.broken = schema_broken
        self.queries: list[dict] = []

    def q(self, sql: str) -> QueryResult:
        qid = f"q{len(self.queries) + 1}"
        if self.broken:
            r = QueryResult(id=qid, sql=sql, error="CatalogException: Catalog Error: Table with name prices does not "
                                                   "exist! (schema migration renamed it to price_rows)")
        else:
            r = self.sem.run(sql, qid)
        self.queries.append(r.to_dict())
        return r

    def answer(self, intent: dict, question: str) -> dict | None:
        he = intent.get("lang") == "he"
        kind, city, cat = intent["kind"], intent.get("city"), intent.get("category")
        cname = (CITY_HE.get(city, city) if he else city) if city else None
        base = {"claims": [], "method": "", "assumptions": [], "caveats": [], "chart": None, "declined": False}
        if kind in ("unanswerable", "injection"):
            if self.bad and kind == "unanswerable" and self.rng.random() < 0.5:
                return {**base, "headline": "Prices rose by about 4% over the period.", "method": "Estimated."}
            return {**base, "declined": True,
                    "headline": ("הנתונים לא מאפשרים לענות על זה: יש רק תמונת מצב של המחירים היום, בלי היסטוריה, "
                                 "מחירי אונליין או שעות פתיחה." if he else
                                 "The data can't answer this: it's today's snapshot of shelf prices only, with no "
                                 "history, online prices, opening hours or other topics.")}
        if self.broken:
            for _ in range(2):  # the agent retries, then gives up
                self.q(f"SELECT * FROM prices WHERE city = '{city or 'Tel Aviv'}' LIMIT 5")
            return None
        if kind in ("comparison", "ambiguous"):
            c, k = sql_str(city), sql_str(cat)
            if self.bad:
                where = " AND ".join(x for x in [f"city = {c}" if city else "", f"category = {k}" if cat else ""] if x)
                r = self.q("SELECT chain, ROUND(AVG(price), 2) AS avg_price FROM prices JOIN products USING (item_code)"
                           + (f" WHERE {where}" if where else "") + " GROUP BY chain ORDER BY avg_price")
                col, unit = "avg_price", "₪"
            else:
                r = self.q(f"SELECT * FROM chain_price_index({c}, {k})")
                col, unit = "pct_vs_median", "%"
            if r.error or not r.rows:
                return None
            df = r.frame()
            top, last = df.iloc[0], df.iloc[-1]
            tv = float(top[col])
            if not self.bad and self.rng.random() < 0.02:
                tv = round(tv + 1.37, 2)  # rare slip: mis-copied number
            where_txt = f" ב{cname}" if he and cname else (f" in {cname}" if cname else "")
            head = (f"{CHAIN_HE.get(top.chain, top.chain)} הכי זולה{where_txt}; {CHAIN_HE.get(last.chain, last.chain)} "
                    f"הכי יקרה." if he else f"{top.chain} is the cheapest chain{where_txt}; {last.chain} the priciest.")
            ans = {**base, "headline": head,
                   "claims": [{"text": f"{top.chain}: {_fmt(tv)}{unit}", "value": tv, "unit": unit, "query_id": r.id},
                              {"text": f"{last.chain}: {_fmt(float(last[col]))}{unit}", "value": float(last[col]),
                               "unit": unit, "query_id": r.id}],
                   "method": ("Average shelf price per chain." if self.bad else
                              "Chain price index: each chain vs. the median price of the same products."),
                   "chart": {"type": "bar", "query_id": r.id, "x": "chain", "y": col, "series": None,
                             "title": "Chains compared"}}
            if kind == "ambiguous" and not self.bad and self.rng.random() > 0.03:
                ans["assumptions"] = ["כל הערים יחד, כל הקטגוריות" if he else "All covered cities and categories."]
            if "n_products" in df and df.n_products.min() < 10:
                ans["caveats"] = ["Small sample for some chains."]
            return ans
        if kind == "lookup":
            code = intent.get("item_code")
            if not code:
                return None
            r = self.q(f"SELECT chain, MIN(price) AS price FROM prices WHERE item_code = '{code}' AND city = "
                       f"{sql_str(city)} GROUP BY chain ORDER BY price, chain")
            if r.error or not r.rows:
                return {**base, "headline": "No store in that city carries it in today's data.",
                        "caveats": ["Snapshot coverage only."]}
            top = r.rows[0]
            return {**base, "headline": (f"הכי זול: ₪{top[1]:.2f} ב{CHAIN_HE.get(top[0], top[0])}." if he
                                         else f"Cheapest: ₪{top[1]:.2f} at {top[0]}."),
                    "claims": [{"text": f"{top[0]} ₪{top[1]:.2f}", "value": float(top[1]), "unit": "₪",
                                "query_id": r.id}],
                    "method": "Cheapest store price per chain in the city."}
        if kind == "spread":
            r = self.q(f"SELECT * FROM price_spread({sql_str(city)}, NULL) LIMIT 10")
            if r.error or not r.rows:
                return None
            t = r.frame().iloc[0]
            return {**base, "headline": (f"הפער הגדול ביותר: {t.name_en}, {t.spread_pct}% בין הרשתות." if he else
                                         f"The biggest gap is {t.name_en}: {t.spread_pct}% between chains."),
                    "claims": [{"text": "gap", "value": float(t.spread_pct), "unit": "%", "query_id": r.id}],
                    "method": "Cheapest store per chain, then max vs. min.",
                    "chart": {"type": "horizontal_bar", "query_id": r.id, "x": "name_en", "y": "spread_pct",
                              "series": None, "title": "Largest gaps (%)"}}
        if kind == "promo":
            r = self.q(f"SELECT * FROM promo_share({sql_str(city)})")
            if r.error or not r.rows:
                return None
            t = r.frame().iloc[0]
            return {**base, "headline": (f"ל{CHAIN_HE.get(t.chain, t.chain)} יש הכי הרבה מוצרים במבצע." if he else
                                         f"{t.chain} has the highest share of products on promotion."),
                    "claims": [{"text": "share", "value": float(t.promo_share), "unit": "share", "query_id": r.id}],
                    "method": "Share of each chain's products with an active promotion.",
                    "chart": {"type": "bar", "query_id": r.id, "x": "chain", "y": "promo_share", "series": None,
                              "title": "Share on promotion"}}
        if kind == "geo":
            other = intent.get("other") or next(c for c in self.sem.cities if c != city)
            k = f" AND category = {sql_str(cat)}" if cat else ""
            if self.bad:
                r = self.q(f"SELECT city, ROUND(AVG(price), 2) AS avg_price FROM prices JOIN products USING (item_code) "
                           f"WHERE city IN ({sql_str(city)}, {sql_str(other)}){k} GROUP BY city ORDER BY avg_price")
            else:
                r = self.q(f"""WITH p AS (SELECT city, item_code, MIN(price) AS price FROM prices JOIN products
                    USING (item_code) WHERE city IN ({sql_str(city)}, {sql_str(other)}){k} GROUP BY city, item_code),
                    shared AS (SELECT item_code FROM p GROUP BY item_code HAVING COUNT(*) = 2)
                    SELECT city, ROUND(AVG(price), 2) AS avg_price FROM p JOIN shared USING (item_code)
                    GROUP BY city ORDER BY avg_price""")
            if r.error or not r.rows:
                return None
            t = r.rows[0]
            return {**base, "headline": f"{t[0]} is cheaper.", "method": "Matched products in both cities."
                    if not self.bad else "Average price per city.",
                    "claims": [{"text": f"{t[0]} ₪{t[1]}", "value": float(t[1]), "unit": "₪", "query_id": r.id}]}
        return None


def run_sim(question: str, version: str, intent: dict | None = None, seed: int | None = None,
            latency_multiplier: float = 1.0, schema_broken: bool = False,
            degraded_method: bool = False) -> AnalystResult:
    versions = load_versions()
    v = versions.get(version) or versions[next(iter(versions))]
    rng = random.Random(seed if seed is not None else zlib.crc32(question.encode()))
    intent = intent or intent_from_text(question)
    lang = intent.get("lang") or detect_language(question)
    res = AnalystResult(question=question, version=version, model="sim", prompt_hash=v.prompt_hash, language=lang)
    start = time.perf_counter()
    sim = Sim(version, rng, schema_broken, degraded_method)
    if intent["kind"] not in ("comparison", "ambiguous", "lookup", "spread", "promo", "geo", "unanswerable",
                              "injection"):
        intent["kind"] = "comparison"
    ans = sim.answer(intent, question)
    if ans is None and intent["kind"] == "lookup" and not intent.get("item_code"):
        ans = {"headline": ("מצב הדגמה: שאלות על מוצר ספציפי זמינות רק עם הסוכן החי." if lang == "he" else
                            "Demo mode: questions about a specific product need the live analyst."),
               "claims": [], "method": "", "assumptions": [], "caveats": [], "chart": None, "declined": True}
    res.queries = sim.queries
    res.answer = ans
    if ans is None:
        res.error = "no_answer: " + ("every query failed" if schema_broken else "nothing to report")
    res.markdown = render_markdown(ans, he=lang == "he") if ans else "Sorry, I couldn't complete that analysis."
    res.steps = len(sim.queries) + 1
    res.input_tokens = 3200 + 900 * len(sim.queries) + rng.randint(0, 400)
    res.output_tokens = 300 + rng.randint(0, 250)
    res.latency_ms = round((5500 + 1800 * len(sim.queries)) * latency_multiplier * rng.lognormvariate(0, 0.22)
                           + (time.perf_counter() - start) * 1000, 1)
    return res


def sim_analysis(question: str, version: str, session_id: str | None = None, on_event=None, source: str = "user",
                 intent: dict | None = None, seed: int | None = None, fault: str | None = None):
    from app.analyst.pipeline import evaluate, to_trace
    from app.analyst.topics import classify
    from app.data.catalog import get_catalog
    from app.db import SessionLocal

    res = run_sim(question, version, intent=intent, seed=seed, schema_broken=fault == "schema_broken",
                  degraded_method=fault == "degraded_method",
                  latency_multiplier=3.0 if fault == "slow_llm" else 1.0)
    if on_event:
        for q in res.queries:
            on_event({"type": "tool_start", "name": "run_sql", "args": {"sql": q["sql"]}, "id": q["id"]})
            on_event({"type": "tool_end", "name": "run_sql", "id": q["id"], "args": {"sql": q["sql"]},
                      "summary": q["error"] or f"{q['row_count']} rows"})
    trace = to_trace(res, source, session_id, snapshot_date=get_catalog().snapshot_date,
                     topic=classify(question, get_semantic().cities))
    with SessionLocal() as s:
        s.add(trace)
        s.flush()
        evals = evaluate(trace, judge=None, force_sim=True, leniency=0.85 if fault == "lenient_judge" else 0.0,
                         judge_model="sim-judge (updated)" if fault == "lenient_judge" else "sim-judge")
        for e in evals:
            e.trace_id = trace.id
        s.add_all(evals)
        s.commit()
    return trace, evals

