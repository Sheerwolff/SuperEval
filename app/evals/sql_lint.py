"""SQL method lint: deterministic checks for analysis mistakes that produce plausible numbers.

Rules (applied to the queries an answer cites):
  unmatched_comparison  chains/cities compared with AVG/SUM/MEDIAN(price) over whatever each sells,
                        without matching products (the Simpson's trap) and without a canonical macro
  missing_city_filter   the question names a city but no cited query restricts to it
  small_sample          a cited result reports n < 10 products/stores and the answer gives no caveat
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.textutil import CITY_HE

CANONICAL = re.compile(r"\b(chain_price_index|price_spread|promo_share)\s*\(", re.I)
AGG_PRICE = re.compile(r"\b(avg|mean|sum|median)\s*\(\s*(\w+\.)?price\b", re.I)
GROUP_BY_ENTITY = re.compile(r"group\s+by[^;]*\b(chain|city)\b", re.I)
MATCHING = re.compile(r"item_code\s*(=|in\b)|having\s+count|join\s*\(?\s*select[^)]*item_code|"
                      r"\bshared\b|\bmatched\b|\bcommon\b|\bboth\b", re.I)
N_COLUMNS = ("n", "n_products", "n_stores", "count", "n_items")


@dataclass
class LintFinding:
    rule: str
    query_id: str | None
    message: str


def city_in_question(question: str, cities: list[str]) -> str | None:
    low = question.lower()
    for c in cities:
        if c.lower() in low or (CITY_HE.get(c) and CITY_HE[c] in question):
            return c
    return None


def lint(question: str, analysis: dict, queries: list[dict], cities: list[str]) -> list[LintFinding]:
    cited_ids = {c["query_id"] for c in analysis.get("claims", [])}
    if analysis.get("chart"):
        cited_ids.add(analysis["chart"]["query_id"])
    cited = [q for q in queries if q["id"] in cited_ids and not q.get("error")] or \
            [q for q in queries if not q.get("error")]
    out: list[LintFinding] = []
    for q in cited:
        sql = q["sql"]
        if (AGG_PRICE.search(sql) and GROUP_BY_ENTITY.search(sql) and not CANONICAL.search(sql)
                and not MATCHING.search(sql)):
            out.append(LintFinding("unmatched_comparison", q["id"],
                                   f"{q['id']} compares an average price across chains/cities without matching "
                                   "products: each side averages a different basket."))
    city = city_in_question(question, cities)
    if city and cited and not any(city.lower() in q["sql"].lower() for q in cited):
        out.append(LintFinding("missing_city_filter", None,
                               f"The question is about {city}, but no cited query filters to it."))
    if not analysis.get("caveats"):
        for q in cited:
            cols = q.get("columns") or []
            for col in N_COLUMNS:
                if col in cols:
                    small = [r[cols.index(col)] for r in q["rows"]
                             if isinstance(r[cols.index(col)], (int, float)) and r[cols.index(col)] < 10]
                    if small:
                        out.append(LintFinding("small_sample", q["id"],
                                               f"{q['id']} has {col} < 10 ({small[0]}) and the answer gives no caveat."))
                        break
    return out
