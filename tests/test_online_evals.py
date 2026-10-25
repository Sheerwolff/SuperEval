from types import SimpleNamespace as NS

from app.analyst.semantic import get_semantic
from app.evals import golden, scoring
from app.evals.online import (
    assumption_check,
    canonical_check,
    chart_lint,
    decline_check,
    grounding,
    method_lint,
    sql_health,
)


def trace(question, analysis, queries, language="en"):
    return NS(question=question, analysis=analysis, tool_calls=queries, error=None, language=language,
              latency_ms=1000)


def ans(headline, claims=(), chart=None, assumptions=(), caveats=(), declined=False):
    return {"headline": headline, "claims": list(claims), "method": "m", "assumptions": list(assumptions),
            "caveats": list(caveats), "chart": chart, "declined": declined}


def claim(value, qid="q1"):
    return {"text": "t", "value": value, "unit": "₪", "query_id": qid}


NAIVE = {"id": "q1", "sql": "SELECT chain, AVG(price) AS a FROM prices WHERE city = 'Haifa' GROUP BY chain ORDER BY a",
         "columns": ["chain", "a"], "rows": [["Osher Ad", 16.37], ["Rami Levy", 16.83], ["Yellow", 17.38]],
         "row_count": 3, "error": None}


def test_grounding_catches_miscopied_numbers_and_ignores_product_sizes():
    q = "Which chain is cheapest in Haifa?"
    assert grounding(trace(q, ans("Osher Ad is cheapest.", [claim(16.37)]), [NAIVE])).passed
    bad = grounding(trace(q, ans("Osher Ad is cheapest.", [claim(15.99)]), [NAIVE]))
    assert bad.passed is False and bad.details["ungrounded"]
    assert grounding(trace(q, ans("Osher Ad at ₪99 is cheapest.", [claim(16.37)]), [NAIVE])).passed is False
    assert grounding(trace(q, ans("Water 1.5L is cheapest at Osher Ad.", [claim(16.37)]), [NAIVE])).passed
    assert grounding(trace(q, ans("7.8% below", [claim(0.078)]), [{**NAIVE, "rows": [["x", 0.078]]}])).passed


def test_method_lint_flags_the_simpsons_trap_but_not_canonical_or_matched_sql():
    q = "Which chain is cheapest in Haifa?"
    assert method_lint(trace(q, ans("x", [claim(16.37)]), [NAIVE])).passed is False
    canon = {**NAIVE, "sql": "SELECT * FROM chain_price_index('Haifa', NULL)"}
    assert method_lint(trace(q, ans("x", [claim(16.37)]), [canon])).passed
    matched = {**NAIVE, "sql": "SELECT chain, AVG(price) FROM prices WHERE city='Haifa' AND item_code IN "
                               "(SELECT item_code FROM prices GROUP BY item_code HAVING COUNT(DISTINCT chain)=6) "
                               "GROUP BY chain"}
    assert method_lint(trace(q, ans("x", [claim(16.37)]), [matched])).passed


def test_method_lint_missing_city_and_small_sample():
    q = "Which chain is cheapest in Haifa?"
    nocity = {**NAIVE, "sql": "SELECT * FROM chain_price_index(NULL, NULL)"}
    assert method_lint(trace(q, ans("x", [claim(16.37)]), [nocity])).passed is False
    small = {"id": "q1", "sql": "SELECT * FROM chain_price_index('Haifa', 'pet')", "columns": ["chain", "n_products"],
             "rows": [["Rami Levy", 4]], "row_count": 1, "error": None}
    assert method_lint(trace(q, ans("x", [claim(4)]), [small])).passed is False
    assert method_lint(trace(q, ans("x", [claim(4)], caveats=["only 4 products"]), [small])).passed


def test_canonical_check_compares_with_the_matched_index():
    sem = get_semantic()
    city = sem.cities[0]
    top = sem.run(f"SELECT chain FROM chain_price_index('{city}', NULL)", "q").rows[0][0]
    q = f"Which chain is cheapest in {city}?"
    assert canonical_check(trace(q, ans(f"{top} is cheapest."), [NAIVE])).passed
    last = sem.run(f"SELECT chain FROM chain_price_index('{city}', NULL)", "q").rows[-1][0]
    assert canonical_check(trace(q, ans(f"{last} is cheapest."), [NAIVE])).passed is False


def test_chart_lint_decline_and_assumptions():
    good = {"type": "bar", "query_id": "q1", "x": "chain", "y": "a", "series": None, "title": "t"}
    assert chart_lint(trace("q", ans("x", chart=good), [NAIVE])).passed
    assert chart_lint(trace("q", ans("x", chart={**good, "y": "nope"}), [NAIVE])).passed is False
    assert chart_lint(trace("q", ans("x", chart={**good, "type": "line"}), [NAIVE])).passed is False
    assert decline_check(trace("How did milk prices change over the last year?", ans("No.", declined=True), [])).passed
    assert decline_check(trace("How did milk prices change over the last year?", ans("Up 5%."), [])).passed is False
    assert decline_check(trace("Which chain is cheapest in Haifa?", ans("No.", declined=True), [])).passed is False
    assert assumption_check(trace("Which chain is cheapest?", ans("Rami Levy."), [])).passed is False
    assert assumption_check(trace("Which chain is cheapest?", ans("Rami Levy.", assumptions=["all cities"]), [])).passed
    assert assumption_check(trace("Which chain is cheapest in Haifa?", ans("Rami Levy."), [])).passed is None
    assert sql_health(trace("q", ans("x"), [{**NAIVE, "error": "Binder Error"}])).passed is False


def test_golden_references_and_scorers_selftest():
    res = golden.selftest(golden.load(("approved", "draft")))
    assert res["items"] >= 60 and not res["problems"], res["problems"][:3]


def test_ranking_scorer_accepts_ties():
    item = {"answer_type": "ranking", "reference_sql": "SELECT 'A' AS chain, 1.0 AS v UNION ALL SELECT 'B', 1.0 "
                                                       "UNION ALL SELECT 'C', 2.0", "answer_key":
            {"column": "chain", "value_column": "v"}}
    sem = get_semantic()
    assert scoring.score(item, ans("B is cheapest."), [], sem).passed
    assert scoring.score(item, ans("C is cheapest."), [], sem).passed is False
