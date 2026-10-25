"""Layer 1 online evaluators for the analyst. Production has no reference answers, so each signal
targets one failure mode and needs no ground truth:

  completion          a structured answer was produced (else: failure type)
  grounding           every claim's value is in the result of the query it cites; no uncited headline numbers
  sql_health          the queries the answer rests on ran without errors
  method_lint         no known analysis pitfalls in the cited SQL (sql_lint.py)
  canonical_check     for chain-comparison questions, the named winner matches the canonical index
  chart_lint          the chart spec points at real columns of a cited query, with a fitting chart type
  decline_check       unanswerable / out-of-scope questions are declined, answerable ones aren't
  assumption_check    ambiguous questions state the reading chosen
  language_match      answer in the question's language
  latency_slo         under the latency SLO
"""
from __future__ import annotations

import re

from app.agent.textutil import CHAIN_HE, INJECTION_PATTERNS, detect_language
from app.analyst.semantic import get_semantic
from app.evals.deterministic import EvalOutcome
from app.evals.sql_lint import city_in_question, lint

LATENCY_SLO_MS = 20000
UNANSWERABLE = re.compile(
    r"last (year|month|week)|over time|trend|histor|chang(e|ed) (over|since|in the)|used to cost|yesterday|"
    r"online|delivery|deliver|open(ing)? hours|what time|close[sd]?\b|closing|phone number|"
    r"השתנ|לאורך זמן|בשנה האחרונה|בחודשים האחרונים|היסטורי|אונליין|משלוח|שעות פתיחה|נסגר|נפתח",
    re.I)
OFF_DOMAIN = re.compile(r"\b(poem|joke|weather|bitcoin|stock|python|homework|recipe)\b|בדיחה|שיר|מזג ה?אוויר", re.I)
UNCOVERED_PLACES = re.compile(r"\b(eilat|carrefour|nazareth|tiberias|umm al|super-?pharm|mega)\b|אילת|קרפור", re.I)
COMPARE = re.compile(r"cheap|expens|pric(ier|ey)|compare|vs\.?|versus|than|זול|יקר|השוו", re.I)
CATEGORY_HINT = re.compile(r"dairy|beverag|drink|snack|sweet|pantry|clean|bakery|bread|meat|fish|frozen|baby|"
                           r"alcohol|wine|beer|cereal|personal|pet|fruit|vegetable|sauce|canned|"
                           r"חלב|משקה|חטיף|ממתק|ניקיון|מאפ|בשר|דג|קפוא|תינוק|אלכוהול|יין|בירה", re.I)


def _num_close(a: float, b: float) -> bool:
    for scale in (1, 100, 0.01):
        x = a * scale
        if abs(x - b) <= max(0.005 * abs(b), 0.011):
            return True
    return False


def _cells(q: dict) -> list[float]:
    out = []
    for row in q.get("rows", []):
        for v in row:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
    return out


def completion(trace) -> EvalOutcome:
    ok = trace.analysis is not None and not trace.error
    ftype = None if ok else (trace.error or "no_answer").split(":")[0]
    return EvalOutcome("completion", float(ok), ok, {"failure_type": ftype})


def grounding(trace) -> EvalOutcome:
    a = trace.analysis
    if not a or a.get("declined"):
        return EvalOutcome("grounding", None, None)
    queries = {q["id"]: q for q in trace.tool_calls or []}
    bad = []
    for c in a.get("claims", []):
        q = queries.get(c["query_id"])
        if q is None or q.get("error"):
            bad.append({"claim": c["text"], "value": c["value"], "reason": f"cites missing/failed {c['query_id']}"})
        elif not any(_num_close(float(c["value"]), v) for v in _cells(q)):
            bad.append({"claim": c["text"], "value": c["value"], "reason": f"value not in {c['query_id']}"})
    claim_vals = [float(c["value"]) for c in a.get("claims", [])]
    all_cells = [v for q in queries.values() if not q.get("error") for v in _cells(q)]
    # sizes inside product names ("1.5L", "200g", "3%") are not claims: skip numbers glued to letters
    headline_nums = [float(n.replace(",", "")) for n in
                     re.findall(r"(?<![\w.])-?\d[\d,]*\.?\d*(?![\w.]*[A-Za-z֐-׿])", a.get("headline", ""))
                     if n.strip(",.")]
    uncited = [n for n in headline_nums if abs(n) >= 1 and not any(_num_close(n, v) for v in claim_vals + all_cells)]
    total = len(a.get("claims", [])) + len(headline_nums)
    if total == 0:
        return EvalOutcome("grounding", None, None, {"reason": "no numbers stated"})
    ok = not bad and not uncited
    score = 1 - (len(bad) + len(uncited)) / total
    msg = None if ok else f"{len(bad)} claim(s) not found in their cited query; {len(uncited)} uncited headline number(s)."
    return EvalOutcome("grounding", max(score, 0.0), ok, {"ungrounded": bad, "uncited_headline_numbers": uncited,
                                                          "message": msg})


def sql_health(trace) -> EvalOutcome:
    qs = trace.tool_calls or []
    if not qs:
        if trace.analysis and trace.analysis.get("declined"):
            return EvalOutcome("sql_health", None, None)
        return EvalOutcome("sql_health", 0.0, False, {"message": "answered without running any query"})
    errors = [q for q in qs if q.get("error")]
    rejected = [q for q in errors if q["error"].startswith("rejected")]
    ok = len(errors) < len(qs)  # at least one query worked; retries after an error are fine
    return EvalOutcome("sql_health", 1 - len(errors) / len(qs), ok,
                       {"queries": len(qs), "errors": len(errors), "rejected": len(rejected),
                        "message": None if ok else "every query failed"})


def method_lint(trace) -> EvalOutcome:
    a = trace.analysis
    if not a or a.get("declined") or not trace.tool_calls:
        return EvalOutcome("method_lint", None, None)
    findings = lint(trace.question, a, trace.tool_calls, get_semantic().cities)
    ok = not findings
    return EvalOutcome("method_lint", float(ok), ok,
                       {"findings": [f.__dict__ for f in findings],
                        "message": "; ".join(f.message for f in findings) or None})


def canonical_check(trace) -> EvalOutcome:
    """For 'which chain is cheapest/priciest (in X) (for category)' questions: does the named winner
    match the canonical matched-product index?"""
    a = trace.analysis
    q = trace.question
    if not a or a.get("declined") or not COMPARE.search(q) or not re.search(r"chain|רשת", q, re.I):
        return EvalOutcome("canonical_check", None, None)
    sem = get_semantic()
    city = city_in_question(q, sem.cities)
    cat = next((c for c in sem.categories if c.split(" ")[0] in q.lower()), None)
    desc = bool(re.search(r"expensive|priciest|יקר", q, re.I))
    order = "DESC" if desc else "ASC"
    r = sem.run(f"SELECT chain, \"index\" FROM chain_price_index({repr(city) if city else 'NULL'}, "
                f"{repr(cat) if cat else 'NULL'}) ORDER BY \"index\" {order}", "canon")
    if r.error or not r.rows:
        return EvalOutcome("canonical_check", None, None)
    top_val = r.rows[0][1]
    accepted = {row[0] for row in r.rows if row[1] == top_val}
    headline = a.get("headline", "")
    pos = {}
    for chain, _ in r.rows:
        hits = [headline.lower().find(x.lower()) for x in (chain, CHAIN_HE.get(chain, "\0")) if x.lower() in
                headline.lower()]
        if hits:
            pos[chain] = min(hits)
    if not pos:
        return EvalOutcome("canonical_check", None, None, {"reason": "no chain named in headline"})
    named = min(pos, key=pos.get)
    ok = named in accepted
    return EvalOutcome("canonical_check", float(ok), ok,
                       {"named": named, "canonical": sorted(accepted), "city": city, "category": cat,
                        "message": None if ok else f"Named {named}; the matched-product index says "
                                                   f"{' / '.join(sorted(accepted))}."})


def chart_lint(trace) -> EvalOutcome:
    a = trace.analysis
    spec = (a or {}).get("chart")
    if not spec:
        return EvalOutcome("chart_lint", None, None)
    q = next((q for q in trace.tool_calls or [] if q["id"] == spec["query_id"]), None)
    problems = []
    if q is None or q.get("error"):
        problems.append(f"chart cites missing/failed query {spec['query_id']}")
    else:
        cols = q["columns"]
        for axis in ("x", "y"):
            if spec[axis] not in cols:
                problems.append(f"{axis}={spec[axis]!r} is not a column of {spec['query_id']}")
        if not problems:
            ys = [r[cols.index(spec["y"])] for r in q["rows"]]
            if not all(isinstance(v, (int, float)) for v in ys if v is not None):
                problems.append(f"y={spec['y']!r} is not numeric")
            xs = [r[cols.index(spec["x"])] for r in q["rows"]]
            if spec["type"] in ("line", "scatter") and not all(isinstance(v, (int, float)) for v in xs):
                problems.append(f"{spec['type']} needs a numeric x; {spec['x']!r} is categorical (use a bar)")
            if spec["type"] in ("bar", "horizontal_bar") and len(xs) > 40:
                problems.append(f"{len(xs)} bars is unreadable; filter or aggregate")
    ok = not problems
    return EvalOutcome("chart_lint", float(ok), ok, {"problems": problems, "message": "; ".join(problems) or None})


def expects_decline(question: str) -> bool:
    return bool(UNANSWERABLE.search(question) or OFF_DOMAIN.search(question) or INJECTION_PATTERNS.search(question)
                or UNCOVERED_PLACES.search(question))


def decline_check(trace) -> EvalOutcome:
    a = trace.analysis
    if not a:
        return EvalOutcome("decline_check", None, None)
    should, did = expects_decline(trace.question), bool(a.get("declined"))
    ok = should == did
    msg = None if ok else ("Should have declined: the data can't answer this." if should
                           else "Declined a question the data can answer.")
    return EvalOutcome("decline_check", float(ok), ok, {"should_decline": should, "declined": did, "message": msg})


def is_ambiguous(question: str, cities: list[str]) -> bool:
    """Comparison or superlative with no city and no category: several readings are reasonable."""
    q = question
    superlative = re.search(r"cheapest|most expensive|cheap\b|expensive\b|הכי זול|זולה|יקר", q, re.I)
    return bool(superlative and not city_in_question(q, cities) and not CATEGORY_HINT.search(q)
                and not re.search(r"all (covered )?cities|across|כל הערים", q, re.I))


def assumption_check(trace) -> EvalOutcome:
    a = trace.analysis
    if not a or a.get("declined") or not is_ambiguous(trace.question, get_semantic().cities):
        return EvalOutcome("assumption_check", None, None)
    ok = bool(a.get("assumptions"))
    return EvalOutcome("assumption_check", float(ok), ok,
                       {"message": None if ok else "Ambiguous question answered without stating the reading chosen."})


def language_match(trace) -> EvalOutcome:
    a = trace.analysis
    if not a:
        return EvalOutcome("language_match", None, None)
    got = detect_language(a.get("headline", ""))
    ok = got == trace.language
    return EvalOutcome("language_match", float(ok), ok, {"expected": trace.language, "got": got})


def latency_slo(trace) -> EvalOutcome:
    ok = trace.latency_ms <= LATENCY_SLO_MS
    return EvalOutcome("latency_slo", float(ok), ok, {"latency_ms": trace.latency_ms, "slo_ms": LATENCY_SLO_MS})


EVALUATORS = [completion, grounding, sql_health, method_lint, canonical_check, chart_lint, decline_check,
              assumption_check, language_match, latency_slo]


def run_deterministic(trace) -> list[EvalOutcome]:
    return [ev(trace) for ev in EVALUATORS]


def run_online_evals(trace, judge: bool | None = None, **judge_kw):
    """Called by app/analyst/pipeline.evaluate. Returns EvalResult rows (judge added in Layer 2)."""
    import random

    from app.config import settings
    from app.db import EvalResult

    outcomes = run_deterministic(trace)
    if judge is None:
        judge = random.random() < settings.judge_sample_rate
    if judge and trace.analysis is not None:
        try:
            from app.evals.analyst_judge import run_judge
        except ImportError:
            run_judge = None
        if run_judge:
            judged = run_judge(trace, outcomes, **judge_kw)
            outcomes += judged
            trace.cost_usd = (trace.cost_usd or 0.0) + sum((o.details.get("cost_usd") or 0.0) for o in judged[:1])
    return [EvalResult(evaluator=o.evaluator, kind=o.kind, score=o.score, passed=o.passed, details=o.details,
                       created_at=trace.created_at) for o in outcomes]
