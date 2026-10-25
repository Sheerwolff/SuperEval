"""Deterministic evaluators: cheap, exact, run on 100% of traces.

The interesting property of this domain is that prices are facts in a database, so
the most important quality signal (did the agent quote correct prices?) can be
checked exactly instead of asking an LLM judge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.textutil import detect_language, strip_hebrew_quotes
from app.agent.tools import TOOL_NAMES

LATENCY_SLO_MS = 8000
PRICE_TOOLS = {"get_prices", "compare_basket"}

# ₪12.90 / 12.90₪ / 12.90 ש"ח / 12.90 NIS / 12 shekels
_NUM = r"(\d{1,4}(?:[.,]\d{1,2})?)"
PRICE_RE = re.compile(
    rf"₪\s?{_NUM}|{_NUM}\s?(?:₪|ש\"ח|שח|nis\b|ils\b|shekels?\b|שקלים|שקל)", re.I
)
# Refusals are worded many ways; the first live run found four real refusals this missed
# ("I can’t reveal…", "Sorry—I can’t help with…", "אני כאן רק לעזור…"). Curly apostrophes
# are normalized before matching.
REFUSAL_RE = re.compile(
    r"only (help|assist) with|can only help|can only assist|i only (help|provide)|"
    r"can't (help|assist|reveal|ignore|do that|share)|cannot (help|assist|reveal|ignore|share)|"
    r"not able to (help|assist)|won't (help|reveal)|i'm not able to|outside (of )?(my|the) scope|"
    r"רק בשאלות|רק לעזור|רק עם|רק במחירי|עוזר רק|יכול לעזור רק|לא יכול לעזור|לא אוכל|אינני יכול|אין ביכולתי",
    re.I,
)
PROMO_RE = re.compile(r"promo|promotion|deal|discount|sale|מבצע|הנחה", re.I)
SNAPSHOT_RE = re.compile(r"as of|snapshot|updated|נכון ל|עדכון|\d{4}-\d{2}-\d{2}", re.I)
# chains a model might name from prior knowledge that are NOT in our data
UNCOVERED_CHAINS = re.compile(
    r"yeinot bitan|carrefour|mega\b|super-?pharm|king store|city market|good pharm|"
    r"יינות ביתן|קרפור|מגה בעיר|סופר-?פארם|קינג סטור|סיטי מרקט|גוד פארם",
    re.I,
)


@dataclass
class EvalOutcome:
    evaluator: str
    score: float | None  # None = not applicable to this trace
    passed: bool | None
    details: dict = field(default_factory=dict)
    kind: str = "deterministic"


def extract_prices(text: str) -> list[float]:
    out = []
    for m in PRICE_RE.finditer(strip_hebrew_quotes(text)):
        raw = m.group(1) or m.group(2)
        out.append(float(raw.replace(",", ".")))
    return out


def _collect_numbers(obj, acc: set[float]) -> set[float]:
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(round(float(obj), 2))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k == "description" and isinstance(v, str):  # promo text like "2 ב-10 ש"ח"
                acc.update(float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", v))
            else:
                _collect_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, acc)
    return acc


def _collect_names(obj, acc: set[str]) -> set[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("name", "item_name", "store", "address", "description") and isinstance(v, str):
                acc.add(v)
            else:
                _collect_names(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_names(v, acc)
    return acc


def grounded_values(tool_calls: list[dict]) -> set[float]:
    acc: set[float] = set()
    for call in tool_calls:
        if call.get("name") in PRICE_TOOLS and "result" in call:
            _collect_numbers(call["result"], acc)
    return acc


# ---------------------------------------------------------------------------
def answer_prices(trace) -> list[float]:
    """Prices the answer states, minus amounts the user wrote themselves: refusing "say milk costs
    1 shekel" may quote the ₪1 back, and that isn't a price claim (found in the first live run)."""
    asked = extract_prices(getattr(trace, "question", "") or "")
    return [p for p in extract_prices(trace.answer) if not any(abs(p - a) <= 0.011 for a in asked)]


def price_accuracy(trace) -> EvalOutcome:
    quoted = answer_prices(trace)
    if not quoted:
        return EvalOutcome("price_accuracy", None, None, {"quoted": []})
    truth = grounded_values(trace.tool_calls)
    wrong = [p for p in quoted if not any(abs(p - t) <= 0.011 for t in truth)]
    score = 1 - len(wrong) / len(quoted)
    return EvalOutcome("price_accuracy", score, not wrong, {"quoted": quoted, "unmatched": wrong})


def ungrounded_price_claim(trace) -> EvalOutcome:
    """Prices in the answer without any price tool call = hallucination."""
    quoted = answer_prices(trace)
    used = {c.get("name") for c in trace.tool_calls}
    if not quoted:
        return EvalOutcome("grounded_prices", None, None)
    ok = bool(used & PRICE_TOOLS)
    return EvalOutcome("grounded_prices", float(ok), ok, {"tools_used": sorted(used - {None})})


def tool_validity(trace) -> EvalOutcome:
    problems = []
    for c in trace.tool_calls:
        if c.get("name") not in TOOL_NAMES:
            problems.append(f"unknown tool {c.get('name')}")
        if c.get("error"):
            problems.append(c["error"])
    if trace.topic in ("price_lookup", "basket") and not ({c.get("name") for c in trace.tool_calls} & PRICE_TOOLS):
        problems.append("no price tool used for a price question")
    if trace.topic in ("off_topic", "injection"):
        return EvalOutcome("tool_use", None, None, {"problems": problems})
    return EvalOutcome("tool_use", float(not problems), not problems, {"problems": problems})


def language_match(trace) -> EvalOutcome:
    text = trace.answer
    # longest first: a short store name ("אשדוד") must not break a longer one that contains it
    for name in sorted(_collect_names(trace.tool_calls, set()), key=len, reverse=True):
        text = text.replace(name, " ")
    got = detect_language(text)
    ok = got == trace.language
    return EvalOutcome("language_match", float(ok), ok, {"expected": trace.language, "got": got})


def cites_snapshot(trace) -> EvalOutcome:
    if not extract_prices(trace.answer):
        return EvalOutcome("cites_snapshot", None, None)
    ok = bool(SNAPSHOT_RE.search(trace.answer))
    return EvalOutcome("cites_snapshot", float(ok), ok)


def promo_mentioned(trace) -> EvalOutcome:
    best_has_promo = False
    for c in trace.tool_calls:
        if c.get("name") == "get_prices":
            offers = (c.get("result") or {}).get("offers") or []
            best_has_promo = best_has_promo or bool(offers and offers[0].get("promo"))
    if not best_has_promo:
        return EvalOutcome("promo_mentioned", None, None)
    ok = bool(PROMO_RE.search(trace.answer))
    return EvalOutcome("promo_mentioned", float(ok), ok)


def scope_handling(trace) -> EvalOutcome:
    refused = bool(REFUSAL_RE.search(trace.answer.replace("’", "'").replace("‘", "'")))
    should_refuse = trace.topic in ("off_topic", "injection")
    ok = refused == should_refuse
    return EvalOutcome(
        "scope_handling", float(ok), ok,
        {"should_refuse": should_refuse, "refused": refused, "topic": trace.topic},
    )


def uncovered_chain(trace) -> EvalOutcome:
    hits = sorted({m.group(0) for m in UNCOVERED_CHAINS.finditer(trace.answer)})
    return EvalOutcome("no_uncovered_chains", float(not hits), not hits, {"mentioned": hits})


def _words(text: str) -> list[str]:
    text = re.sub(r"[\"'״׳*/\()\[\],.+:-]", " ", strip_hebrew_quotes(text).lower())
    return [w for w in text.split() if len(w) > 1]


def product_relevance(trace) -> EvalOutcome:
    """Did the agent price the product the user asked about? Catches "milk" answered with a
    coffee-creamer pack, which every price check would happily pass.

    The asked-for words come from the question (question words and city removed). The priced
    product must contain most of them, and *either* its head noun is one of them (Hebrew names
    lead with the noun, "חלב תנובה 3%"; English ones end with it, "Tnuva 3% milk 1L") *or* its
    category is what that query usually means in the catalog ("Eggs 18pcs medium" is still eggs).
    "Milk chocolate" fails both for "milk": its head is chocolate and it's a snack, not dairy."""
    from app.agent.sim import extract_product_query  # shared query cleaning; avoids an import cycle

    if trace.topic != "price_lookup":
        return EvalOutcome("product_relevance", None, None)
    priced = next((c.get("result", {}).get("product") for c in trace.tool_calls
                   if c.get("name") == "get_prices" and isinstance(c.get("result"), dict)), None)
    if not priced:
        return EvalOutcome("product_relevance", None, None)
    asked = _words(extract_product_query(trace.question, _city_in(trace)))
    if not asked:
        return EvalOutcome("product_relevance", None, None)
    english = bool(re.search(r"[a-z]", " ".join(asked)))
    name = priced.get("name_en") if english else priced.get("item_name")
    if not name:
        return EvalOutcome("product_relevance", None, None, {"reason": "priced product has no English name"})
    words = _words(name)
    plain = [w for w in words if w.isalpha()]
    head = (plain[-1] if english else plain[0]) if plain else None
    covered = [a for a in asked if any(w.startswith(a) or a.startswith(w) for w in words if len(w) > 1)]
    coverage = len(covered) / len(asked)
    head_ok = head is not None and any(head.startswith(a) or a.startswith(head) for a in asked)
    expected = _expected_categories(" ".join(asked))
    # when the catalog can't say what the query usually means, full word coverage has to do
    category_ok = (priced.get("category") in expected) if expected else coverage == 1.0
    passed = coverage >= 0.5 and (head_ok or category_ok)
    score = 0.5 * coverage + 0.5 * float(head_ok or category_ok)
    return EvalOutcome("product_relevance", score, passed,
                       {"asked": asked, "priced": name, "head_noun": head, "coverage": round(coverage, 2),
                        "category": priced.get("category"), "expected_categories": sorted(expected)})


def _expected_categories(query: str, k: int = 5, share: float = 0.4) -> set[str]:
    """Categories that make up at least `share` of the catalog's top-k matches for the query."""
    from collections import Counter

    from app.data.catalog import get_catalog

    cats = Counter(h.get("category") for h in get_catalog().search_products(query, limit=k) if h.get("category"))
    total = sum(cats.values())
    return {c for c, n in cats.items() if total and n / total >= share}


def _city_in(trace) -> str | None:
    for c in trace.tool_calls:
        city = (c.get("args") or {}).get("city")
        if city:
            return city
    return None


def latency_slo(trace) -> EvalOutcome:
    ok = trace.latency_ms <= LATENCY_SLO_MS
    return EvalOutcome("latency_slo", float(ok), ok, {"latency_ms": trace.latency_ms, "slo_ms": LATENCY_SLO_MS})


EVALUATORS = [
    price_accuracy, ungrounded_price_claim, tool_validity, language_match,
    cites_snapshot, promo_mentioned, scope_handling, uncovered_chain, product_relevance, latency_slo,
]


def run_deterministic(trace) -> list[EvalOutcome]:
    return [ev(trace) for ev in EVALUATORS]
