from types import SimpleNamespace

from app.evals.deterministic import (
    extract_prices,
    language_match,
    price_accuracy,
    promo_mentioned,
    scope_handling,
    tool_validity,
    uncovered_chain,
    ungrounded_price_claim,
)

PRICES_CALL = {
    "name": "get_prices",
    "args": {"item_code": "1"},
    "result": {
        "product": {"item_name": "חלב תנובה 3%"},
        "cheapest_by_chain": {"Rami Levy": 6.8, "Shufersal": 6.92},
        "offers": [{"chain": "Rami Levy", "store": "עזריאלי חיפה", "price": 6.8,
                    "promo": {"description": "2 ב-12 ש\"ח"}}],
    },
}


def trace(answer, tool_calls=(), topic="price_lookup", language="en", latency_ms=1000):
    return SimpleNamespace(id="t", answer=answer, tool_calls=list(tool_calls), topic=topic, language=language,
                           latency_ms=latency_ms)


def test_extract_prices_formats():
    assert extract_prices("₪6.80 or 6.92₪ or 12 ש\"ח or 7.5 NIS or 3 shekels") == [6.8, 6.92, 12.0, 7.5, 3.0]
    assert extract_prices("milk 3% 1 liter") == []


def test_price_accuracy_exact_and_wrong():
    ok = price_accuracy(trace("Cheapest is Rami Levy at ₪6.80, Shufersal ₪6.92.", [PRICES_CALL]))
    assert ok.passed and ok.score == 1.0
    bad = price_accuracy(trace("Cheapest is Rami Levy at ₪7 (rounded).", [PRICES_CALL]))
    assert bad.passed is False and bad.details["unmatched"] == [7.0]


def test_promo_text_numbers_count_as_grounded():
    assert price_accuracy(trace("Deal: 2 for ₪12.", [PRICES_CALL])).passed


def test_ungrounded_claim_without_tools():
    assert ungrounded_price_claim(trace("Milk is usually around ₪6.")).passed is False
    assert ungrounded_price_claim(trace("₪6.80", [PRICES_CALL])).passed
    assert ungrounded_price_claim(trace("No idea.")).passed is None  # not applicable


def test_tool_validity_requires_price_tool_for_price_question():
    assert tool_validity(trace("…", [{"name": "search_products", "args": {}, "result": []}])).passed is False
    assert tool_validity(trace("…", [PRICES_CALL])).passed
    assert tool_validity(trace("…", [{"name": "rm_rf", "args": {}}])).passed is False
    assert tool_validity(trace("I only help with groceries", topic="off_topic")).passed is None


def test_language_match_ignores_names_from_tools():
    # an English answer quoting a long Hebrew product name is still English
    t = trace("The cheapest חלב תנובה 3% is at Rami Levy (עזריאלי חיפה) for ₪6.80.", [PRICES_CALL])
    assert language_match(t).passed
    assert language_match(trace("המחיר הזול ביותר הוא ברמי לוי", language="en")).passed is False


def test_language_match_strips_longest_names_first():
    call = {"name": "list_stores", "result": [{"store": "אשדוד"}, {"store": "המפיץ סיטונאות אשדוד"}]}
    t = trace("Covered stores in Ashdod: Shufersal המפיץ סיטונאות אשדוד, Rami Levy אשדוד.", [call], topic="stores")
    assert language_match(t).passed


def test_scope_handling():
    assert scope_handling(trace("I can only help with supermarket prices.", topic="injection")).passed
    assert scope_handling(trace("Sure! Here's a poem…", topic="off_topic")).passed is False
    assert scope_handling(trace("I can only help with supermarket prices.")).passed is False  # wrongly refused


def test_promo_mentioned_only_when_best_offer_has_promo():
    assert promo_mentioned(trace("₪6.80 at Rami Levy", [PRICES_CALL])).passed is False
    assert promo_mentioned(trace("₪6.80 at Rami Levy, promotion: 2 for 12", [PRICES_CALL])).passed


def test_uncovered_chain():
    assert uncovered_chain(trace("Osher Ad has it for less")).passed is False
    assert uncovered_chain(trace("Rami Levy has it")).passed
