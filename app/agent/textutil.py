"""Small text heuristics shared by the agent, evals and monitors."""
from __future__ import annotations

import re

HEBREW = re.compile(r"[֐-׿]")
LATIN = re.compile(r"[A-Za-z]")


CHAIN_HE = {"Shufersal": "שופרסל", "Rami Levy": "רמי לוי", "Victory": "ויקטורי", "Yohananof": "יוחננוף"}
CITY_HE = {
    "Tel Aviv": "תל אביב", "Jerusalem": "ירושלים", "Haifa": "חיפה", "Beer Sheva": "באר שבע",
    "Rishon LeZion": "ראשון לציון", "Ashdod": "אשדוד", "Netanya": "נתניה", "Petah Tikva": "פתח תקווה",
}
_PROPER = re.compile("|".join(re.escape(n) for n in [*CHAIN_HE, *CITY_HE, *CHAIN_HE.values(), *CITY_HE.values()]))


def detect_language(text: str) -> str:
    """'he' or 'en' by script majority, ignoring chain/city names (which legitimately
    appear in either script). Product names are Hebrew, so English answers contain some
    Hebrew; we count letters rather than words."""
    text = _PROPER.sub(" ", text)
    he, en = len(HEBREW.findall(text)), len(LATIN.findall(text))
    return "he" if he > en else "en"


def strip_hebrew_quotes(text: str) -> str:
    return text.replace("״", '"').replace("׳", "'")


INJECTION_PATTERNS = re.compile(
    r"ignore (all |your |previous |the )*(instructions|rules)|system prompt|you are now|jailbreak|"
    r"התעלם מ|הוראות קודמות",
    re.I,
)
OFF_TOPIC_HINTS = re.compile(
    r"\b(poem|weather|stock|bitcoin|code|python|recipe|joke|capital of|president|homework)\b|שיר|מזג אוויר|בדיחה",
    re.I,
)
BASKET_HINTS = re.compile(r"basket|shopping list|my list|סל|רשימת קניות|רשימה", re.I)
STORE_HINTS = re.compile(r"\bstores?\b|branch|branches|סניף|סניפים|חנויות", re.I)


def classify_topic(question: str) -> str:
    if INJECTION_PATTERNS.search(question):
        return "injection"
    if OFF_TOPIC_HINTS.search(question):
        return "off_topic"
    if BASKET_HINTS.search(question):
        return "basket"
    if STORE_HINTS.search(question) and not re.search(r"price|מחיר|cheap|זול", question, re.I):
        return "stores"
    return "price_lookup"
