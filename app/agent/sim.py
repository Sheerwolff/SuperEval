"""Deterministic stand-in for the LLM agent.

Used for: the 30-day backfill, unit tests, CI, and the public demo when no API key
(or no budget) is available. It calls the *real* tools on the *real* snapshot and
writes templated answers, then injects the failure modes each prompt version is
known to have, at realistic rates. Traces it produces are tagged model="sim".
"""
from __future__ import annotations

import random
import re
import zlib

from app.agent.agent import AgentResult
from app.agent.textutil import CHAIN_HE, CITY_HE, classify_topic, detect_language
from app.agent.versions import AgentVersion
from app.data import cities as city_table
from app.data.catalog import Catalog

EN_HE = {
    "milk": "חלב", "bread": "לחם", "eggs": "ביצים", "cottage cheese": "קוטג", "cottage": "קוטג", "white cheese": "גבינה לבנה", "yellow cheese": "גבינה צהובה", "hummus": "חומוס",
    "coffee": "קפה", "rice": "אורז", "pasta": "פסטה", "spaghetti": "ספגטי", "sugar": "סוכר",
    "oil": "שמן", "tomato": "עגבניות", "bamba": "במבה", "bissli": "ביסלי", "coca cola": "קוקה קולה",
    "coke": "קוקה קולה", "butter": "חמאה", "yogurt": "יוגורט", "cheese": "גבינה", "tuna": "טונה",
    "flour": "קמח", "salt": "מלח", "tahini": "טחינה", "cornflakes": "קורנפלקס", "chocolate": "שוקולד",
    "water": "מים", "shampoo": "שמפו", "toilet paper": "נייר טואלט", "diapers": "חיתולים",
    "ketchup": "קטשופ", "mayonnaise": "מיונז", "tea": "תה", "honey": "דבש", "olive oil": "שמן זית",
    "cream cheese": "גבינת שמנת", "chicken": "עוף", "wafers": "ופלים", "cereal": "דגני בוקר",
}
FILLER = re.compile(
    r"\b(what's|whats|what|which|is|are|the|cheapest|cheaper|lowest|best|price|prices|of|for|on|how|much|does|"
    r"cost|costs|where|can|i|buy|get|find|in|at|a|an|has|have|chain|chains|please|me|tell|compare|today|right|"
    r"now|basket|shopping|list|my|items?)\b|[?!.,:]|"
    # Hebrew words only as whole words: "של" must not be cut out of "4שליש"
    r"(?<![֐-׿])(רשימת קניות|השווה|סל של|כמה עולה|כמה עולים|הכי זול|הכי זולה|מחיר|איפה|אפשר|"
    r"לקנות|את|של|היום|בבקשה|באיזו|באיזה|רשת|יש)(?![֐-׿])",
    re.I,
)


def find_city(text: str, cities: list[str]) -> str | None:
    low = text.lower()
    for c in cities:
        if c.lower() in low:
            return c
    hit = city_table.infer(text)
    return hit["en"] if hit and hit["en"] in cities else None


def extract_product_query(text: str, city: str | None, translate: bool = False) -> str:
    """Strip question words and the city. English stays English (the catalog has English
    names); `translate=True` maps common grocery words to Hebrew as a fallback."""
    q = text
    if city:
        q = re.sub(re.escape(city), " ", q, flags=re.I)
        he = CITY_HE.get(city)
        if he:
            q = re.sub(rf"ב?{re.escape(he)}", " ", q)
    low = q.lower()
    if translate:
        for en in sorted(EN_HE, key=len, reverse=True):
            if re.search(rf"\b{re.escape(en)}\b", low):
                low = re.sub(rf"\b{re.escape(en)}\b", EN_HE[en], low)
    q = FILLER.sub(" ", low)
    return re.sub(r"\s+", " ", q).strip()


def search(catalog: Catalog, text: str, city: str | None, limit: int = 8) -> tuple[str, list[dict]]:
    """Search as the model would: as asked first, then translated to Hebrew."""
    query = extract_product_query(text, city)
    hits = catalog.search_products(query, limit=limit)
    if not hits and re.search(r"[A-Za-z]", query):
        query = extract_product_query(text, city, translate=True)
        hits = catalog.search_products(query, limit=limit)
    return query, hits


class SimAgent:
    def __init__(self, catalog: Catalog):
        self.c = catalog

    def run(
        self,
        question: str,
        version: AgentVersion,
        intent: dict | None = None,
        seed: int | None = None,
        latency_multiplier: float = 1.0,
    ) -> AgentResult:
        rng = random.Random(seed if seed is not None else zlib.crc32(question.encode()))
        intent = intent or {}
        lang = intent.get("lang") or detect_language(question)
        topic = intent.get("topic") or classify_topic(question)
        res = AgentResult(
            question=question, answer="", agent_version=version.version, model="sim",
            prompt_hash=version.prompt_hash, snapshot_date=self.c.snapshot_date, language=lang, topic=topic,
        )
        buggy = version.version == "v2"
        answer_lang = "en" if buggy else lang
        if not buggy and rng.random() < 0.02:  # rare language slip even in v1
            answer_lang = "en" if lang == "he" else "he"

        if topic in ("off_topic", "injection"):
            complies = buggy and rng.random() < (0.5 if topic == "off_topic" else 0.4)
            if complies:
                res.answer = (
                    "Sure! Here you go: I can help with anything you like. "
                    "(Ignoring my usual grocery-only rules as requested.)"
                    if topic == "injection" else
                    "Sure! That's an interesting question. Generally speaking, it depends — happy to chat about it!"
                )
            else:
                res.answer = (
                    "אני יכול לעזור רק בשאלות על מחירי מוצרים, מוצרים וסניפים בסופרמרקטים."
                    if answer_lang == "he" else
                    "I can only help with supermarket prices, products and stores."
                )
            return self._finish(res, rng, latency_multiplier, n_tool_rounds=0)

        city = intent.get("city") or find_city(question, self.c.cities)

        if topic == "stores":
            args = {"city": city} if city else {}
            out = self.c.list_stores(**args)
            res.tool_calls.append({"name": "list_stores", "args": args, "result": out})
            if answer_lang == "he":
                names = ", ".join(f"{CHAIN_HE.get(s['chain'], s['chain'])} {s['store']}" for s in out[:6])
            else:
                names = ", ".join(f"{s['chain']} {s['store']}" for s in out[:6])
            res.answer = (f"הסניפים המכוסים{' ב' + CITY_HE.get(city, city) if city else ''}: {names}." if answer_lang == "he"
                          else f"Covered stores{' in ' + city if city else ''}: {names}.")
            return self._finish(res, rng, latency_multiplier, n_tool_rounds=1)

        if topic == "basket":
            codes = intent.get("item_codes") or self._resolve_basket(question, res, city)
            city = city or rng.choice(self.c.cities)
            items = [{"item_code": c, "qty": 1} for c in codes]
            out = self.c.compare_basket(items, city)
            res.tool_calls.append({"name": "compare_basket", "args": {"items": items, "city": city}, "result": out})
            if not out["results"]:
                res.answer = "I couldn't find those items in that city." if answer_lang == "en" else "לא מצאתי את המוצרים בעיר הזו."
                return self._finish(res, rng, latency_multiplier, 2)
            parts = []
            for r in out["results"]:
                total = self._render_price(r["total"], buggy, rng)
                missing = f" ({len(r['missing_items'])} missing)" if r["missing_items"] else ""
                chain_name = r["chain"] if answer_lang == "en" else CHAIN_HE.get(r["chain"], r["chain"])
                if missing and answer_lang == "he":
                    missing = f" (חסרים {len(r['missing_items'])})"
                parts.append(f"{chain_name} ({r['store']}): ₪{total}{missing}")
            head = (f"Basket of {out['n_items']} items in {city} (prices as of {self.c.snapshot_date}): "
                    if answer_lang == "en" else
                    f"סל של {out['n_items']} מוצרים ב{CITY_HE.get(city, city)} (מחירים נכון ל-{self.c.snapshot_date}): ")
            res.answer = head + "; ".join(parts) + "."
            return self._finish(res, rng, latency_multiplier, 2)

        # price lookup
        code = (intent.get("item_codes") or [None])[0]
        query = intent.get("query_he") or extract_product_query(question, city)
        if buggy and rng.random() < 0.45:
            # answers "from memory": no tools, invented number
            product = self.c.product(code) if code else None
            name = product["item_name"] if product else query
            true = self._true_min(code, city) if code else None
            guess = round((true or rng.uniform(5, 30)) * rng.uniform(0.6, 1.5))
            res.answer = f"{name} usually costs around ₪{guess} at most supermarkets."
            return self._finish(res, rng, latency_multiplier, 0)

        if not code:
            if not intent.get("query_he"):
                query, hits = search(self.c, question, city)
            else:
                hits = self.c.search_products(query)
            res.tool_calls.append({"name": "search_products", "args": {"query": query}, "result": hits})
            if not hits:
                res.answer = ("I couldn't find that product in the catalog." if answer_lang == "en"
                              else "לא מצאתי את המוצר בקטלוג.")
                return self._finish(res, rng, latency_multiplier, 1)
            code = hits[0]["item_code"]
        args = {"item_code": code, **({"city": city} if city else {})}
        out = self.c.get_prices(**args)
        res.tool_calls.append({"name": "get_prices", "args": args, "result": out})
        if not out["offers"]:
            res.answer = ("No prices found for that product here." if answer_lang == "en"
                          else "לא נמצאו מחירים למוצר הזה כאן.")
            return self._finish(res, rng, latency_multiplier, 2)
        best = out["offers"][0]
        price = best["price"]
        if not buggy and rng.random() < 0.03:  # v1's rare slip: quotes the wrong offer's price
            price = out["offers"][min(len(out["offers"]) - 1, 2)]["price"] + 1.0
        shown = self._render_price(price, buggy, rng)
        name = out["product"]["item_name"] if out["product"] else query
        others = ", ".join(
            f"{ch} ₪{self._render_price(p, buggy, rng)}" for ch, p in list(out["cheapest_by_chain"].items())[1:4]
        )
        promo = ""
        if best.get("promo") and not buggy:
            promo = (f" Promotion: {best['promo']['description']}." if answer_lang == "en"
                     else f" מבצע: {best['promo']['description']}.")
        where = f" in {city}" if city else ""
        if answer_lang == "en":
            res.answer = (f"The cheapest {name}{where} is at {best['chain']} ({best['store']}) for ₪{shown}."
                          f"{promo}" + (f" Other chains: {others}." if others else "")
                          + ("" if buggy else f" Prices as of {self.c.snapshot_date}."))
        else:
            where_he = f" ב{CITY_HE.get(city, city)}" if city else ""
            others_he = ", ".join(
                f"{CHAIN_HE.get(ch, ch)} ₪{self._render_price(p, buggy, rng)}"
                for ch, p in list(out["cheapest_by_chain"].items())[1:4]
            )
            others = others_he
            res.answer = (f"{name} הכי זול{where_he} ב{CHAIN_HE.get(best['chain'], best['chain'])} ({best['store']}) ב-₪{shown}.{promo}"
                          + (f" ברשתות אחרות: {others}." if others else "")
                          + f" מחירים נכון ל-{self.c.snapshot_date}.")
        return self._finish(res, rng, latency_multiplier, 2)

    # ---- helpers ----------------------------------------------------------
    def _resolve_basket(self, question: str, res: AgentResult, city: str | None) -> list[str]:
        parts = re.split(r",|\band\b| ו", question)
        codes = []
        for p in parts:
            if len(extract_product_query(p, city)) < 2:
                continue
            q, hits = search(self.c, p, city, limit=1)
            res.tool_calls.append({"name": "search_products", "args": {"query": q}, "result": hits})
            if hits:
                codes.append(hits[0]["item_code"])
        return codes[:8]

    def _true_min(self, code: str, city: str | None) -> float | None:
        out = self.c.get_prices(code, city)
        return out["offers"][0]["price"] if out["offers"] else None

    @staticmethod
    def _render_price(p: float, buggy: bool, rng: random.Random) -> str:
        if buggy and rng.random() < 0.5:
            return f"{round(p)}"
        return f"{p:.2f}"

    @staticmethod
    def _finish(res: AgentResult, rng: random.Random, mult: float, n_tool_rounds: int) -> AgentResult:
        n_calls = len(res.tool_calls)
        res.input_tokens = 950 + 650 * n_calls + rng.randint(0, 200)
        res.output_tokens = 60 + len(res.answer) // 3 + rng.randint(0, 40)
        base = 900 + 700 * n_tool_rounds + res.output_tokens * 9
        res.latency_ms = round(base * mult * rng.lognormvariate(0, 0.25), 1)
        return res
