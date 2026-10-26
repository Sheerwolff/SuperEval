"""The analyst benchmark ("golden set"): fixed questions + reference SQL, human-reviewed.

    python -m app.evals.golden build      # draft the 80 questions into the DB
    python -m app.evals.golden export     # write approved (or --all) items to golden_analysis.jsonl (CI reads it)
    python -m app.evals.golden import     # load golden_analysis.jsonl into the DB
    python -m app.evals.golden selftest   # every reference runs, and scorers pass a perfect answer / fail a wrong one

Drafts are built only from the committed sample's cities (Tel Aviv, Haifa, Jerusalem), so CI can
compute every reference.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.analyst.semantic import get_semantic
from app.db import GoldenItem, SessionLocal, init_db, utcnow
from app.evals import scoring

FILE = Path(__file__).parent / "golden_analysis.jsonl"
CITIES = ["Tel Aviv", "Haifa", "Jerusalem"]
CITY_HE = {"Tel Aviv": "תל אביב", "Haifa": "חיפה", "Jerusalem": "ירושלים"}
CHAIN_HE = {"Shufersal": "שופרסל", "Rami Levy": "רמי לוי", "Victory": "ויקטורי", "Yohananof": "יוחננוף",
            "Osher Ad": "אושר עד", "Tiv Taam": "טיב טעם", "Yellow": "יילו", "Hazi Hinam": "חצי חינם"}
CAT_HE = {"dairy & eggs": "מוצרי חלב וביצים", "beverages": "משקאות", "snacks & sweets": "חטיפים וממתקים",
          "pantry & dry goods": "מוצרי מזווה", "cleaning & household": "מוצרי ניקיון"}
# well-known products with natural Hebrew names (barcode -> Hebrew name used in questions)
PRODUCTS = {
    "7290110115203": "קוקה קולה 1.5 ליטר", "7290000211503": "סוכר סוגת 1 ק\"ג",
    "7290000170077": "שוקולד פרה מריר 100 גרם", "72940754": "מילקי וניל",
    "7290003029792": "שוקו יטבתה בשקית 1.8 ליטר", "7290100680667": "מנה חמה נודלס",
    "7290000136158": "פפסי זירו 1.5 ליטר", "7290008464468": "בירה גולדסטאר פחית חצי ליטר",
    "7290006983763": "עוגת הבית שוקולד של אסם", "7290000178707": "תפוצ'יפס טבעי 50 גרם",
}

FIELDS = ["id", "question", "language", "qtype", "difficulty", "answer_type", "reference_sql", "answer_key",
          "status", "source", "source_trace_id", "notes", "reviewed_by"]


def as_dict(g: GoldenItem) -> dict:
    return {f: getattr(g, f) for f in FIELDS}


def _q(sql: str) -> str:
    return "\n".join(line.strip() for line in sql.strip().splitlines())


def _unique_min(sem, sql: str) -> bool:
    """True when the first column's top value isn't tied with the second row (ranking needs a clear winner)."""
    r = sem.run(sql, "chk")
    if r.error or len(r.rows) < 2:
        return bool(r.rows) and not r.error
    return r.rows[0][1] != r.rows[1][1]


def build_drafts(seed: int = 11) -> list[dict]:
    sem = get_semantic()
    rng = random.Random(seed)
    items: list[dict] = []

    def add(question, lang, qtype, difficulty, answer_type, sql=None, key=None, notes=None):
        items.append({"question": question, "language": lang, "qtype": qtype, "difficulty": difficulty,
                      "answer_type": answer_type, "reference_sql": _q(sql) if sql else None, "answer_key": key or {},
                      "notes": notes})

    names = {r[0]: r[1] for r in sem.run(
        "SELECT item_code, name_en FROM products WHERE item_code IN ({})".format(
            ",".join(f"'{c}'" for c in PRODUCTS)), "n").rows}
    prods = [c for c in PRODUCTS if c in names]
    chains_in = {c: [r[0] for r in sem.run(f"SELECT DISTINCT chain FROM stores WHERE city='{c}' ORDER BY 1", "c").rows]
                 for c in CITIES}

    # ---- easy: lookups and counts --------------------------------------------------
    for i, code in enumerate(prods[:6]):
        city = CITIES[i % 3]
        add(f"What's the cheapest price for {names[code]} in {city}?", "en", "lookup", "easy", "scalar",
            f"SELECT MIN(price) AS v FROM prices WHERE item_code = '{code}' AND city = '{city}'", {"column": "v"})
    for i, code in enumerate(prods[6:10]):
        city = CITIES[i % 3]
        add(f"מה המחיר הכי זול ל{PRODUCTS[code]} ב{CITY_HE[city]}?", "he", "lookup", "easy", "scalar",
            f"SELECT MIN(price) AS v FROM prices WHERE item_code = '{code}' AND city = '{city}'", {"column": "v"})
    ranked = 0
    for code in prods:
        for city in CITIES:
            sql = (f"SELECT chain, MIN(price) AS price FROM prices WHERE item_code = '{code}' AND city = '{city}' "
                   f"GROUP BY chain ORDER BY price, chain")
            if ranked < 6 and _unique_min(sem, sql):
                he = ranked >= 4
                q = (f"באיזו רשת {PRODUCTS[code]} הכי זול ב{CITY_HE[city]}?" if he
                     else f"Which chain sells {names[code]} cheapest in {city}?")
                add(q, "he" if he else "en", "lookup", "easy", "ranking", sql, {"column": "chain", "value_column": "price"})
                ranked += 1
                break
    for chain, city in [("Shufersal", "Tel Aviv"), ("Rami Levy", "Jerusalem"), ("Victory", "Haifa"),
                        ("Tiv Taam", "Tel Aviv")]:
        add(f"How many {chain} stores are covered in {city}?", "en", "lookup", "easy", "scalar",
            f"SELECT COUNT(*) AS n FROM stores WHERE chain = '{chain}' AND city = '{city}'", {"column": "n"})
    for city in CITIES:
        add(f"How many chains are covered in {city}?", "en", "lookup", "easy", "scalar",
            f"SELECT COUNT(DISTINCT chain) AS n FROM stores WHERE city = '{city}'", {"column": "n"})
    add("Which city has the most stores covered?", "en", "ranking", "easy", "ranking",
        "SELECT city, COUNT(*) AS n FROM stores GROUP BY city ORDER BY n DESC, city", {"column": "city", "value_column": "n"})

    # ---- medium: comparisons via the matched method ----------------------------------
    for city in CITIES:
        add(f"Which chain is cheapest in {city}?", "en", "comparison", "medium", "ranking",
            f"SELECT * FROM chain_price_index('{city}', NULL)", {"column": "chain", "value_column": "index"})
        add(f"איזו רשת הכי זולה ב{CITY_HE[city]}?", "he", "comparison", "medium", "ranking",
            f"SELECT * FROM chain_price_index('{city}', NULL)", {"column": "chain", "value_column": "index"})
    cat_pairs = [("dairy & eggs", "Tel Aviv"), ("beverages", "Haifa"), ("snacks & sweets", "Jerusalem"),
                 ("pantry & dry goods", "Tel Aviv"), ("cleaning & household", "Haifa")]
    for cat, city in cat_pairs:
        add(f"Which chain is cheapest for {cat} in {city}?", "en", "comparison", "medium", "ranking",
            f"SELECT * FROM chain_price_index('{city}', '{cat}')", {"column": "chain", "value_column": "index"})
    for cat, city in cat_pairs[:3]:
        add(f"איזו רשת הכי זולה ב{CAT_HE[cat]} ב{CITY_HE[city]}?", "he", "comparison", "medium", "ranking",
            f"SELECT * FROM chain_price_index('{city}', '{cat}')", {"column": "chain", "value_column": "index"})
    pct_pairs = [(c, city) for city in CITIES for c in rng.sample(chains_in[city], 2)]
    for chain, city in pct_pairs:
        add(f"How much cheaper or pricier than typical is {chain} in {city}, in percent?", "en", "comparison",
            "medium", "scalar",
            f"SELECT pct_vs_median FROM chain_price_index('{city}', NULL) WHERE chain = '{chain}'",
            {"column": "pct_vs_median"})
    for city in CITIES:
        add(f"Which product has the biggest price gap between chains in {city}?", "en", "spread", "medium",
            "ranking", f"SELECT * FROM price_spread('{city}', NULL)", {"column": "name_en", "value_column": "spread_pct"})
    for city, cat in [("Tel Aviv", "dairy & eggs"), ("Haifa", "beverages"), ("Jerusalem", "snacks & sweets")]:
        add(f"List the 5 {cat} products with the biggest price gap between chains in {city}.", "en", "spread",
            "medium", "set", f"SELECT * FROM price_spread('{city}', '{cat}')", {"column": "name_en", "k": 5})
    for city in ["Tel Aviv", "Jerusalem"]:
        add(f"Which chain has the most products on promotion in {city}, as a share of what it sells?", "en",
            "promo", "medium", "ranking", f"SELECT * FROM promo_share('{city}')", {"column": "chain", "value_column": "promo_share"})
    add("לאיזו רשת יש הכי הרבה מוצרים במבצע בחיפה, באחוזים ממה שהיא מוכרת?", "he", "promo", "medium", "ranking",
        "SELECT * FROM promo_share('Haifa')", {"column": "chain", "value_column": "promo_share"})
    for chain, city in [("Shufersal", "Haifa"), ("Victory", "Tel Aviv")]:
        add(f"What share of {chain}'s products in {city} are on promotion?", "en", "promo", "medium", "scalar",
            f"SELECT promo_share FROM promo_share('{city}') WHERE chain = '{chain}'", {"column": "promo_share"})
    for chain, city in [("Rami Levy", "Haifa"), ("Shufersal", "Jerusalem"), ("Osher Ad", "Tel Aviv")]:
        add(f"How many different products does {chain} sell in {city}?", "en", "lookup", "medium", "scalar",
            f"SELECT COUNT(DISTINCT item_code) AS n FROM prices WHERE chain = '{chain}' AND city = '{city}'",
            {"column": "n"})
    for cat in ["dairy & eggs", "beverages"]:
        add(f"Across all covered cities, which chain is cheapest for {cat}?", "en", "comparison", "medium",
            "ranking", f"SELECT * FROM chain_price_index(NULL, '{cat}')", {"column": "chain", "value_column": "index"})
    for city in CITIES:
        add(f"Which chain is the most expensive in {city}?", "en", "comparison", "medium", "ranking",
            f"SELECT * FROM chain_price_index('{city}', NULL) ORDER BY \"index\" DESC", {"column": "chain", "value_column": "index"})
    add("For beverages in Tel Aviv, is Shufersal or Rami Levy cheaper?", "en", "comparison", "medium", "ranking",
        "SELECT * FROM chain_price_index('Tel Aviv', 'beverages') WHERE chain IN ('Shufersal', 'Rami Levy')",
        {"column": "chain", "value_column": "index"})

    # ---- hard: traps, cross-city, unanswerable, ambiguous -----------------------------
    for city in CITIES:
        naive = sem.run(f"SELECT chain, AVG(price) a FROM prices WHERE city = '{city}' GROUP BY chain ORDER BY a",
                        "n").rows
        idx = sem.run(f"SELECT chain FROM chain_price_index('{city}', NULL)", "i").rows
        note = (f"Trap: naive AVG(price) ranks {naive[0][0]} first; the matched index ranks {idx[0][0]} first."
                if naive and idx and naive[0][0] != idx[0][0] else "Trap: naive average over unmatched products.")
        add(f"Just compare the average shelf price at each chain in {city}: which chain is cheapest?", "en",
            "trap_simpson", "hard", "ranking", f"SELECT * FROM chain_price_index('{city}', NULL)",
            {"column": "chain", "value_column": "index"}, note)
    add("פשוט תשווה את המחיר הממוצע בכל רשת בתל אביב: איזו הכי זולה?", "he", "trap_simpson", "hard", "ranking",
        "SELECT * FROM chain_price_index('Tel Aviv', NULL)", {"column": "chain", "value_column": "index"},
        "Trap: the matched index is the defensible answer even when asked for averages.")
    # pairs with a real gap (~4-6% on matched products); Tel Aviv vs Haifa is within 0.5%, too close to ask
    for cat, a, b in [("dairy & eggs", "Tel Aviv", "Jerusalem"), ("beverages", "Haifa", "Jerusalem")]:
        add(f"Is {cat} cheaper in {a} or in {b}?", "en", "geo", "hard", "ranking", f"""
            WITH p AS (SELECT city, item_code, MIN(price) AS price FROM prices JOIN products USING (item_code)
                       WHERE category = '{cat}' AND city IN ('{a}', '{b}') GROUP BY city, item_code),
                 both_cities AS (SELECT item_code FROM p GROUP BY item_code HAVING COUNT(*) = 2)
            SELECT city, AVG(price) AS avg_matched_price, COUNT(*) AS n_products
            FROM p JOIN both_cities USING (item_code) GROUP BY city ORDER BY avg_matched_price""",
            {"column": "city", "value_column": "avg_matched_price"}, "Matched products only: those sold in both cities.")
    for q, lang in [("How did milk prices change over the last year?", "en"),
                    ("איך השתנו מחירי הלחם בחודשים האחרונים?", "he"),
                    ("What are the online-delivery prices at Shufersal?", "en"),
                    ("What time does Rami Levy in Haifa close?", "en"),
                    ("Which chain is cheapest in Eilat?", "en"),
                    ("How do Carrefour's prices compare with Shufersal's?", "en")]:
        add(q, lang, "trap_unanswerable", "hard", "decline",
            notes="Not in the data (no history, online prices, hours, uncovered city/chain).")
    for q, lang in [("Which chain is cheapest?", "en"), ("Is Rami Levy cheap?", "en"),
                    ("What's the most expensive product?", "en"), ("איזו רשת הכי זולה?", "he")]:
        add(q, lang, "trap_ambiguous", "hard", "assumption", notes="Ambiguous: must state the reading it chose.")

    for i, it in enumerate(items, 1):
        it["id"] = f"g{i:03d}"
    return items


# ---- DB / file sync ---------------------------------------------------------------
def save_drafts(items: list[dict]) -> int:
    init_db()
    with SessionLocal() as s:
        existing = {g.id for g in s.query(GoldenItem.id)}
        n = 0
        for it in items:
            if it["id"] in existing:
                continue
            s.add(GoldenItem(**{k: v for k, v in it.items() if k in FIELDS}, status="draft", source="drafted"))
            n += 1
        s.commit()
    return n


def load(statuses: tuple[str, ...] = ("approved",)) -> list[dict]:
    """Items for the benchmark: from the DB if it has any, else from the committed file."""
    init_db()
    with SessionLocal() as s:
        rows = [as_dict(g) for g in s.query(GoldenItem).filter(GoldenItem.status.in_(statuses))
                .order_by(GoldenItem.id)]
    if rows:
        return rows
    if FILE.exists():
        items = [json.loads(line) for line in FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [i for i in items if i.get("status") in statuses]
    return []


def export(include_all: bool = False) -> int:
    init_db()
    with SessionLocal() as s:
        q = s.query(GoldenItem).order_by(GoldenItem.id)
        if not include_all:
            q = q.filter(GoldenItem.status == "approved")
        rows = [as_dict(g) for g in q]
    FILE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def import_file() -> int:
    items = [json.loads(line) for line in FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    init_db()
    with SessionLocal() as s:
        for it in items:
            g = s.get(GoldenItem, it["id"]) or GoldenItem(id=it["id"])
            for k in FIELDS:
                setattr(g, k, it.get(k))
            g.updated_at = utcnow()
            s.merge(g)
        s.commit()
    return len(items)


# ---- selftest ---------------------------------------------------------------------
def perfect_answer(item: dict, sem) -> dict:
    """An answer that states exactly the reference result (what a perfect analyst would say)."""
    base = {"claims": [], "method": "", "assumptions": [], "caveats": [], "chart": None, "declined": False}
    if item["answer_type"] == "decline":
        return {**base, "headline": "The data can't answer this.", "declined": True}
    if item["answer_type"] == "assumption":
        return {**base, "headline": "Answer.", "assumptions": ["Read as X."]}
    cols, rows = scoring.reference(item, sem)
    col = item["answer_key"].get("column", cols[0])
    vals = [r[cols.index(col)] for r in rows]
    if item["answer_type"] == "scalar":
        return {**base, "headline": "Here it is.", "claims": [{"text": "v", "value": float(vals[0]), "unit": "",
                                                               "query_id": "q1"}]}
    if item["answer_type"] == "ranking":
        return {**base, "headline": f"{vals[0]} comes first."}
    k = int(item["answer_key"].get("k", 5))
    return {**base, "headline": "These: " + "; ".join(str(v) for v in vals[:k])}


def wrong_answer(item: dict, sem) -> dict:
    ans = perfect_answer(item, sem)
    at = item["answer_type"]
    if at == "decline":
        return {**ans, "declined": False, "headline": "Prices went up 5%."}
    if at == "assumption":
        return {**ans, "assumptions": []}
    if at == "scalar":
        ans["claims"][0]["value"] = ans["claims"][0]["value"] * 1.37 + 3
        return ans
    cols, rows = scoring.reference(item, sem)
    col = item["answer_key"].get("column", cols[0])
    vals = [r[cols.index(col)] for r in rows]
    if at == "ranking":
        return {**ans, "headline": f"{vals[-1]} comes first."}
    return {**ans, "headline": "These: " + "; ".join(str(v) for v in vals[-5:])}


def selftest(items: list[dict]) -> dict:
    sem = get_semantic()
    problems = []
    for it in items:
        if it["reference_sql"] and scoring.reference(it, sem) is None:
            problems.append(f"{it['id']}: reference returns nothing")
            continue
        good = scoring.score(it, perfect_answer(it, sem), [], sem)
        bad = scoring.score(it, wrong_answer(it, sem), [], sem)
        if not good.passed:
            problems.append(f"{it['id']}: perfect answer fails ({good.details})")
        if bad.passed:  # the scorer can't tell a wrong answer from the right one
            problems.append(f"{it['id']}: wrong answer passes ({bad.details})")
    return {"items": len(items), "problems": problems}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "export", "import", "selftest", "stats"])
    ap.add_argument("--all", action="store_true", help="export: include drafts/rejected; selftest: all statuses")
    args = ap.parse_args()
    if args.cmd == "build":
        items = build_drafts()
        print(f"drafted {len(items)}; new drafts: {save_drafts(items)}")
    elif args.cmd == "export":
        print(f"exported {export(args.all)} items to {FILE}")
    elif args.cmd == "import":
        print(f"imported {import_file()} items")
    elif args.cmd == "selftest":
        res = selftest(load(("approved", "draft") if args.all else ("approved",)))
        print(json.dumps(res, ensure_ascii=False, indent=2))
        raise SystemExit(1 if res["problems"] else 0)
    else:
        from collections import Counter

        items = load(("approved", "draft", "rejected"))
        for f in ("status", "difficulty", "qtype", "answer_type", "language"):
            print(f, dict(Counter(i[f] for i in items)))


if __name__ == "__main__":
    main()
