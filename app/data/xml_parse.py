"""Parsers for the Israeli Price Transparency XML files.

Every chain publishes the same logical schema, but in practice the files differ:
UTF-8 vs UTF-16, with/without BOM, `SubChainID` vs `SubChainId`, `ZIPCode` vs
`ZipCode`, and city as a name (most chains) or a CBS numeric code (Shufersal).
We lower-case every tag and decode defensively so one parser handles all chains.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# CBS locality codes used by Shufersal -> canonical English city name.
CBS_CITY_CODES = {
    "5000": "Tel Aviv", "3000": "Jerusalem", "4000": "Haifa", "9000": "Beer Sheva",
    "8300": "Rishon LeZion", "7900": "Petah Tikva", "70": "Ashdod", "7400": "Netanya",
    "6100": "Bnei Brak", "8600": "Ramat Gan", "6600": "Holon", "6200": "Bat Yam",
    "8400": "Rehovot", "7100": "Ashkelon", "6400": "Herzliya", "8700": "Raanana",
    "6900": "Kfar Saba", "1200": "Modiin",
}

# Hebrew city spellings seen in the files -> canonical English city name.
HEBREW_CITY_NAMES = {
    "תל אביב": "Tel Aviv", "תל אביב יפו": "Tel Aviv", "תל אביב-יפו": "Tel Aviv", "ת\"א": "Tel Aviv",
    "ירושלים": "Jerusalem", "חיפה": "Haifa", "באר שבע": "Beer Sheva", "באר-שבע": "Beer Sheva",
    "ראשון לציון": "Rishon LeZion", "ראשון-לציון": "Rishon LeZion", "ראשל\"צ": "Rishon LeZion",
    "פתח תקווה": "Petah Tikva", "פתח תקוה": "Petah Tikva", "פתח-תקווה": "Petah Tikva",
    "אשדוד": "Ashdod", "נתניה": "Netanya", "בני ברק": "Bnei Brak", "רמת גן": "Ramat Gan",
    "חולון": "Holon", "בת ים": "Bat Yam", "רחובות": "Rehovot", "אשקלון": "Ashkelon",
    "הרצליה": "Herzliya", "רעננה": "Raanana", "כפר סבא": "Kfar Saba",
    "מודיעין": "Modiin", "מודיעין מכבים רעות": "Modiin", "ראשלצ": "Rishon LeZion",
}


def read_xml(path: Path) -> ET.Element:
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)
    root = ET.fromstring(text)
    for el in root.iter():
        el.tag = el.tag.lower()
    return root


def _txt(el: ET.Element, tag: str) -> str | None:
    child = el.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip() or None


def normalize_city(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw in CBS_CITY_CODES:
        return CBS_CITY_CODES[raw]
    cleaned = re.sub(r"\s+", " ", raw)
    return HEBREW_CITY_NAMES.get(cleaned) or HEBREW_CITY_NAMES.get(cleaned.replace("-", " "))


def infer_city(*texts: str | None) -> str | None:
    """Fallback when the City field is empty or '0': look for a known city name in free text."""
    for text in texts:
        if not text:
            continue
        for heb, eng in sorted(HEBREW_CITY_NAMES.items(), key=lambda kv: -len(kv[0])):
            if heb in text:
                return eng
    return None


def parse_stores(path: Path, chain: str) -> list[dict]:
    root = read_xml(path)
    rows = []
    for store in root.iter("store"):
        city_raw = _txt(store, "city")
        name, address = _txt(store, "storename"), _txt(store, "address")
        city = normalize_city(city_raw)
        city_source = "field" if city else None
        if not city:
            city = infer_city(name, address)
            city_source = "inferred" if city else None
        rows.append({
            "chain": chain,
            "store_id": str(int(_txt(store, "storeid") or 0)),
            "store_name": name,
            "address": address,
            "city_raw": city_raw,
            "city": city,
            "city_source": city_source,
        })
    return rows


FILE_TS = re.compile(r"-(20\d{6})-?(\d{4})")


def file_timestamp(path: Path) -> str | None:
    """Publish time from the file name, in UTC. Names carry Israel local time:
    PriceFull...-20261025-033100.xml -> 2026-10-25T00:31 (UTC, during IDT)."""
    m = FILE_TS.search(Path(path).name)
    if not m:
        return None
    d, t = m.groups()
    local = datetime(int(d[:4]), int(d[4:6]), int(d[6:]), int(t[:2]), int(t[2:]), tzinfo=ZoneInfo("Asia/Jerusalem"))
    return local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M")


def _num(s: str | None) -> float | None:
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def parse_prices(path: Path, chain: str) -> list[dict]:
    root = read_xml(path)
    store_id = str(int(_txt(root, "storeid") or 0))
    rows = []
    for item in root.iter("item"):
        price = _num(_txt(item, "itemprice"))
        code = _txt(item, "itemcode")
        if price is None or not code:
            continue
        rows.append({
            "chain": chain,
            "store_id": store_id,
            "item_code": code,
            "item_name": _txt(item, "itemname"),
            "manufacturer": _txt(item, "manufacturername") or _txt(item, "manufacturename"),
            "quantity": _num(_txt(item, "quantity")),
            "unit": _txt(item, "unitqty"),
            "is_weighted": _txt(item, "bisweighted") == "1",
            "price": price,
            "unit_price": _num(_txt(item, "unitofmeasureprice")),
            "price_updated_at": _txt(item, "priceupdatedate") or _txt(item, "priceupdatetime"),
        })
    return rows


def parse_promos(path: Path, chain: str) -> list[dict]:
    """One row per promotion; item codes as a list. Coupons and zero-price gifts are dropped
    because a shopper can't act on them from a price comparison."""
    root = read_xml(path)
    store_id = str(int(_txt(root, "storeid") or 0))
    rows = []
    for promo in root.iter("promotion"):
        if _txt(promo, "additionaliscoupon") == "1":
            continue
        items = list(promo.iter("promotionitem")) or list(promo.iter("item"))
        first = items[0] if items else promo
        discounted = _num(_txt(first, "discountedprice") or _txt(promo, "discountedprice"))
        if discounted is not None and discounted <= 0:
            continue
        club = _txt(promo, "clubid") or "0"
        rows.append({
            "chain": chain,
            "store_id": store_id,
            "promotion_id": _txt(promo, "promotionid"),
            "description": _txt(promo, "promotiondescription"),
            "start": _txt(promo, "promotionstartdatetime") or _txt(promo, "promotionstartdate"),
            "end": _txt(promo, "promotionenddatetime") or _txt(promo, "promotionenddate"),
            "min_qty": _num(_txt(first, "minqty") or _txt(promo, "minqty")),
            "discounted_price": discounted,
            "club_only": not club.strip().startswith("0"),
            "item_codes": [c for c in (_txt(i, "itemcode") for i in items) if c],
        })
    return rows
