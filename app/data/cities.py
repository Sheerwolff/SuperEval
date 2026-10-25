"""City normalization against the official CBS locality table (data/reference/localities.csv,
from data.gov.il). Chains publish city as a Hebrew name (many spellings), a CBS code, or 0."""
from __future__ import annotations

import csv
import re
from functools import cache
from pathlib import Path

LOCALITIES = Path(__file__).resolve().parents[2] / "data" / "reference" / "localities.csv"

# Friendlier English than the official transliteration ("Rishon Leziyyon", "Be'Er Sheva").
DISPLAY_EN = {
    "5000": "Tel Aviv", "3000": "Jerusalem", "4000": "Haifa", "9000": "Beer Sheva", "8300": "Rishon LeZion",
    "7900": "Petah Tikva", "70": "Ashdod", "7400": "Netanya", "6100": "Bnei Brak", "8600": "Ramat Gan",
    "6600": "Holon", "6200": "Bat Yam", "8400": "Rehovot", "7100": "Ashkelon", "6400": "Herzliya",
    "8700": "Raanana", "6900": "Kfar Saba", "1200": "Modiin", "2640": "Rosh HaAyin", "2610": "Beit Shemesh",
    "2650": "Ramat HaSharon", "6500": "Hadera", "7000": "Lod", "8500": "Ramla", "2800": "Kiryat Shmona",
    "7700": "Afula", "1139": "Karmiel", "2600": "Eilat", "9700": "Hod HaSharon", "7300": "Nazareth",
    "3797": "Modiin Illit", "3780": "Beitar Illit", "1031": "Sderot", "2630": "Kiryat Gat",
    "6300": "Givatayim", "2660": "Yavne", "7200": "Nes Ziona", "1034": "Kiryat Malakhi",
    "6800": "Kiryat Ata", "9800": "Binyamina", "2400": "Or Yehuda", "8800": "Tiberias", "2100": "Tirat Carmel",
    "1015": "Mevaseret Zion", "3616": "Maale Adumim", "9100": "Nahariya", "6700": "Kiryat Bialik",
    "2300": "Kiryat Tivon", "2550": "Gedera", "166": "Gan Yavne", "1309": "Elad",
}
ALIASES = {
    'ת"א': "5000", "תל אביב": "5000", "תל-אביב": "5000", "תל אביב יפו": "5000", "ת א": "5000",
    'ראשל"צ': "8300", "ראשלצ": "8300", 'פ"ת': "7900", "פתח תקוה": "7900", 'ב"ש': "9000", "באר-שבע": "9000",
    "מודיעין": "1200", "מודיעין מכבים רעות": "1200", 'ק. גת': "2630", "בית-שמש": "2610",
}


def _clean(text: str) -> str:
    text = text.replace("״", '"').replace("׳", "'").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


@cache
def _table() -> tuple[dict, dict]:
    by_code, by_name = {}, {}
    with LOCALITIES.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["code"]
            by_code[code] = {"code": code, "he": row["name_he"],
                             "en": DISPLAY_EN.get(code) or row["name_en"] or row["name_he"]}
            by_name[_clean(row["name_he"])] = code
            # "תל אביב - יפו" -> also "תל אביב"
            by_name.setdefault(_clean(row["name_he"].split(" - ")[0]), code)
    for alias, code in ALIASES.items():
        by_name[_clean(alias)] = code
    return by_code, by_name


def lookup(raw: str | None) -> dict | None:
    """CBS code or Hebrew name (any common spelling) -> {code, he, en}."""
    if not raw:
        return None
    by_code, by_name = _table()
    raw = raw.strip()
    if raw.isdigit():
        return by_code.get(str(int(raw)))
    code = by_name.get(_clean(raw))
    return by_code.get(code) if code else None


def infer(*texts: str | None) -> dict | None:
    """City mentioned anywhere in free text (store name / address); longest name wins."""
    by_code, by_name = _table()
    names = _names_by_length()
    for text in texts:
        if not text:
            continue
        cleaned = f" {_clean(text)} "
        for name in names:
            if f" {name} " in cleaned or f" ב{name} " in cleaned:
                return by_code[by_name[name]]
    return None


@cache
def _names_by_length() -> list[str]:
    _, by_name = _table()
    return sorted((n for n in by_name if len(n) >= 3), key=len, reverse=True)


def english(code_or_name: str) -> str | None:
    hit = lookup(code_or_name)
    return hit["en"] if hit else None


def hebrew_for(english_name: str) -> str | None:
    by_code, _ = _table()
    for row in by_code.values():
        if row["en"] == english_name:
            return row["he"]
    return None
