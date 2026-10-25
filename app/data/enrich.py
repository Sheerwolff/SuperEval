"""Give every product an English name, a category and a brand with a small LLM.

    python -m app.data.enrich                 # enrich products in the snapshot that aren't cached yet
    python -m app.data.enrich --limit 100     # try it on a few first

Results are cached in data/reference/product_enrichment.parquet (committed), keyed by
barcode, so the daily ingest only pays for new products.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.config import ROOT, cost_usd, settings

CACHE = ROOT / "data" / "reference" / "product_enrichment.parquet"
MODEL = os.getenv("ENRICH_MODEL", "gpt-5-mini")
BATCH = 50

Category = Literal[
    "dairy & eggs", "bakery", "fruit & vegetables", "meat & fish", "frozen", "pantry & dry goods",
    "canned & jarred", "sauces & spreads", "snacks & sweets", "breakfast & cereal", "beverages",
    "alcohol", "baby", "cleaning & household", "personal care", "pet", "other",
]
CATEGORIES: list[str] = list(Category.__args__)

PROMPT = """You label Israeli supermarket products. For each item (Hebrew name as published by the chain,
manufacturer, size) return:
- name_en: a short, natural English product name a shopper would search for, including brand and size
  when present (e.g. "Tnuva 3% milk 1L", "Osem Bamba peanut snack 80g"). Expand Hebrew abbreviations.
- category: exactly one of the allowed categories.
- brand: the brand transliterated to English (e.g. "תנובה" -> "Tnuva", "עלית" -> "Elite"), or "" if
  unknown/store brand. Never Hebrew.
Well-known international products keep their real English names (e.g. "Tabasco", "Nature Valley").
If a Hebrew word is ambiguous, prefer the grocery meaning.
Keep item_code exactly as given. Return every item."""


class Item(BaseModel):
    item_code: str
    name_en: str = Field(min_length=1)
    category: Category
    brand: str


class Batch(BaseModel):
    items: list[Item]


def load_cache() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    return pd.DataFrame(columns=["item_code", "name_en", "category", "brand", "model"])


def _enrich_batch(client, rows: list[dict]) -> tuple[list[dict], float]:
    payload = json.dumps(rows, ensure_ascii=False)
    for attempt in range(3):
        try:
            resp = client.chat.completions.parse(
                model=MODEL,
                messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": payload}],
                response_format=Batch,
                reasoning_effort="minimal",
            )
            usage = resp.usage
            cost = cost_usd(MODEL, usage.prompt_tokens, usage.completion_tokens) if usage else 0.0
            wanted = {r["item_code"] for r in rows}
            items = [i.model_dump() for i in resp.choices[0].message.parsed.items if i.item_code in wanted]
            return items, cost
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return [], 0.0


def enrich(products: pd.DataFrame, limit: int | None = None, workers: int = 8) -> dict:
    from openai import OpenAI

    cache = load_cache()
    todo = products[~products.item_code.isin(cache.item_code)]
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        return {"enriched": 0, "cached": len(cache), "cost_usd": 0.0}
    client = OpenAI(api_key=settings.openai_api_key, timeout=120)
    rows = [{"item_code": r.item_code, "name_he": r.item_name, "manufacturer": r.manufacturer or "",
             "size": f"{r.quantity:g} {r.unit}" if pd.notna(r.quantity) and r.unit else ""}
            for r in todo.itertuples()]
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    out, total_cost, failed = [], 0.0, 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(_enrich_batch, client, b) for b in batches]
        for n, f in enumerate(as_completed(futures), 1):
            try:
                items, cost = f.result()
                out.extend(items)
                total_cost += cost
            except Exception as exc:
                failed += 1
                print(f"batch failed: {exc}")
            if n % 10 == 0:
                print(f"{n}/{len(batches)} batches, ${total_cost:.3f}", flush=True)
    new = pd.DataFrame(out).assign(model=MODEL)
    merged = pd.concat([cache, new], ignore_index=True).drop_duplicates("item_code", keep="last")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(CACHE, index=False)
    return {"enriched": len(new), "missing": len(rows) - len(new), "failed_batches": failed,
            "cached": len(merged), "cost_usd": round(total_cost, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(settings.snapshot_dir))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    products = pd.read_parquet(Path(args.snapshot) / "products.parquet")
    print(enrich(products, args.limit))


if __name__ == "__main__":
    main()
