"""Cut a small, committed sample from the full snapshot for tests, CI and offline runs.

    python -m app.data.make_sample          # data/snapshot -> data/sample
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from app.config import ROOT

CITIES = ["Tel Aviv", "Haifa"]
MAX_PRODUCTS = 1500


def make(src: Path = ROOT / "data" / "snapshot", dst: Path = ROOT / "data" / "sample") -> dict:
    stores = pd.read_parquet(src / "stores.parquet")
    keep_stores = stores[stores.selected & stores.city.isin(CITIES)][["chain", "store_id"]]
    prices = pd.read_parquet(src / "prices.parquet").merge(keep_stores, on=["chain", "store_id"])
    products = pd.read_parquet(src / "products.parquet")
    top = (prices.groupby("item_code").chain.nunique().sort_values(ascending=False).head(MAX_PRODUCTS).index)
    products = products[products.item_code.isin(top)]
    prices = prices[prices.item_code.isin(top)]
    promos = pd.read_parquet(src / "promos.parquet").merge(keep_stores, on=["chain", "store_id"])
    promos = promos[promos.item_code.isin(top)]
    stores = stores.assign(selected=stores.selected & stores.city.isin(CITIES))
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    stores.to_parquet(dst / "stores.parquet", index=False)
    products.to_parquet(dst / "products.parquet", index=False)
    prices.to_parquet(dst / "prices.parquet", index=False)
    promos.to_parquet(dst / "promos.parquet", index=False)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    meta.update(cities=CITIES, n_products=len(products), n_prices=len(prices), n_promos=len(promos), sample=True)
    for chain, m in meta["chains"].items():
        m["price_rows"] = int((prices.chain == chain).sum())
        m["stores_selected"] = int(((stores.chain == chain) & stores.selected).sum())
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"products": len(products), "prices": len(prices), "promos": len(promos)}


if __name__ == "__main__":
    print(make())
