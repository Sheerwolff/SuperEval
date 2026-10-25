"""Read-only access to the price snapshot. These functions back the agent's tools
and are also the ground truth the deterministic evals check answers against."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from app.config import settings


class Catalog:
    def __init__(self, snapshot_dir: Path):
        self.dir = Path(snapshot_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.stores = pd.read_parquet(self.dir / "stores.parquet")
        self.stores = self.stores[self.stores.selected].copy()
        self.products = pd.read_parquet(self.dir / "products.parquet")
        prices = pd.read_parquet(self.dir / "prices.parquet").drop_duplicates(
            ["chain", "store_id", "item_code"], keep="last"
        )
        promos_path = self.dir / "promos.parquet"
        self.promos = pd.read_parquet(promos_path) if promos_path.exists() else pd.DataFrame()
        self.prices = prices.merge(
            self.stores[["chain", "store_id", "store_name", "city"]], on=["chain", "store_id"], how="left"
        )
        for col in ("manufacturer", "unit"):
            self.products[col] = self.products[col].where(~self.products[col].isin(UNKNOWN), None)
        self._norm_names = self.products.item_name.fillna("").map(_norm)
        self._by_code = self.products.set_index("item_code")

    # ---- metadata -------------------------------------------------------
    @property
    def chains(self) -> list[str]:
        return sorted(self.prices.chain.unique())

    @property
    def cities(self) -> list[str]:
        return sorted(self.stores.city.dropna().unique())

    @property
    def snapshot_date(self) -> str:
        return self.meta["fetched_at"][:10]

    def product(self, item_code: str) -> dict | None:
        if item_code not in self._by_code.index:
            return None
        row = self._by_code.loc[item_code]
        return {"item_code": item_code, **{k: _py(v) for k, v in row.items()}}

    # ---- tools ----------------------------------------------------------
    def search_products(self, query: str, limit: int = 8) -> list[dict]:
        """Every query token must appear in the name; rank by "name starts with the query",
        then by how many chains sell it (comparable products first), then shorter names.
        Falls back to fuzzy matching for typos."""
        tokens = _norm(query).split()
        if not tokens:
            return []
        mask = pd.Series(True, index=self.products.index)
        for tok in tokens:
            mask &= self._norm_names.str.contains(tok, regex=False)
        cand = self.products[mask]
        if not cand.empty:
            names = self._norm_names[mask]
            cand = cand.assign(
                _starts=~names.str.startswith(tokens[0]),
                _len=names.str.len(),
                score=100.0,
            ).sort_values(["_starts", "n_chains", "_len"], ascending=[True, False, True])
        else:
            hits = process.extract(_norm(query), self._norm_names.tolist(), scorer=fuzz.token_set_ratio, limit=limit * 3)
            idx = [i for _n, sc, i in hits if sc >= 70]
            cand = self.products.iloc[idx].assign(score=[sc for _n, sc, _i in hits if sc >= 70])
        out, seen = [], set()
        for row in cand.itertuples():
            if row.item_name in seen:
                continue
            seen.add(row.item_name)
            out.append({
                "item_code": row.item_code,
                "name": row.item_name,
                "manufacturer": _py(row.manufacturer),
                "size": _size(row),
                "sold_by_chains": int(row.n_chains),
                "match_score": round(float(row.score), 1),
            })
            if len(out) >= limit:
                break
        return out

    def get_prices(self, item_code: str, city: str | None = None, chain: str | None = None) -> dict:
        df = self.prices[self.prices.item_code == item_code]
        if city:
            df = df[df.city == city]
        if chain:
            df = df[df.chain == chain]
        product = self.product(item_code)
        if df.empty:
            return {"product": product, "offers": [], "note": "no prices found for this filter"}
        promos = self._promos_for(item_code)
        offers = []
        for r in df.sort_values("price").itertuples():
            promo = promos.get((r.chain, r.store_id))
            offers.append({
                "chain": r.chain,
                "store": r.store_name,
                "city": r.city,
                "price": round(float(r.price), 2),
                "promo": promo,
                "price_updated_at": str(r.price_updated_at)[:10],
            })
        cheapest_by_chain = df.groupby("chain").price.min().round(2).sort_values().to_dict()
        return {
            "product": product,
            "cheapest_by_chain": cheapest_by_chain,
            "offers": offers[:12],
            "n_offers": len(offers),
            "snapshot_date": self.snapshot_date,
        }

    def compare_basket(self, items: list[dict], city: str) -> dict:
        """items: [{"item_code": str, "qty": number}] -> cheapest store per chain in a city."""
        codes = {i["item_code"]: float(i.get("qty", 1) or 1) for i in items}
        df = self.prices[(self.prices.city == city) & (self.prices.item_code.isin(codes))]
        if df.empty:
            return {"city": city, "results": [], "note": "no stores in this city carry these items"}
        df = df.assign(line_total=df.price * df.item_code.map(codes))
        per_store = (
            df.groupby(["chain", "store_id", "store_name"])
            .agg(total=("line_total", "sum"), found=("item_code", "nunique"))
            .reset_index()
        )
        per_store["missing"] = len(codes) - per_store.found
        best = per_store.sort_values(["missing", "total"]).groupby("chain").head(1).sort_values(["missing", "total"])
        results = []
        for r in best.itertuples():
            have = set(df[(df.chain == r.chain) & (df.store_id == r.store_id)].item_code)
            results.append({
                "chain": r.chain,
                "store": r.store_name,
                "total": round(float(r.total), 2),
                "items_found": int(r.found),
                "missing_items": sorted(set(codes) - have),
            })
        return {"city": city, "n_items": len(codes), "results": results, "snapshot_date": self.snapshot_date}

    def list_stores(self, city: str | None = None, chain: str | None = None) -> list[dict]:
        df = self.stores
        if city:
            df = df[df.city == city]
        if chain:
            df = df[df.chain == chain]
        return [
            {"chain": r.chain, "store": r.store_name, "address": _py(r.address), "city": r.city}
            for r in df.itertuples()
        ]

    def _promos_for(self, item_code: str) -> dict:
        if self.promos.empty or "item_code" not in self.promos:
            return {}
        p = self.promos[self.promos.item_code == item_code]
        return {
            (r.chain, r.store_id): {
                "description": r.description,
                "min_qty": _py(r.min_qty),
                "discounted_price": _py(r.discounted_price),
                "ends": str(r.end)[:10] if r.end else None,
            }
            for r in p.itertuples()
        }


UNKNOWN = {"לא ידוע", "unknown", "כללי", "-", "0"}


def _norm(text: str) -> str:
    text = re.sub(r"[\"'״׳*/\()\[\],.+-]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def _py(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def _size(row) -> str | None:
    q, u = _py(row.quantity), _py(row.unit)
    if q is None or u is None:
        return None
    return f"{q:g} {u}"


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    return Catalog(settings.snapshot_dir)
