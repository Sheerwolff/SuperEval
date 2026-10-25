"""Download a fresh price snapshot from the chains and write it as parquet.

    python -m app.data.ingest --out data/snapshot

Scope: 8 chains; the cities where at least MIN_CHAINS_PER_CITY of them have stores
(about 25); up to 2 stores per chain per city; products sold by at least two chains
(those are the ones worth comparing). Two phases: store lists first (to pick cities
across all chains), then price and promo files for the picked stores only.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.data.xml_parse import file_timestamp, parse_prices, parse_promos, parse_stores

log = logging.getLogger("ingest")

# scraper enum name -> display name used everywhere in the app
CHAINS = {
    "SHUFERSAL": "Shufersal",
    "RAMI_LEVY": "Rami Levy",
    "VICTORY_NEW_SOURCE": "Victory",
    "YOHANANOF": "Yohananof",
    "OSHER_AD": "Osher Ad",
    "TIV_TAAM": "Tiv Taam",
    "YELLOW": "Yellow",
    "HAZI_HINAM": "Hazi Hinam",
}
MIN_CHAINS_PER_CITY = 5
MAX_CITIES = 25
STORES_PER_CITY = 2
MAX_PRODUCTS = 6000


def _scrape(scraper: str, file_types: list[str], dump_dir: Path, regex: str | None = None, limit=None) -> Path:
    from il_supermarket_scarper import ScarpingTask

    task = ScarpingTask(
        enabled_scrapers=[scraper],
        files_types=file_types,
        file_name_regex=regex,
        multiprocessing=1,
        output_configuration={"output_mode": "disk", "base_storage_path": str(dump_dir)},
        status_configuration={"database_type": "json", "base_path": str(dump_dir / "status")},
    )
    task.start(limit=limit)
    task.join()
    return dump_dir


def _files(dump_dir: Path, prefix: str) -> list[Path]:
    return sorted(p for p in dump_dir.rglob(f"{prefix}*.xml"))


def pick_cities(stores: pd.DataFrame) -> list[str]:
    """Cities where enough chains are present to make comparisons meaningful."""
    reach = stores.dropna(subset=["city"]).groupby("city").chain.nunique()
    reach = reach[reach >= MIN_CHAINS_PER_CITY].sort_values(ascending=False)
    return sorted(reach.head(MAX_CITIES).index)


def select_stores(stores: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    picked = (
        stores[stores.city.isin(cities)]
        .sort_values(["chain", "city", "store_id"], key=lambda s: s.astype(str))
        .groupby(["chain", "city"])
        .head(STORES_PER_CITY)
    )
    return picked.reset_index(drop=True)


def run(out_dir: Path) -> dict:
    from il_supermarket_scarper import FileTypesFilters

    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    work = Path(tempfile.mkdtemp(prefix="pricewatch-"))
    all_stores, price_frames, all_promos, chain_meta = [], [], [], {}
    try:
        # phase 1: store lists for every chain
        for scraper, chain in CHAINS.items():
            chain_dir = work / scraper
            chain_dir.mkdir()
            try:
                _scrape(scraper, [FileTypesFilters.STORE_FILE.name], chain_dir, limit=1)
                store_files = _files(chain_dir, "Stores")
                if not store_files:
                    raise RuntimeError("no Stores file published")
                stores = pd.DataFrame(parse_stores(store_files[-1], chain))
            except Exception as exc:
                log.error("%s: store list failed: %s", chain, exc)
                chain_meta[chain] = {"ok": False, "error": f"store list: {exc}"}
                continue
            all_stores.append(stores)
            chain_meta[chain] = {
                "ok": False,
                "stores_total": len(stores),
                "stores_missing_city_field": int((stores.city_source != "field").sum()),
                "stores_city_unresolved": int(stores.city.isna().sum()),
            }
        cities = pick_cities(pd.concat(all_stores, ignore_index=True))
        log.info("cities (%d): %s", len(cities), cities)

        # phase 2: prices and promos for the picked stores
        for scraper, chain in CHAINS.items():
            if chain not in chain_meta or "error" in chain_meta[chain]:
                continue
            stores = next(s for s in all_stores if s.chain.iat[0] == chain)
            picked = select_stores(stores, cities)
            if picked.empty:
                chain_meta[chain]["error"] = "no stores in the selected cities"
                continue
            chain_dir = work / scraper
            ids = "|".join(sorted(set(picked.store_id)))
            try:
                _scrape(scraper, [FileTypesFilters.PRICE_FULL_FILE.name, FileTypesFilters.PROMO_FULL_FILE.name],
                        chain_dir, regex=rf"(PriceFull|PromoFull)\d+-\d+-0*({ids})-")
            except Exception as exc:
                log.error("%s: price download failed: %s", chain, exc)
                chain_meta[chain]["error"] = f"prices: {exc}"
                continue
            price_files = _files(chain_dir, "PriceFull")
            frames = [pd.DataFrame(parse_prices(f, chain)) for f in price_files]
            prices = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
            published = max((t for t in map(file_timestamp, price_files) if t), default=None)
            promos = [r for f in _files(chain_dir, "PromoFull") for r in parse_promos(f, chain)]
            got = set(prices.store_id) if len(prices) else set()
            picked = picked[picked.store_id.isin(got)]
            prices = prices[prices.store_id.isin(set(picked.store_id))] if len(prices) else prices
            log.info("%s: %d stores, %d price rows, %d promos", chain, len(picked), len(prices), len(promos))
            all_stores[[s.chain.iat[0] for s in all_stores].index(chain)] = stores.assign(
                selected=stores.store_id.isin(picked.store_id))
            price_frames.append(prices)
            all_promos.extend(promos)
            shutil.rmtree(chain_dir, ignore_errors=True)  # keep disk use flat
            chain_meta[chain].update({
                "ok": len(prices) > 0,
                "stores_selected": len(picked),
                "price_rows_raw": len(prices),
                "price_files": len(price_files),
                "published_at": published,
            })
    finally:
        shutil.rmtree(work, ignore_errors=True)

    all_stores = [s if "selected" in s else s.assign(selected=False) for s in all_stores]
    stores_df = pd.concat(all_stores, ignore_index=True)
    prices_df = pd.concat(price_frames, ignore_index=True)
    prices_df = prices_df.merge(
        stores_df[stores_df.selected][["chain", "store_id"]], on=["chain", "store_id"], how="inner"
    )

    # keep products sold by >= 2 chains, most widely stocked first
    reach = prices_df.groupby("item_code").agg(chains=("chain", "nunique"), stores=("store_id", "count"))
    keep = reach[reach.chains >= 2].sort_values(["chains", "stores"], ascending=False).head(MAX_PRODUCTS).index
    prices_df = prices_df[prices_df.item_code.isin(keep)]

    # product catalog: the most common name per barcode (chains spell things differently)
    products = (
        prices_df.groupby("item_code")
        .agg(
            item_name=("item_name", lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0]),
            manufacturer=("manufacturer", lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else None),
            quantity=("quantity", "median"),
            unit=("unit", lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else None),
            is_weighted=("is_weighted", "max"),
            n_chains=("chain", "nunique"),
        )
        .reset_index()
    )
    # a store can appear in more than one PriceFull file; keep its most recent price per item
    prices_df = (
        prices_df.sort_values("price_updated_at")
        .drop_duplicates(["chain", "store_id", "item_code"], keep="last")
        [["chain", "store_id", "item_code", "price", "unit_price", "price_updated_at"]]
    )

    promos_df = pd.DataFrame(all_promos)
    if not promos_df.empty:
        promos_df = promos_df.explode("item_codes").rename(columns={"item_codes": "item_code"})
        promos_df = promos_df[promos_df.item_code.isin(keep)]
        today = fetched_at[:10]
        promos_df = promos_df[promos_df.end.fillna("9999").str[:10] >= today]
        promos_df = promos_df.merge(
            stores_df[stores_df.selected][["chain", "store_id"]], on=["chain", "store_id"], how="inner"
        ).drop_duplicates(["chain", "store_id", "promotion_id", "item_code"])

    out_dir.mkdir(parents=True, exist_ok=True)
    stores_df.to_parquet(out_dir / "stores.parquet", index=False)
    products.to_parquet(out_dir / "products.parquet", index=False)
    prices_df.to_parquet(out_dir / "prices.parquet", index=False)
    promos_df.to_parquet(out_dir / "promos.parquet", index=False)

    for chain in chain_meta:
        sub = prices_df[prices_df.chain == chain]
        chain_meta[chain]["price_rows"] = len(sub)
        chain_meta[chain]["latest_price_update"] = sub.price_updated_at.max() if len(sub) else None
    meta = {
        "fetched_at": fetched_at,
        "chains": chain_meta,
        "cities": cities,
        "n_products": len(products),
        "n_prices": len(prices_df),
        "n_promos": len(promos_df),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/snapshot")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("Logger", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(json.dumps(run(Path(args.out)), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
