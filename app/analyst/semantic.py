"""The semantic layer the analyst agent queries: DuckDB tables over the price snapshot,
documented for the model, plus canonical metric macros that encode the *right* method.

Safety: tables are loaded into memory, then file access is disabled and only single
SELECT statements are accepted, so model-written SQL can read but never write or touch disk.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import duckdb
import pandas as pd

from app.config import settings

ROW_CAP = 200  # rows returned to the model per query
TIMEOUT_S = 10.0

SCHEMA_DOC = """\
Tables (today's snapshot of prices published under the Israeli Price Transparency law; no history):

stores(chain, store_id, store_name, address, city)
  One row per store covered. city is the English city name, e.g. 'Tel Aviv', 'Haifa'.
products(item_code, name_he, name_en, brand, category, size, n_chains)
  One row per product (barcode). name_he is the published Hebrew name, name_en an English name.
  category is one of: {categories}.
  n_chains = how many chains sell it.
prices(chain, store_id, store_name, city, item_code, price)
  Shelf price in ₪ of one product at one store. A chain usually has several stores per city with
  slightly different prices.
promos(chain, store_id, city, item_code, description, discounted_price, min_qty, ends, club_only)
  Active promotions. discounted_price is the total for min_qty units ("2 for ₪10" -> 10, min_qty 2).

Chains: {chains}.
Cities: {cities}.

Canonical metrics (table macros; prefer these for comparisons, they compare like with like):
chain_price_index(city, category)
  -> chain, n_products, index, pct_vs_median
  For every product sold by >= 3 chains in the city, each chain's cheapest price is divided by the
  city's median price for that product; index is the median of those ratios (1.00 = typical).
  Pass NULL for city or category to include all.
price_spread(city, category)
  -> item_code, name_en, n_chains, min_price, max_price, cheapest_chain, priciest_chain, spread_pct
  Price gap between chains for each product in the city (cheapest store per chain).
promo_share(city)
  -> chain, n_products, n_on_promo, promo_share
  Share of a chain's products in the city that have an active promotion.

Pitfall: AVG(price) grouped by chain compares *different* products per chain and is misleading.
Compare chains on the same products (chain_price_index does this).
"""

MACROS = [
    """CREATE MACRO chain_price_index(c, cat) AS TABLE
    WITH p AS (
        SELECT pr.chain, pr.item_code, MIN(pr.price) AS price
        FROM prices pr JOIN products pd USING (item_code)
        WHERE (c IS NULL OR pr.city = c) AND (cat IS NULL OR pd.category = cat) AND pr.price > 0
        GROUP BY pr.chain, pr.item_code),
    shared AS (SELECT item_code, MEDIAN(price) AS med FROM p GROUP BY item_code HAVING COUNT(*) >= 3)
    SELECT p.chain, COUNT(*) AS n_products, ROUND(MEDIAN(p.price / s.med), 4) AS "index",
           ROUND(100 * (MEDIAN(p.price / s.med) - 1), 1) AS pct_vs_median
    FROM p JOIN shared s USING (item_code)
    GROUP BY p.chain ORDER BY "index\"""",
    """CREATE MACRO price_spread(c, cat) AS TABLE
    WITH p AS (
        SELECT pr.chain, pr.item_code, MIN(pr.price) AS price
        FROM prices pr JOIN products pd USING (item_code)
        WHERE (c IS NULL OR pr.city = c) AND (cat IS NULL OR pd.category = cat) AND pr.price > 0
        GROUP BY pr.chain, pr.item_code)
    SELECT p.item_code, pd.name_en, COUNT(*) AS n_chains, MIN(p.price) AS min_price, MAX(p.price) AS max_price,
           ARG_MIN(p.chain, p.price) AS cheapest_chain, ARG_MAX(p.chain, p.price) AS priciest_chain,
           ROUND(100 * (MAX(p.price) / MIN(p.price) - 1), 1) AS spread_pct
    FROM p JOIN products pd USING (item_code)
    GROUP BY p.item_code, pd.name_en HAVING COUNT(*) >= 2 ORDER BY spread_pct DESC""",
    """CREATE MACRO promo_share(c) AS TABLE
    WITH sold AS (SELECT DISTINCT chain, item_code FROM prices WHERE c IS NULL OR city = c),
    promo AS (SELECT DISTINCT chain, item_code FROM promos WHERE c IS NULL OR city = c)
    SELECT s.chain, COUNT(*) AS n_products, COUNT(pm.item_code) AS n_on_promo,
           ROUND(COUNT(pm.item_code) / COUNT(*), 4) AS promo_share
    FROM sold s LEFT JOIN promo pm USING (chain, item_code)
    GROUP BY s.chain ORDER BY promo_share DESC""",
]


class SQLRejected(Exception):
    """The statement isn't a single read-only SELECT."""


@dataclass
class QueryResult:
    id: str
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)  # first ROW_CAP rows
    row_count: int = 0
    error: str | None = None
    ms: float = 0.0

    def to_dict(self) -> dict:
        return {"id": self.id, "sql": self.sql, "columns": self.columns, "rows": self.rows,
                "row_count": self.row_count, "error": self.error, "ms": round(self.ms, 1)}

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


class Semantic:
    def __init__(self, snapshot_dir: Path):
        snap = Path(snapshot_dir)
        self.con = duckdb.connect()
        stores = pd.read_parquet(snap / "stores.parquet")
        stores = stores[stores.selected][["chain", "store_id", "store_name", "address", "city"]]
        products = pd.read_parquet(snap / "products.parquet")
        from app.data.catalog import _with_enrichment

        products = _with_enrichment(products)
        products["size"] = [f"{q:g} {u}" if pd.notna(q) and u else None
                            for q, u in zip(products.quantity, products.unit, strict=True)]
        products = products.rename(columns={"item_name": "name_he"})[
            ["item_code", "name_he", "name_en", "brand", "category", "size", "n_chains"]]
        prices = (pd.read_parquet(snap / "prices.parquet")
                  .drop_duplicates(["chain", "store_id", "item_code"], keep="last")
                  .merge(stores[["chain", "store_id", "store_name", "city"]], on=["chain", "store_id"]))
        prices = prices[["chain", "store_id", "store_name", "city", "item_code", "price"]]
        promos_path = snap / "promos.parquet"
        promos = pd.read_parquet(promos_path) if promos_path.exists() else pd.DataFrame()
        if not promos.empty:
            promos = promos.merge(stores[["chain", "store_id", "city"]], on=["chain", "store_id"])
            promos = promos.assign(ends=promos.end.astype(str).str[:10])[
                ["chain", "store_id", "city", "item_code", "description", "discounted_price", "min_qty", "ends",
                 "club_only"]]
        for name, df in {"stores": stores, "products": products, "prices": prices, "promos": promos}.items():
            self.con.register(f"_{name}", df)
            self.con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
            self.con.unregister(f"_{name}")
        for m in MACROS:
            self.con.execute(m)
        self.con.execute("SET enable_external_access = false")
        self.chains = sorted(stores.chain.unique())
        self.cities = sorted(stores.city.dropna().unique())
        self.categories = sorted(c for c in products.category.dropna().unique() if c)
        self._lock = threading.Lock()

    def schema(self) -> str:
        return SCHEMA_DOC.format(categories=", ".join(self.categories), chains=", ".join(self.chains),
                                 cities=", ".join(self.cities))

    def check(self, sql: str) -> None:
        try:
            statements = self.con.extract_statements(sql)
        except duckdb.Error as exc:
            raise SQLRejected(f"could not parse: {exc}") from exc
        if len(statements) != 1:
            raise SQLRejected("exactly one statement is allowed")
        if statements[0].type != duckdb.StatementType.SELECT:
            raise SQLRejected(f"only SELECT is allowed, got {statements[0].type.name}")

    def run(self, sql: str, qid: str) -> QueryResult:
        """Run one read-only query with a timeout; never raises (errors go into the result)."""
        res = QueryResult(id=qid, sql=sql.strip())
        start = time.perf_counter()
        try:
            self.check(sql)
            with self._lock:  # one DuckDB connection; cursors keep it thread-safe enough for a demo
                cur = self.con.cursor()
                timer = threading.Timer(TIMEOUT_S, cur.interrupt)
                timer.start()
                try:
                    cur.execute(sql)
                    res.columns = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                finally:
                    timer.cancel()
            res.row_count = len(rows)
            res.rows = [[_jsonable(v) for v in r] for r in rows[:ROW_CAP]]
        except SQLRejected as exc:
            res.error = f"rejected: {exc}"
        except duckdb.InterruptException:
            res.error = f"timeout after {TIMEOUT_S:.0f}s"
        except duckdb.Error as exc:
            res.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
        res.ms = (time.perf_counter() - start) * 1000
        return res


def _jsonable(v):
    if isinstance(v, float):
        return round(v, 6)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)


@cache
def get_semantic() -> Semantic:
    return Semantic(settings.snapshot_dir)
