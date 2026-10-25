from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from app.config import settings
from app.data.catalog import get_catalog
from app.db import IngestRun, SessionLocal, utcnow
from lib import STATUS, base_layout, ink, setup, tiles

setup("Data health", "🩺")
st.markdown("## 🩺 Data health")
st.caption("The agent can only be right if the data is. The data comes from XML files each chain must publish "
           "under the Israeli Price Transparency law, and they're messy.")

cat = get_catalog()
meta = cat.meta
now = utcnow()

st.markdown(f"#### Snapshot · fetched {meta['fetched_at'][:16].replace('T', ' ')} UTC"
            + (" · *committed sample*" if meta.get("sample") else ""))
items = []
for chain, m in meta["chains"].items():
    pub = m.get("published_at")
    age = (now - datetime.fromisoformat(pub)).total_seconds() / 3600 if pub else None
    ok = bool(m.get("ok")) and age is not None and age <= settings.freshness_sla_hours
    word = "✅ fresh" if ok else ("⛔ " + m["error"] if m.get("error") else "⚠️ stale")
    items.append((chain, "—" if age is None else f"{age:.0f}h",
                  f"{word} · {m.get('stores_selected', 0)} stores · {m.get('price_rows', 0):,} prices", not ok))
tiles(items)
st.caption(f"Age = time since the chain published its latest PriceFull file. SLA: {settings.freshness_sla_hours:.0f}h. "
           f"{len(cat.cities)} cities · {meta['n_products']:,} products · {meta['n_prices']:,} prices.")

st.markdown("#### Data quality findings")
dq = pd.DataFrame([{
    "chain": c,
    "stores published": m.get("stores_total"),
    "no usable City field": m.get("stores_missing_city_field"),
    "city still unresolved": m.get("stores_city_unresolved"),
} for c, m in meta["chains"].items()])
st.dataframe(dq, use_container_width=True, hide_index=True)
st.markdown(
    """
- **Encodings differ by chain.** UTF-16 with BOM (Rami Levy, Yohananof) vs. UTF-8 (Shufersal, Victory). Tag casing
  differs too (`SubChainID` / `SubChainId`, `ZIPCode` / `ZipCode`). The parser lower-cases every tag.
- **City is a name (in many spellings), a numeric CBS code, or `0`.** Every value is resolved against the official CBS
  locality table from data.gov.il (1,300+ localities). Yohananof publishes `0` for every store, so its city is
  inferred from the store name and address.
- **File names carry Israel local time**, which the ingest converts to UTC so feed age is right.
- **Same barcode, different names.** Chains spell the same product differently, so the catalog takes the most common
  name. An LLM pass then adds an English name, brand and category.
- **Promotions** include coupons, club-only deals and ₪0 gifts. Coupons and gifts are dropped; expired promos are
  filtered by date.
- **Flaky sources.** FTP servers time out and files arrive truncated. The ingest retries, and failures surface here.
"""
)

st.markdown("#### English names & categories")
st.caption("Generated once per product by an LLM (Structured Outputs, fixed category list) and cached by barcode, so "
           "the daily ingest only pays for new products. Here is a random sample to review:")
prods = cat.products
sample = prods[prods.name_en.notna()].sample(min(30, len(prods)), random_state=int(now.timestamp()) // 86400)
st.dataframe(sample[["item_name", "name_en", "brand", "category"]].rename(columns={"item_name": "published name"}),
             hide_index=True, use_container_width=True)
cats = prods.category.fillna("not enriched").value_counts()
fig = go.Figure(go.Bar(x=cats.values, y=cats.index, orientation="h", marker_color=ink()["secondary"],
                       marker_line_color=ink()["surface"], marker_line_width=2,
                       hovertemplate="%{y}: %{x} products<extra></extra>"))
base_layout(fig, height=60 + 22 * len(cats), title="Products per category")
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(showgrid=True, gridcolor=ink()["grid"])
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Ingest history")
with SessionLocal() as s:
    runs = pd.DataFrame(s.execute(select(IngestRun.created_at, IngestRun.chain, IngestRun.ok,
                                         IngestRun.price_rows)).all())
if not runs.empty:
    fig = go.Figure()
    for ok, color, name in [(True, STATUS["good"], "succeeded"), (False, STATUS["critical"], "failed")]:
        d = runs[runs.ok == ok]
        fig.add_trace(go.Scatter(x=d.created_at, y=d.chain, mode="markers", name=name,
                                 marker=dict(size=9, color=color, symbol="circle" if ok else "x",
                                             line=dict(width=2, color=ink()["surface"])),
                                 hovertemplate="%{y} · %{x|%b %d %H:%M}<extra>" + name + "</extra>"))
    base_layout(fig, height=60 + 26 * runs.chain.nunique())
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Daily ingest per chain. The ✗ runs are the seeded Victory outage that tripped the freshness alert.")
