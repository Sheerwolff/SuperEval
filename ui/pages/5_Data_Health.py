from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from app.config import settings
from app.data.catalog import get_catalog
from app.db import IngestRun, SessionLocal, utcnow
from lib import STATUS, base_layout, setup

setup("Data health", "🩺")
st.markdown("## 🩺 Data health")
st.caption("The agent can only be right if the data is. The data comes from XML files each chain must publish "
           "under the Israeli Price Transparency law, and they're messy.")

cat = get_catalog()
meta = cat.meta
now = utcnow()

st.markdown(f"#### Snapshot · fetched {meta['fetched_at'][:16].replace('T', ' ')} UTC")
cols = st.columns(len(meta["chains"]))
for col, (chain, m) in zip(cols, meta["chains"].items()):
    pub = m.get("published_at")
    age = (now - datetime.fromisoformat(pub)).total_seconds() / 3600 if pub else None
    ok = age is not None and age <= settings.freshness_sla_hours
    icon, word = ("✅", "fresh") if ok else ("⚠️", "stale")
    col.markdown(
        f"<div class='tile{'' if ok else ' bad'}'><div class='label'>{chain}</div>"
        f"<div class='value'>{'—' if age is None else f'{age:.0f}h'}</div>"
        f"<div class='sub'>{icon} {word} · {m.get('stores_selected', 0)} stores · {m.get('price_rows', 0):,} prices</div>"
        "</div>",
        unsafe_allow_html=True,
    )
st.caption(f"Age = time since the chain published its latest PriceFull file. SLA: {settings.freshness_sla_hours:.0f}h.")

st.markdown("#### Data quality findings")
dq = pd.DataFrame([{
    "chain": c,
    "stores published": m.get("stores_total"),
    "stores with no usable City field": m.get("stores_missing_city_field"),
    "share": (m.get("stores_missing_city_field") or 0) / max(1, m.get("stores_total") or 1),
} for c, m in meta["chains"].items()])
st.dataframe(dq.style.format({"share": "{:.0%}"}), use_container_width=True, hide_index=True)
st.markdown(
    """
- **Encodings differ by chain.** UTF-16 with BOM (Rami Levy, Yohananof) vs. UTF-8 (Shufersal, Victory). Tag casing
  differs too (`SubChainID` / `SubChainId`, `ZIPCode` / `ZipCode`). The parser lower-cases every tag.
- **City is a name, a numeric CBS code, or `0`.** Shufersal publishes CBS locality codes; Yohananof publishes `0` for
  every store, so its city is inferred from the store name and address.
- **Same barcode, different names.** Chains spell the same product differently, so the catalog takes the most common name.
- **Promotions** include coupons, club-only deals and ₪0 gifts. Coupons and gifts are dropped; expired promos are filtered by date.
- **Flaky sources.** Downloads time out and files arrive truncated. The ingest retries, and failures surface here.
"""
)

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
                                             line=dict(width=2, color="#fcfcfb")),
                                 hovertemplate="%{y} · %{x|%b %d %H:%M}<extra>" + name + "</extra>"))
    base_layout(fig, height=220)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Daily ingest per chain. The ✗ run is the seeded Victory outage that tripped the freshness alert.")
