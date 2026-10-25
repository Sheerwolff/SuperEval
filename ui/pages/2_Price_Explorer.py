import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data.catalog import get_catalog
from lib import base_layout, ink, mode, series, setup

setup("Price explorer", "🧺")
st.markdown("## 🧺 Price explorer")
st.caption("The data the agent works from: today's prices from 8 chains, as published under the Israeli "
           "Price Transparency law. Search in Hebrew or English.")

cat = get_catalog()

f1, f2, f3 = st.columns([3, 2, 2])
query = f1.text_input("Product", value="milk", placeholder="e.g. milk, במבה, coffee")
category = f2.selectbox("Category", ["All"] + cat.categories)
city = f3.selectbox("City", cat.cities, index=cat.cities.index("Tel Aviv") if "Tel Aviv" in cat.cities else 0)

hits = cat.search_products(query, limit=12, category=None if category == "All" else category) if query else []
if not hits:
    st.info("No products match. Try fewer words, or the Hebrew name.")
else:
    label = {h["item_code"]: f"{h.get('name_en') or h['name']}  ·  {h['name']}" for h in hits}
    code = st.selectbox(f"{len(hits)} matches", list(label), format_func=label.get)
    out = cat.get_prices(code, city)
    prod = out["product"] or {}
    st.markdown(f"#### {prod.get('name_en') or prod.get('item_name')}")
    st.caption(f"{prod.get('item_name')} · {prod.get('category') or 'uncategorized'} · barcode {code} · "
               f"sold by {prod.get('n_chains')} chains")
    df = cat.prices[(cat.prices.item_code == code) & (cat.prices.city == city)]
    if df.empty:
        st.warning(f"No store in {city} carries this product in the snapshot.")
    else:
        INK, S = ink(), series()
        best = df.loc[df.price.idxmin()]
        by_chain = df.groupby("chain").price.min().sort_values()
        spread = (df.price.max() / df.price.min() - 1) if df.price.min() > 0 else 0
        st.markdown(
            "<div class='tile-grid'>"
            f"<div class='tile'><div class='label'>Cheapest in {city}</div><div class='value'>₪{best.price:.2f}</div>"
            f"<div class='sub'>{best.chain} · {best.store_name}</div></div>"
            f"<div class='tile'><div class='label'>Most expensive</div><div class='value'>₪{df.price.max():.2f}</div>"
            f"<div class='sub'>{spread:.0%} above the cheapest</div></div>"
            f"<div class='tile'><div class='label'>Stores carrying it</div><div class='value'>{len(df)}</div>"
            f"<div class='sub'>{df.chain.nunique()} chains</div></div></div>",
            unsafe_allow_html=True,
        )
        a, b = st.columns(2, gap="large")
        fig = go.Figure(go.Bar(
            y=by_chain.index, x=by_chain.values, orientation="h", marker_color=S["v1"],
            marker_line_color=INK["surface"], marker_line_width=2,
            text=[f"₪{v:.2f}" for v in by_chain.values], textposition="outside",
            textfont=dict(color=INK["secondary"]), cliponaxis=False,
            hovertemplate="%{y}: ₪%{x:.2f}<extra>cheapest store</extra>",
        ))
        base_layout(fig, height=60 + 34 * len(by_chain), title="Cheapest store per chain")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=INK["grid"], tickprefix="₪", rangemode="tozero")
        fig.update_layout(hovermode="closest", showlegend=False)
        a.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Scatter(
            x=df.price, y=df.chain, mode="markers",
            marker=dict(size=11, color=S["v1"], opacity=0.85, line=dict(width=2, color=INK["surface"])),
            customdata=df.store_name, hovertemplate="%{y} · %{customdata}: ₪%{x:.2f}<extra></extra>",
        ))
        base_layout(fig2, height=60 + 34 * len(by_chain), title=f"Every store in {city}")
        fig2.update_yaxes(categoryorder="array", categoryarray=list(by_chain.index[::-1]), showgrid=False)
        fig2.update_xaxes(showgrid=True, gridcolor=INK["grid"], tickprefix="₪")
        fig2.update_layout(hovermode="closest", showlegend=False)
        b.plotly_chart(fig2, use_container_width=True)

        promos = [o for o in out["offers"] if o.get("promo")]
        if promos:
            st.markdown("**Active promotions**")
            st.dataframe(pd.DataFrame([{"chain": o["chain"], "store": o["store"], "promotion": o["promo"]["description"],
                                        "ends": o["promo"]["ends"]} for o in promos]),
                         hide_index=True, use_container_width=True)

st.divider()
st.markdown(f"#### Which chain is cheapest, by category, in {city}?")
st.caption("For every product a chain sells in this city, its cheapest price there is compared to the city's median "
           "price for the same product. Each cell is the median of that ratio: blue is cheaper than typical, "
           "red is pricier. Categories need 15+ shared products per chain to show.")


@st.cache_data(ttl=3600, show_spinner="Comparing chains…")
def price_index(city: str) -> pd.DataFrame:
    c = get_catalog()
    df = c.prices[c.prices.city == city]
    cheapest = df.groupby(["item_code", "chain"]).price.min().reset_index()
    med = cheapest.groupby("item_code").price.median().rename("median")
    cheapest = cheapest.join(med, on="item_code")
    cheapest = cheapest[(cheapest["median"] > 0) & (cheapest.groupby("item_code").chain.transform("size") >= 3)]
    cheapest["ratio"] = cheapest.price / cheapest["median"] - 1
    cheapest = cheapest.merge(c.products[["item_code", "category"]], on="item_code")
    g = cheapest.groupby(["category", "chain"]).agg(idx=("ratio", "median"), n=("ratio", "size")).reset_index()
    g = g[g.n >= 15]
    return g


idx = price_index(city)
if idx.empty:
    st.info("Not enough overlapping products in this city.")
else:
    piv = idx.pivot(index="category", columns="chain", values="idx")
    n = idx.pivot(index="category", columns="chain", values="n")
    order = piv.mean(axis=1).sort_values().index
    piv, n = piv.loc[order], n.loc[order]
    piv = piv[piv.mean().sort_values().index]
    n = n[piv.columns]
    lim = max(0.05, float(piv.abs().max().max()))
    dark = mode() == "dark"
    neutral = "#383835" if dark else "#f0efec"
    scale = [[0, "#104281" if not dark else "#5598e7"], [0.5, neutral], [1, "#b42828" if not dark else "#e66767"]]
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index, zmin=-lim, zmax=lim, colorscale=scale, xgap=2, ygap=2,
        customdata=n.values, colorbar=dict(tickformat="+.0%", title=None, thickness=10),
        hovertemplate="%{x} · %{y}: %{z:+.1%} vs. city median (n=%{customdata} products)<extra></extra>",
        text=[[f"{v:+.0%}" if pd.notna(v) else "" for v in row] for row in piv.values], texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    base_layout(fig, height=80 + 30 * len(piv))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(side="top", showline=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Table view"):
        st.dataframe((piv * 100).round(1).astype("string") + "%", use_container_width=True)
