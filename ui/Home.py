import streamlit as st

from components import TOUR, connect_card, tour_start
from lib import fmt, frames, kpis, setup, tiles

setup("Home", "🛒")

wrap = st.session_state.get("tour_step") == len(TOUR) - 1

if wrap:
    st.markdown("<div class='hero'><h1>That's the tour 🎉</h1></div>", unsafe_allow_html=True)
    st.markdown(
        """
In about three minutes you saw:

- **An agent on real data.** Prices from 8 Israeli chains across 25 cities, published daily by law, with messy
  XML normalized and every product given English names and categories.
- **Layer 1: evals on the agent.** Every ₪ figure is checked exactly against the database. Grounding, tool use,
  language, scope and product relevance are checked too, and an LLM judge covers the rest.
- **A bad deploy caught.** Statistically gated SLOs fired on v2 within seconds and resolved on rollback.
- **Layer 2: evals on the judge.** Canaries, recall on known-bad answers, flip rate and human κ. The judge's
  own regression was invisible to Layer 1 and caught here.
"""
    )
    connect_card("Let's connect", "Thanks for taking the tour. I'd be glad to walk you through the design "
                 "decisions, or talk about what you're building.")
    a, b = st.columns(2)
    if a.button("Explore on my own", use_container_width=True, type="primary"):
        st.session_state.tour_step = None
        st.rerun()
    if b.button("Restart the tour", use_container_width=True):
        tour_start()
    st.stop()

st.markdown(
    """
<div class="hero">
<h1>PriceWatch Evals</h1>
<p>A live LLM agent answers grocery price questions over <b>real, daily-published Israeli supermarket
data</b>. Every answer is evaluated in production, and so is the LLM judge doing the evaluating.</p>
</div>
""",
    unsafe_allow_html=True,
)
connect_card()

c1, c2 = st.columns([1, 2], vertical_alignment="center")
if c1.button("▶  Take the 3-minute tour", type="primary", use_container_width=True):
    tour_start()
c2.caption("Ask the agent → break production → investigate → see who watches the judge. "
           "Or use the sidebar to explore freely.")

traces, evals = frames(days=1)
metrics = ["price_accuracy", "grounded_prices", "product_relevance", "language_match", "judge_helpfulness",
           "latency_p95_ms"]
vals = kpis(evals, traces, metrics)
labels = {"price_accuracy": "Price accuracy", "grounded_prices": "Grounded prices",
          "product_relevance": "Right product", "language_match": "Language match",
          "judge_helpfulness": "Judge helpfulness", "latency_p95_ms": "Latency p95"}
st.markdown("##### Last 24 hours")
tiles([("Requests", f"{len(traces):,}", "all sources", False)]
      + [(labels[m], fmt(m, vals[m][0]), f"n = {vals[m][1]:,}", False) for m in metrics])

st.write("")
st.markdown("#### Two layers of evaluation")
a, b = st.columns(2, gap="large")
with a:
    st.markdown("##### Layer 1 · the agent")
    st.markdown(
        """
| Signal | How |
|---|---|
| Every quoted ₪ price matches a tool result | deterministic, 100% |
| No price without a lookup; no chains we don't cover | deterministic |
| Priced the product that was asked for | deterministic + judge |
| Right tool, user's language, off-topic/injection declined | deterministic |
| Helpfulness and groundedness (1–5) | LLM judge |
| Latency, cost, errors, input drift, data freshness | from traces and ingests |
"""
    )
with b:
    st.markdown("##### Layer 2 · the judge")
    st.markdown(
        """
| Signal | How |
|---|---|
| Fails answers we *know* are bad | recall vs. deterministic ground truth |
| Still right on known cases | daily canary set |
| Same verdict when asked twice | flip rate |
| Agrees with humans | Cohen's κ from labels |
| Rewards length? favours a language? | bias checks |
| Errors, latency, cost | judge ops |
"""
    )
st.caption("The 30-day history is seeded by a simulator with five scripted incidents; the 'backfill' source tag "
           "marks it everywhere. Live chats and synthetic bursts are added on top in real time.")
