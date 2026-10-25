import streamlit as st

from components import TOUR, connect_card, tour_start
from lib import setup

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
    connect_card()
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

if st.button("▶  Take the 3-minute tour", type="primary"):
    tour_start()
