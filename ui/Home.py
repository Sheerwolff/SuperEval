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

- **An LLM data analyst on real data.** Prices from 8 Israeli chains in 25 cities, published daily by law. It
  answers with SQL, and every number in its answer is tied to the query it came from.
- **Layer 1: evals on every answer.** Numbers checked against their query results, SQL linted for misleading
  methods (like averaging different products per chain), declines and assumptions checked, plus an LLM judge.
- **Faults caught by the right layer.** A model regression kept every number grounded, and the method checks
  caught it. A schema break stopped answers, and completion caught it.
- **Layer 2: evals on the judge.** Canaries with known verdicts, recall on known-bad answers, flip rate and
  human κ. The judge's own regression was invisible to Layer 1 and caught here.
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
<h1>SuperEval</h1>
<p><b>A production eval framework to agent answers about Israeli supermarket prices.</b></p>
<p>A live LLM data-analyst agent answers analytical questions about grocery prices over <b>real, daily-published Israeli supermarket
data</b>. Every answer is evaluated in production, and so is the LLM judge doing the evaluating.</p>
</div>
""",
    unsafe_allow_html=True,
)
connect_card()

if st.button("▶  Take the 3-minute tour", type="primary"):
    tour_start()
