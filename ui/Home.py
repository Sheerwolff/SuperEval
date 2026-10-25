import streamlit as st

from components import connect_card
from lib import setup

setup("Home", "🛒")

st.markdown(
    """
<div class="hero">
<h1>SuperEval</h1>
<p><b>Production-grade evals for an AI agent answering questions about <a href="https://www.gov.il/he/pages/cpfta_prices_regulations" target="_blank">Israeli supermarket prices</a>.</b></p>
</div>
""",
    unsafe_allow_html=True,
)
connect_card()
