import streamlit as st

from lib import setup

setup("Architecture", "🧭")
st.markdown("## 🧭 How it's built, and why")

st.graphviz_chart(
    """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#fcfcfb", color="#c3c2b7", fontname="Segoe UI", fontsize=11];
  edge [color="#898781", fontname="Segoe UI", fontsize=9];
  gov   [label="Chains' XML feeds\\n(Price Transparency law)"];
  ing   [label="Daily ingest\\n(GitHub Actions cron)"];
  snap  [label="Parquet snapshot\\nprices · promos · stores"];
  user  [label="Visitor / synthetic\\nshoppers"];
  api   [label="FastAPI /chat\\n+ cost guard"];
  agent [label="Agent\\nOpenAI function calling"];
  evals [label="Evals\\ndeterministic 100%\\nLLM judge (sampled)"];
  db    [label="Postgres\\ntraces · evals · alerts"];
  mon   [label="Monitors\\nSLO · z-score · PSI\\nfreshness · budget"];
  ui    [label="Streamlit dashboard"];
  ci    [label="CI: offline golden set\\nblocks bad prompts"];
  gov -> ing -> snap;
  user -> api -> agent;
  agent -> snap [label="tools"];
  agent -> evals [label="trace"];
  snap -> evals [label="ground truth", style=dashed];
  evals -> db -> mon -> db;
  db -> ui; mon -> ui [label="alerts"];
  ci -> agent [label="gate", style=dashed];
}
""",
    use_container_width=True,
)

st.markdown(
    """
### Design decisions

**Exact evals first, judge second.** Prices are facts in a database, so the core quality question (is this
number right?) needs no LLM judge. Every ₪ amount in a reply is extracted and matched against the tool results the
agent actually received. It's cheap, deterministic, and runs on 100% of traffic. The judge covers what rules can't
(helpfulness, holistic groundedness) on a sample, and is itself **calibrated against human labels** (Cohen's κ).

**Golden questions, not golden answers.** Prices change daily, so storing expected answers would rot within a day.
The offline suite fixes the *questions*; expectations come from the snapshot at run time. CI runs it on every PR
and blocks merges that drop pass rates below the gates.

**Statistically honest alerts.** A 95% SLO checked on 15 samples "breaches" on one miss. Rate SLOs need n ≥ 20
and fire only when the whole 99% Wilson interval is below the SLO. Anomaly rules need both z > 3 and a ≥50% effect
size. Alert quality is tested: replaying the seeded month under 5 random seeds catches all four incidents every
time, with zero false positives.

**Everything is sliceable by what was deployed.** Each trace records agent version, model, prompt hash and data
snapshot date, so any regression can be pinned to a deploy, a model change or a data change.

**Data health is part of eval health.** A wrong answer from stale data looks the same as a hallucination to the user.
Feed age per chain is monitored against an SLA, and ingest quirks (encodings, missing city fields, flaky servers)
are surfaced, not hidden.

**A public demo with a budget.** A per-session rate limit and a daily USD budget (agent + judge spend, from token
usage) guard the live model. When either trips, a deterministic simulator answers instead. It uses the same tools,
the same evals and the same dashboards, and the reply is clearly labelled.

### Stack
Python · FastAPI · OpenAI SDK (function calling, Structured Outputs) · SQLAlchemy (SQLite locally, Neon Postgres
in production) · pandas/parquet · Streamlit + Plotly · GitHub Actions · Docker on Hugging Face Spaces.
"""
)
