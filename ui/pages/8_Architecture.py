import streamlit as st

from lib import ink, setup

setup("Architecture", "🧭")
st.markdown("## 🧭 How it's built, and why")

INK = ink()
st.graphviz_chart(
    f"""
digraph G {{
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="{INK['surface']}", color="{INK['axis']}",
        fontcolor="{INK['primary']}", fontname="Segoe UI", fontsize=11];
  edge [color="{INK['muted']}", fontcolor="{INK['muted']}", fontname="Segoe UI", fontsize=9];
  gov   [label="8 chains' XML feeds\\n(Price Transparency law)"];
  ing   [label="Daily ingest + LLM enrichment\\n(GitHub Actions)"];
  snap  [label="Snapshot (release asset)\\nprices · promos · stores\\nEnglish names · categories"];
  user  [label="Visitor / synthetic\\nshoppers"];
  api   [label="FastAPI /chat/stream\\n+ cost guard"];
  agent [label="Agent\\nOpenAI function calling"];
  l1    [label="Layer 1 evals (agent)\\ndeterministic 100%\\n+ LLM judge"];
  l2    [label="Layer 2 evals (judge)\\nground-truth recall · canaries\\nflip rate · κ · bias"];
  db    [label="Postgres\\ntraces · evals · alerts"];
  mon   [label="Monitors\\nWilson SLOs · z-score · PSI\\nfreshness · budget"];
  ui    [label="Streamlit dashboard"];
  ci    [label="CI: golden set\\nblocks bad prompts"];
  gov -> ing -> snap;
  user -> api -> agent;
  agent -> snap [label="tools"];
  agent -> l1 [label="trace"];
  snap -> l1 [label="ground truth", style=dashed];
  l1 -> l2 [label="judge verdicts vs.\\ndeterministic truth"];
  l1 -> db; l2 -> db; db -> mon -> db;
  db -> ui; mon -> ui [label="alerts"];
  ci -> agent [label="gate", style=dashed];
}}
""",
    use_container_width=True,
)

st.markdown(
    """
### Design decisions

**Exact evals first, judge second.** Prices are facts in a database, so the core quality question (is this
number right?) needs no LLM judge. Every ₪ amount in a reply is extracted and matched against the tool results the
agent actually received. It's cheap, deterministic, and runs on 100% of traffic. The judge covers what rules can't:
helpfulness, holistic groundedness, and whether the agent priced the product that was actually asked for.

**Two layers: evaluate the evaluator.** An LLM judge is a model, and it can regress too (a provider model update, a
rubric edit). A judge that silently goes lenient makes the agent look *better* while it isn't, so Layer 2 watches the
judge:
- its **recall on known-bad answers**, using the deterministic checks as free, always-on labels
- **daily canaries** with known verdicts
- **flip rate** when a trace is judged twice
- **bias checks** for length and language
- **human κ** from labels

The seeded month includes exactly that incident. Layer 1's judge scores improved; only Layer 2 caught it.

**Golden questions, not golden answers.** Prices change daily, so storing expected answers would rot within a day.
The offline suite fixes the *questions*; expectations come from the snapshot at run time. CI runs it on every PR
and blocks merges that drop pass rates below the gates.

**Statistically honest alerts.** A 95% SLO checked on 15 samples "breaches" on one miss. Rate SLOs need a minimum
sample and fire only when the whole 99% Wilson interval is past the SLO. Anomaly rules need both z > 3 and a ≥50%
effect size. Alert quality is tested: `scripts/alert_quality.py` replays the seeded month under several random
seeds and checks that every incident is caught and nothing else fires.

**Everything is sliceable by what changed.** Each trace records agent version, model, prompt hash and data snapshot
date; each judge verdict records the judge model and rubric hash. Any regression can be pinned to an agent deploy, a
judge change or a data change.

**Data health is part of eval health.** A wrong answer from stale data looks the same as a hallucination to the user.
Feed age per chain is monitored against an SLA, and ingest quirks (encodings, city codes, flaky servers) are
surfaced, not hidden.

**A public demo with a budget.** A per-session rate limit and a daily USD budget (agent + judge spend, from token
usage) guard the live model. When either trips, a deterministic simulator answers instead. It uses the same tools,
the same evals and the same dashboards, and the reply is clearly labelled.

### Stack
Python · FastAPI (SSE) · OpenAI SDK (function calling, Structured Outputs) · SQLAlchemy (SQLite locally, Neon
Postgres in production) · pandas/parquet · Streamlit + Plotly · GitHub Actions · Docker on Hugging Face Spaces.
"""
)
