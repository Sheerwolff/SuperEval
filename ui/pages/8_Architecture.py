import streamlit as st

from lib import setup

setup("Architecture", "🧭")
st.markdown("## 🧭 How it's built, and why")
st.caption("Purpose → failure modes → KPIs → evals. Every check on this site exists because of a specific way an "
           "analysis can go wrong.")

st.graphviz_chart(
    """
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fillcolor="#ffe699", color="#bf980d", penwidth=1.5,
        fontcolor="#061a2d", fontname="Segoe UI", fontsize=11, margin="0.2,0.08"];
  edge [color="#e97816", fontcolor="#bb6011", fontname="Segoe UI", fontsize=9];
  gov   [label="8 chains' XML feeds\\n(Price Transparency law)"];
  sem   [label="Semantic layer (DuckDB, read-only)\\nstores · products · prices · promos\\ncanonical metric macros"];
  user  [label="Visitor / synthetic\\nquestions"];
  api   [label="FastAPI /chat/stream\\n+ cost guard"];
  agent [label="Analyst agent\\nrun_sql → submit_answer\\n(claims cite query ids)"];
  l1    [label="Layer 1 · the analyst\\ngrounding re-execution · SQL lint\\ncanonical cross-check · declines\\n100% of traffic"];
  judge [label="LLM judge (gpt-5)\\n30% sample"];
  l2    [label="Layer 2 · the judge\\nrecall on known-bad · canaries\\nflip rate · κ vs owner · bias"];
  hum   [label="Golden set page\\nhuman judgments\\n→ override rate, κ"];
  gold  [label="Golden set (80 Qs)\\nreference SQL, human-approved\\nCI gate on PRs"];
  db    [label="Postgres\\ntraces · evals · alerts"];
  mon   [label="Monitors\\nWilson SLOs · z-score · PSI\\nfreshness · budget"];
  gov -> sem;
  user -> api -> agent;
  agent -> sem [label="SQL"];
  agent -> l1 [label="trace"];
  sem -> l1 [label="re-run cited queries", style=dashed];
  agent -> judge;
  l1 -> l2 [label="truth labels"];
  judge -> l2;
  hum -> l2 [label="owner labels"];
  hum -> gold [label="promote", style=dashed];
  gold -> agent [label="gate", style=dashed];
  l1 -> db; l2 -> db; db -> mon -> db;
}
""",
    use_container_width=True,
)

st.markdown(
    """
### Purpose
**Answer analytical questions about grocery prices in Israel: correctly, honestly, and with the method shown.**
Typical questions:
- "Which chain is really cheapest in Haifa?"
- "Which products have the biggest gap between chains?"
- "How common are promotions at each chain?"

The answer is a plain-language headline for consumers, with the method and SQL one click away for analysts.

### Failure modes → evals
| Failure mode | How it's caught |
|---|---|
| **Ungrounded number** (hallucinated or mis-copied) | **Grounding**: every claim cites a query id; the query is re-run and the value must be in its result |
| **Misleading method** (comparing chains on different baskets: Simpson's trap; wrong promo denominator) | **SQL method lint** + **canonical cross-check** against the matched-basket macro + judge method score |
| **Wrong SQL** | SQL health (errors, empty results), canonical cross-check |
| **Answering the unanswerable** (price history, online prices) | Decline check against the question classifier |
| **Silent ambiguity** | An ambiguous question must come with a stated assumption |
| **Misleading chart** | Chart lint: columns exist, bars start at zero, the type fits the data |
| **Wrong language, slow, expensive** | Language match, latency SLO, cost per completed task |

### KPIs (separate, no composite score)
- **Correctness (offline):** golden-set accuracy of the production prompt.
- **Grounding (online):** share of answers whose every number re-checks.
- **Human override rate:** share of human-reviewed answers judged wrong.
- **Cost per completed task:** agent + judge spend ÷ completed answers.

A composite would hide trade-offs, like a cheaper model that's less grounded. Secondary metrics come sliced by
question type and language.

### Design decisions
**Golden questions, not golden answers.** Prices change daily. The benchmark fixes 80 questions with *reference
SQL*, and the expected answer is computed from today's snapshot. Items are drafted, then human-reviewed; only
approved items gate CI. A selftest proves every scorer passes a perfect answer and fails a wrong one.

**Two layers: evaluate the evaluator.** A judge that silently goes lenient makes the analyst look *better*. Layer 2
uses the deterministic checks as free truth labels (does the judge fail answers we *know* are bad?). It also re-judges
16 canary analyses with known verdicts daily, tracks flip rate and bias, and measures κ against the owner's
judgments.

**Statistically honest alerts.** Rate SLOs need n ≥ 20 and fire only when the whole 99% Wilson interval is past the
SLO. Anomalies need z > 3 *and* a ≥ 50% effect. Alert quality is tested by replaying a seeded month with six
incidents under five random seeds.

**Model-written SQL, safely.** The DuckDB connection is read-only, takes a single SELECT, has external access
disabled, a timeout and a row cap. Charts are declarative specs rendered by the app; no model-written code runs.

**A public demo with a budget.** Each visitor gets $1 of live LLM calls a day, across chat and golden-set runs. Past
it, a simulator that runs real SQL on the real snapshot answers instead, and is labelled as such.

### Stack
Python · FastAPI (SSE) · OpenAI SDK (function calling, Structured Outputs) · DuckDB · SQLAlchemy (SQLite locally,
Neon Postgres in production) · Streamlit + Plotly · GitHub Actions · Docker on Hugging Face Spaces.
"""
)
