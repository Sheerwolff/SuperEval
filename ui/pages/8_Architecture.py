import streamlit as st

from lib import setup

setup("Architecture", "🧭")
st.markdown("## 🧭 How it's built, and why")
st.caption("Purpose → failure modes → success criteria → evals. Every check on this site exists because of a specific "
           "way an analysis, or the judge grading it, can go wrong.")

st.graphviz_chart(
    """
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fillcolor="#fbe7a1", color="#a2810b", penwidth=1.5,
        fontcolor="#0d3c68", fontname="Segoe UI", fontsize=11, margin="0.2,0.08"];
  edge [color="#ee9649", fontcolor="#b65e11", fontname="Segoe UI", fontsize=9];
  gov   [label="8 chains' XML feeds\\n(Price Transparency law)"];
  sem   [label="Semantic layer (DuckDB, read-only)\\nstores · products · prices · promos\\ncanonical metric macros"];
  user  [label="Chat visitors · synthetic traffic"];
  api   [label="FastAPI\\nchat stream · golden-set runs · calibration\\ncost guard: $1 per user per day"];
  agent [label="Analyst agent\\nrun_sql → submit_answer\\n(claims cite query ids)"];
  l1    [label="Layer 1 · rule checks\\ngrounding re-execution · SQL lint\\ncanonical cross-check · declines\\n100% of traffic"];
  judge [label="LLM judge (gpt-5)\\n30% sample\\n+ ground-truth evidence"];
  ev    [label="Ground-truth evidence\\nreviewed reference query\\nor canonical chain index"];
  fb    [label="Chat feedback\\nagree / disagree with the verdicts"];
  gold  [label="Golden set\\nquestions + reference SQL\\nread → trace → judge → approve"];
  evals [label="Agent Evals\\nlive golden-set test · success criteria"];
  jv    [label="Judge Validation\\nconfusion matrix vs human gold set\\nwith/without evidence · missed failures"];
  db    [label="Postgres\\ntraces · evals · judgments · alerts"];
  mon   [label="Monitors (background)\\nWilson SLOs · z-score · PSI · canaries\\nfreshness · spend"];
  gov -> sem;
  user -> api -> agent;
  agent -> sem [label="SQL"];
  agent -> l1 [label="trace"];
  sem -> l1 [label="re-run cited queries", style=dashed];
  agent -> judge;
  ev -> judge [label="facts", style=dashed];
  sem -> ev [style=dashed];
  gold -> ev [label="approved items\\n(calibrate)", style=dashed];
  user -> fb;
  fb -> gold [label="disagreement\\n→ draft"];
  evals -> agent [label="test run"];
  gold -> evals [label="questions"];
  evals -> gold [label="answers to review", style=dashed];
  gold -> jv [label="owner labels"];
  judge -> jv;
  fb -> jv;
  l1 -> db; judge -> db; db -> mon -> db;
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

The answer is a plain-language headline for consumers, with the method and SQL one click away for analysts. SuperEval
is everything around that agent that proves the answers are right, and keeps proving it in production.

### Failure modes → evals
| Failure mode | How it's caught |
|---|---|
| **Ungrounded number** (hallucinated or mis-copied) | **Grounding**: every claim cites a query id; the query is re-run and the value must be in its result |
| **Misleading method** (comparing chains on different baskets: Simpson's trap; wrong promo denominator) | **SQL method lint** + **canonical cross-check** against the matched-basket macro + judge method score |
| **Wrong SQL** | SQL health (errors, empty results), canonical cross-check |
| **Wrong answer to a known question** | **Golden-set test** on Agent Evals: the live agent against reviewed reference SQL on today's data |
| **Answering the unanswerable** (price history, online prices) | Decline check against the question classifier |
| **Silent ambiguity** | An ambiguous question must come with a stated assumption |
| **Misleading chart** | Chart lint: columns exist, bars start at zero, the type fits the data |
| **Wrong language, slow, expensive** | Language match, latency, cost per completed task |
| **A judge that's wrong** (too lenient or too strict) | **Judge Validation**: confusion matrix against the human-annotated gold set, recall first; canaries in the background |
| **Failures no check catches** | **Human feedback**: chat 👍 / 👎 on the verdicts and owner judgments on the Golden set page, listed on Judge Validation as failures the suite missed |

### Success criteria (separate, no composite score)
Shown on Agent Evals, each with its definition on hover:
- **Accuracy:** share of golden-set questions answered correctly in the latest live test.
- **Helpfulness:** the judge's 1–5 rating of whether an answer addresses the question.
- **Grounding · SQL health · Sound method · Correct declines:** the Layer 1 rule checks on live traffic.
- **Latency p95** and **cost per completed task** (agent + judge spend ÷ completed answers).
- **Human override rate:** share of human-reviewed answers judged wrong.

A composite would hide trade-offs, like a cheaper model that's less grounded.

### Design decisions
**Golden questions, not golden answers.** Prices change daily, so the golden set fixes *questions* with *reference
SQL*, and the expected answer is computed from today's snapshot. It starts from generated questions and grows from
real traffic: anyone can add an answer from the Golden set page, and every disagreement in Chat becomes a draft. The
owner approves items by working through a test answer: read the question and answer, read the trace, judge it, approve.
Only approved items gate CI, and a selftest proves every scorer passes a perfect answer and fails a wrong one.

**Tested live, reviewed like traffic.** Agent Evals runs the live agent on the golden set in the background. Every
answer is saved as a *golden* trace, so it can be reviewed on the Golden set page, but it's kept out of the
live-traffic metrics and the human override rate. Its cost still counts toward spend.

**Two layers: evaluate the evaluator.** A judge that silently goes lenient makes the analyst look *better*. Judge
Validation compares the judge with the owner's own verdicts on the gold set (recall first, since missing a wrong answer
is the costly error). In the background, the deterministic checks act as free truth labels, 16 canaries with known
verdicts are re-judged daily, and flip rate and bias are tracked.

**Give the judge ground truth.** A judge that only sees the agent's own queries can check consistency, not
correctness. So it also gets records retrieved independently of the agent: the reviewed reference query of a matching
golden question, or the canonical chain price index for chain comparisons. When nothing reliable exists, it gets
nothing rather than a guess. **Calibrate** refreshes that evidence set from the approved golden questions and re-judges
the gold set with and without evidence, so the ablation shows whether it helps.

**Statistically honest alerts.** Rate SLOs need n ≥ 20 and fire only when the whole 99% Wilson interval is past the
SLO. Anomalies need z > 3 *and* a ≥ 50% effect. Alert quality is tested by replaying a seeded month with six
incidents under five random seeds.

**Model-written SQL, safely.** The DuckDB connection is read-only, takes a single SELECT, has external access
disabled, a timeout and a row cap. Charts are declarative specs rendered by the app; no model-written code runs.

**A public demo with a budget.** Each visitor gets $1 of live LLM calls a day, across chat, golden-set runs and judge
calibration. Past it, chat is answered by a simulator that runs real SQL on the real snapshot, labelled as such, and
paid runs are refused. Calibrating is owner-only.

### Stack
Python · FastAPI (SSE, background jobs) · OpenAI SDK (function calling, Structured Outputs) · DuckDB · SQLAlchemy
(SQLite locally, Neon Postgres in production) · Streamlit + Plotly + Graphviz · GitHub Actions · Docker on Google Cloud Run (built by Cloud Build on every
push).
"""
)
