---
title: SuperEval
emoji: 🛒
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Two-layer eval monitoring for an LLM data-analyst agent
---

# SuperEval

*Production-grade evals for an AI agent answering questions about [Israeli supermarket prices](https://www.gov.il/he/pages/cpfta_prices_regulations).*

**How I monitor LLM evals in production, including monitoring the LLM judge itself.** It runs on a real agent
over real data.

PriceWatch is an LLM **data-analyst agent**. It answers analytical questions in Hebrew and English about the prices
Israeli supermarket chains are legally required to publish every day: 8 chains, 25 cities, ~6,000 products. Typical
questions:
- *"Which chain is really cheapest in Haifa?"*
- *"Which products have the biggest gap between chains?"*

It writes SQL against a read-only semantic layer and returns a structured answer: a headline, claims that each cite
the query they came from, the method, assumptions, caveats and a chart.

Every answer is evaluated in production in two layers:
- **Layer 1, the analyst:** deterministic checks on 100% of traffic, plus a sampled LLM judge. The checks re-run
  every cited query, lint the SQL method, and cross-check against canonical metrics.
- **Layer 2, the judge:** is the judge still trustworthy? It's checked against ground-truth labels, daily canaries,
  self-consistency, owner judgments and bias checks.

**▶ Live demo:** `https://huggingface.co/spaces/<you>/pricewatch-evals`.

---

## Purpose → failure modes → evals

**Purpose:** answer analytical questions about grocery prices correctly, honestly, and with the method shown.

| Failure mode | Eval | Coverage |
|---|---|---|
| Ungrounded or mis-copied number | **Grounding**: re-run each claim's cited query; the value must be in the result | 100% |
| Misleading method (Simpson's trap from comparing chains on different baskets, wrong promo denominator, tiny samples) | **SQL method lint** + **canonical cross-check** against the matched-basket macro | 100% |
| Broken SQL | SQL health: errors, empty results | 100% |
| Answering the unanswerable (price history, online prices) | Decline check | 100% |
| Ambiguity handled silently | Assumption must be stated | 100% |
| Misleading chart | Chart lint: columns exist, bars from zero, type fits the data | 100% |
| Wrong language · slow · expensive | Language match · p95 latency SLO · cost per task | 100% |
| Method validity, helpfulness, caveats | **LLM judge** (gpt-5 judging gpt-5-mini) with a rubric and stored reasoning | 30% |
| Regression before shipping | **Golden set**: 80 questions with reference SQL, human-approved | Every PR |

## KPIs (separate, no composite)

| KPI | Definition | Source |
|---|---|---|
| **Correctness** | Golden-set accuracy of the production prompt | Offline, CI |
| **Grounding** | Share of answers whose every number re-checks | Online, 100% |
| **Human override rate** | Share of human-reviewed answers judged wrong | Golden set page |
| **Cost per completed task** | Agent + judge USD ÷ completed answers | Traces |

Secondary metrics are sliced by question type and language:
- task completion, failure rate by type
- sound method, correct declines, stated assumptions
- latency p95, input mix

## What to try

1. **Ask the analyst.** Every query streams live. The answer comes with a chart, the SQL behind each number, and its
   eval scorecard.
2. **Test the agent.** *Agent evals* runs the live agent on the golden set (about 4 minutes, ~$0.20) and shows
   accuracy, accuracy by question type and every wrong answer, next to the Layer 1 metrics on live traffic. Every
   answer from the run is saved as a trace (source *golden*). The golden set grows from *Golden set* ("Add to the
   golden set" and owner judgments) and from every disagreement with the verdicts in *Chat*.
3. **Investigate.** *Golden set* filters to a failing check or question type, showing the SQL, the result, and why the
   lint flagged it.
4. **Who watches the judge?** *Judge validation* compares the judge with the owner-annotated gold set (confusion
   matrix, recall first), shows how ground-truth evidence changes its verdicts, and lists the failures humans caught
   that the suite missed. The owner's **Calibrate** updates the evidence the judge is fed and re-runs the comparison.
5. **Golden set.** Anyone can judge a trace on the *Golden set* page. Judgments are stored by role (visitor/owner), and
   owner judgments can add an answer to the golden set. With source *golden*, the owner reviews golden-set test
   answers: agreeing with the scorer approves the question, correct SQL replaces its reference, and "bad question"
   rejects it.

## Layer 2: evaluating the judge

A judge is a model too. Every verdict is stamped with the judge model and a rubric hash.

| Signal | How | SLO |
|---|---|---|
| Catches known-bad answers | Recall on answers grounding, lint or the canonical check *know* are wrong | ≥ 80% (72h, n ≥ 10) |
| Right on known cases | 16 canary analyses (Simpson's trap, mis-copied number, wrong city, wrong denominator, invented trend, …) re-judged daily | ≥ 80% |
| Same verdict twice | Flip rate on re-judged traces | ≤ 15% |
| Agrees with the owner | Cohen's κ from owner judgments on the Golden set page | reported, alert when low |
| Bias | Score vs. answer length; Hebrew vs. English gap | reported |
| Ops | Parse/API error rate | ≤ 5% |

## Monitoring and alert quality

- **Rate SLOs:**
  - grounding ≥ 98%
  - completion ≥ 95%
  - SQL health ≥ 95%
  - sound method ≥ 90%
  - correct declines ≥ 90%
  - p95 latency ≤ 20 s

  They need n ≥ 20 and fire only when the whole **99% Wilson interval** is past the SLO.
- **Other rules:**
  - **Anomalies** need z > 3 *and* a ≥ 50% effect.
  - **Input drift** is PSI on question type and language.
  - **Freshness** and **budget** rules round them out.
- **Alert quality is tested.** `scripts/alert_quality.py` replays a seeded month under five random seeds and scores
  recall (every incident caught) and precision (every alert belongs to an incident). The six incidents:
  - a model regression (Simpson's trap)
  - a schema change
  - a lenient judge (Layer 2 only)
  - a stale feed
  - a latency spike
  - a shift toward ambiguous questions

  **Result: 6/6 incidents caught on 5 of 5 seeds, 0 false positives.** The lenient judge is caught by Layer 2 only;
  the model regression is caught by the method checks while grounding stays green.

## Real-LLM benchmark

80 golden questions:
- **Question types:** lookup, comparison, ranking, spread, promo, geo, plus 20% traps (Simpson's,
  unanswerable, ambiguous).
- **Difficulty and language:** 24 easy / 40 medium / 16 hard; 64 English / 16 Hebrew.

Expected answers come from reference SQL on the day's snapshot. The scorers are scalar ±1%, ranking with ties, set
F1, decline and assumption.

| prompt | model | correctness | 95% CI | cost / question | median latency |
|---|---|---|---|---|---|
| a1 | gpt-5-mini | 93.8% | 86–97% | $0.0031 | 9.2 s |
| **a2** (production) | **gpt-5-mini** | **95.0%** | 88–98% | $0.0029 | 9.0 s |

The failures were real analyst errors, not scorer noise:
- **A misattributed price:** Rami Levy's ₪3.00 was reported as ₪2.50, which grounding catches online too.
- **A missed product search:** one product lookup didn't find the item.
- **Geographic questions:** both prompts are weakest here (n = 2), which is next to improve.

On matched products, Jerusalem comes out 4–7% cheaper than Tel Aviv and Haifa.

## The data

Under the Israeli Price Transparency law (חוק קידום התחרות בענף המזון), chains publish full price, promotion
and store files daily. The ingest uses [`il-supermarket-scraper`](https://pypi.org/project/il-supermarket-scraper/)
for Shufersal, Rami Levy, Victory, Yohananof, Osher Ad, Tiv Taam, Yellow and Hazi Hinam.

Handling the messy data:
- **Encodings:** UTF-16 and UTF-8 files.
- **Cities:** Hebrew spellings or CBS codes, resolved against the official CBS locality table.
- **Timestamps:** Israel local time in file names, converted to UTC.
- **Promotions:** ₪0.01 gift coupons are filtered out.
- **Servers:** flaky FTP, handled with retries.

A `gpt-5-mini` pass added English names, brands and categories for all products for about $0.50, cached by
barcode.

**Storage:** the full snapshot is published daily as a GitHub Release asset. A 3-city sample (Tel Aviv, Haifa,
Jerusalem) is committed for tests and CI, and the golden set is built from it so CI can compute every reference.

## Code map

| Path | What |
|---|---|
| `app/analyst/` | Semantic layer (DuckDB, canonical macros), OpenAI analyst, structured answer, simulator with fault knobs, question-type classifier |
| `app/evals/` | Online Layer 1 evals, SQL lint, gpt-5 judge, ground-truth evidence, judge validation, judge monitor (Layer 2), canaries, calibration, human judgments, golden set, scorers, benchmark |
| `app/monitor/` | Aggregates, rules (Wilson SLOs, z-score, PSI, freshness, budget, judge rules), alert lifecycle |
| `app/traffic/` | Synthetic questions; 30-day backfill with six seeded incidents |
| `app/api/` | FastAPI service (SSE chat stream, background golden-set runs, simulated bursts) and per-user cost guard |
| `app/data/` | XML parsing, city normalization, ingest, LLM enrichment, snapshot fetch/sample |
| `ui/` | Streamlit: chat, price explorer, review, agent evals, judge validation, architecture |

## Run it locally

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                            # OPENAI_API_KEY (optional), ADMIN_KEY for the owner role
python -m app.traffic.backfill --days 30 --reset
python -m app.evals.golden import               # load the golden set into the DB
uvicorn app.api.main:app --port 8000 &
python -m streamlit run ui/Home.py
```

Other commands:

```bash
pytest -q
python -m app.evals.golden selftest --all                  # reference SQL vs. scorers
python -m app.evals.bench --version a2 --live --record     # benchmark on the real analyst (cost-capped)
python -m app.evals.canaries --run                         # Layer 2 canaries
python scripts/alert_quality.py --seeds 1 2 3 4 5          # alert recall/precision across seeds
```

## Deploy (free tier)

1. **Database.** Create a [Neon](https://neon.tech) Postgres.
2. **Space.** Create a Hugging Face Space with the *Docker* SDK.
   - **Secrets:** `OPENAI_API_KEY`, `DATABASE_URL`, `SNAPSHOT_REPO`, `ADMIN_KEY` and `CONTACT_WHATSAPP`.
   - **Optional:** `USER_DAILY_BUDGET_USD`, `DAILY_BUDGET_USD`, `AGENT_MODEL`, `JUDGE_MODEL` and `PROMPT_VERSION`.
3. **GitHub.** Add the repo secrets `HF_TOKEN`, `DATABASE_URL` and `OPENAI_API_KEY`, and the variable `HF_SPACE`.
   - **Pushes to `main`** deploy the Space.
   - **PRs** run the golden-set gate.
   - **Scheduled:** a daily job ingests prices, synthetic traffic runs every 30 minutes, and canaries run daily.

Each user (by IP) can spend `USER_DAILY_BUDGET_USD` (default $1) of live LLM calls per UTC day, across chat and
golden-set runs. Past it, chat switches to the simulator and says so, and golden-set runs are refused. There is no
global ceiling. `DAILY_BUDGET_USD` only sets the threshold of the daily spend alert.

## Known limitations

- **The seeded history is simulated.** The simulated judge scores with a leniency knob, so the seeded month shows
  how the monitors behave, not the live judge's real accuracy. The live judge's accuracy is measured by canaries
  and κ.
- **The method lint is heuristic** (regex over SQL). It's backed by the canonical cross-check and the judge's
  method score.
- **Geo-blocking.** Chain servers may throttle non-Israeli IPs. If ingest fails, the freshness monitor flags it.

---

Built by **Sheer Wolff** · [LinkedIn](https://www.linkedin.com/in/sheerwolff) · [sheerwolff@gmail.com](mailto:sheerwolff@gmail.com)
