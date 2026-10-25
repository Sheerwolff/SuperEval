---
title: PriceWatch Evals
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Production eval monitoring for an LLM shopping agent
---

# PriceWatch Evals

**How I monitor LLM evals in production**, shown on a real agent with real data.

PriceWatch is an LLM agent that answers grocery-price questions in Hebrew and English (*"what's the cheapest
cottage cheese in Haifa?"*). It uses the prices that Israeli supermarket chains are legally required to publish every
day. Every answer it gives is evaluated in production. The results feed a monitoring stack with SLOs, anomaly and
drift detection, data-freshness checks and alerting, plus a CI gate that blocks bad prompts before they ship.

**▶ Live demo:** `https://huggingface.co/spaces/<you>/pricewatch-evals` &nbsp;·&nbsp; try *Live monitor → Deploy v2*

---

## What to try in the demo

1. **Chat.** Ask about a product, a basket or a city. Each reply comes with its eval scorecard: every ₪ amount
   checked against the database, grounding, tool use, language, scope, and an LLM judge's verdict with reasoning.
2. **Break production.** On *Live monitor*, deploy **v2**, a plausible "faster & cheaper" prompt that drops the
   grounding rule. Send 40 requests and critical alerts open within seconds (price accuracy collapses, prices get
   answered from memory, the language is wrong). Roll back and they resolve.
3. **Investigate.** *Trace explorer* filters by failing evaluator and shows the tool calls, the numbers that didn't
   match and the judge's reasoning. *Version compare* puts v1 and v2 side by side with confidence intervals.
4. **Read the history.** Thirty days of seeded traffic include four incidents: a bad deploy, a stale price feed,
   a latency spike and an input-drift event. Each one is caught by the monitors, with zero false-positive alerts.

## The eval design

| Layer | What's checked | How | Coverage |
|---|---|---|---|
| Correctness | Every quoted price exists in the tool results the agent received | Deterministic: regex-extract ₪ amounts, match to ±0.01 | 100% |
| Grounding | No price stated without a price lookup; no chains outside the data | Deterministic | 100% |
| Behaviour | Right tool used; reply in the user's language; off-topic and prompt injection declined; promos mentioned; data date cited | Deterministic | 100% |
| Quality | Helpfulness and groundedness (1–5), in-scope behaviour | LLM judge, OpenAI Structured Outputs, rubric, reasoning stored | Sampled |
| Judge quality | Judge vs. human agreement | Cohen's κ from labels added on the *Calibrate* page | Ongoing |
| Regression | Fixed golden questions; expectations derived from the day's data | Offline runner in CI, gated | Every PR |

**Why mostly deterministic?** Prices are facts in a database. The most important question (*is this number
right?*) can be answered exactly, cheaply, on every request, with no judge bias. The LLM judge covers only what
rules can't, and it gets evaluated too.

**Why golden *questions* and not golden answers?** Prices change daily, so stored expected answers rot within a day.
The suite fixes the questions and checks answers against the snapshot at run time.

## The monitoring design

- **SLO rules** per deployed version over a 3h window: price accuracy ≥ 95%, grounded ≥ 95%, language ≥ 90%,
  scope ≥ 90%, p95 latency ≤ 8s, errors ≤ 5%. Rate SLOs need n ≥ 20 and fire only when the whole **99% Wilson
  interval** is past the SLO. Without that, a 95% SLO on 15 samples "breaches" on one miss: naive thresholds
  produced 46 alerts over the seeded month, most of them noise.
- **Anomaly rules**: current window vs. a 7-day baseline, requiring both z > 3 *and* a ≥ 50% effect size.
- **Alert quality is tested, not assumed.** `scripts/alert_quality.py` replays the month under several random seeds
  and scores recall (incidents caught) and precision (alerts that belong to an incident). Current rules: 4/4
  incidents caught, 0 false positives, on 5 of 5 seeds.
- **Input drift**: PSI of topic and language mix, last 24h vs. the previous 7 days.
- **Data freshness**: each chain's feed age vs. a 36h SLA. Stale data looks like a hallucination to the user.
- **Budget**: daily LLM spend (agent + judge, from token usage) vs. a cap.
- Alerts open and resolve automatically, are shown in the dashboard, and can post to a Slack/Discord webhook.

Every trace records the **agent version, model, prompt hash and data snapshot date**, so any regression can be pinned
to a deploy, a model change or a data change.

## The data

Under the Israeli Price Transparency law (חוק קידום התחרות בענף המזון), chains publish full price, promotion
and store files daily. The ingest uses [`il-supermarket-scraper`](https://pypi.org/project/il-supermarket-scraper/)
to download them for **Shufersal, Rami Levy, Victory and Yohananof** across 8 cities, then normalizes them.

The data is messy, and handling it is part of the work:
- Files come in UTF-16 with BOM or UTF-8, and tag casing varies by chain.
- City is a Hebrew name, a numeric CBS code (Shufersal), or `0` for every store (Yohananof, where city is inferred
  from the store name).
- File names carry Israel local time, which the ingest converts to UTC.
- Promotions include coupons, club deals and ₪0 gifts, which are filtered out.
- Servers time out and files arrive truncated, which the ingest retries.

The snapshot (~4,000 products sold by 2+ chains, ~230k prices, ~80k active promos, about 2 MB of parquet) is
committed, so the app runs anywhere. A daily job refreshes it.

## Architecture

```
Chains' XML ─► daily ingest (GH Actions) ─► parquet snapshot ─┐ (ground truth)
                                                              ▼
Visitor / synthetic ─► FastAPI /chat ─► agent (OpenAI tools) ─► trace ─► evals ─► Postgres ─► monitors ─► alerts
                        └ cost guard                                       │                   │
                                                                           └──── Streamlit dashboard ◄┘
PR ─► CI: lint · tests · offline golden-set gate
```

| Path | What |
|---|---|
| `app/data/` | XML parsing (`xml_parse.py`), ingest, catalog (tools + ground truth) |
| `app/agent/` | OpenAI function-calling agent, versioned prompts (`prompts/v1.yaml`, `v2.yaml`), simulator |
| `app/evals/` | Deterministic evaluators, LLM judge, offline runner + golden set, judge calibration |
| `app/monitor/` | Aggregates, rules (SLO / z-score / PSI / freshness / budget), alert lifecycle |
| `app/traffic/` | Synthetic shoppers; 30-day backfill with seeded incidents |
| `app/api/` | FastAPI service and cost guard |
| `ui/` | Streamlit dashboard |

**The simulator.** `app/agent/sim.py` is a deterministic stand-in for the LLM. It calls the *real* tools on the
*real* snapshot and injects each prompt version's known failure modes. It powers the backfill, CI, tests, and the
demo whenever there's no API key or the daily budget is spent. Its traces are tagged `model=sim`, and backfilled
ones `source=backfill`, so they're never confused with live traffic.

## Run it locally

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                            # add OPENAI_API_KEY for the live agent (optional)
python -m app.traffic.backfill --days 30 --reset
uvicorn app.api.main:app --port 8000 &
python -m streamlit run ui/Home.py
```

Or with Docker: `docker compose up --build`, then open http://localhost:7860.

Other commands:

```bash
pytest -q                                              # unit tests
python -m app.evals.offline_runner --version v1 --gate # offline suite (add --live for the real agent)
python -m app.evals.offline_runner --version v2        # watch the bad prompt fail the gate
pip install -r requirements-ingest.txt && python -m app.data.ingest   # fresh prices
```

## Deploy (free tier)

1. **Database.** Create a free [Neon](https://neon.tech) Postgres and copy its connection string.
2. **Space.** Create a Hugging Face Space with the *Docker* SDK. In its settings, add the secrets `OPENAI_API_KEY`
   and `DATABASE_URL`, and optionally `DAILY_BUDGET_USD`, `AGENT_MODEL` and `JUDGE_MODEL`. On first boot it seeds
   30 days of history.
3. **GitHub.** Add the repo secrets `HF_TOKEN` (write), `DATABASE_URL` and `OPENAI_API_KEY`, and the variable
   `HF_SPACE=<you>/pricewatch-evals`. Pushes to `main` deploy, the daily job refreshes prices, and synthetic
   traffic runs every 30 minutes.

Cost: hosting and the database are free. LLM spend is capped by `DAILY_BUDGET_USD` (default $2/day). Past the cap,
the demo switches to the simulator and says so.

## Known limitations

- The simulated judge derives its scores from the deterministic results, so it can't catch problems those miss,
  such as picking a less relevant product. The live OpenAI judge does. That gap is exactly what the *Calibrate*
  page measures.
- Chain servers can block or throttle non-Israeli IPs. If the daily ingest fails from a CI runner, the freshness
  monitor flags it, as designed.
- The seeded history is relative to the first boot. Re-seed with `python -m app.traffic.backfill --reset` to
  re-center it.
