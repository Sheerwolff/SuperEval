---
title: PriceWatch Evals
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Two-layer eval monitoring for an LLM shopping agent
---

# PriceWatch Evals

**How I monitor LLM evals in production, including monitoring the LLM judge itself.** It runs on a real agent
with real data.

PriceWatch is an LLM agent that answers grocery-price questions in Hebrew and English (*"what's the cheapest
cottage cheese in Haifa?"*). It uses the prices Israeli supermarket chains are legally required to publish every
day: 8 chains, 25 cities, ~6,000 products. Every answer is evaluated in production in two layers:

- **Layer 1, the agent:** exact checks against the price database on 100% of traffic, plus an LLM judge.
- **Layer 2, the judge:** is the judge still trustworthy? It's measured against ground truth, daily canaries,
  self-consistency, human labels and bias checks.

Both layers feed statistically gated SLOs, anomaly and drift detection, data-freshness checks and alerting. A CI
gate blocks bad prompts before they ship.

**▶ Live demo:** `https://huggingface.co/spaces/<you>/pricewatch-evals`. Start with the 3-minute guided tour.

---

## What to try (the guided tour)

1. **Chat.** Ask in Hebrew or English. Tool calls stream live ("Searching 'milk' → 8 matches → prices in Haifa"),
   and the reply comes with a per-chain price card and its eval scorecard: every ₪ amount checked against the
   database, grounding, product relevance, tool use, language, scope, and the judge's verdict.
2. **Break production.** On *Live monitor*, deploy **v2**, a plausible "faster & cheaper" prompt that drops the
   grounding rule. Send 40 requests and critical alerts open (price accuracy collapses, prices are answered from
   memory, the language is wrong). Roll back and they resolve.
3. **Investigate.** *Trace explorer* opens pre-filtered to failing answers: the tool calls, the number that isn't in
   any tool result, the judge's reasoning.
4. **Who watches the judge?** *Judge monitor* shows the seeded day-−9 incident: a judge model update made the judge
   lenient. Layer 1's judge scores got *better*; Layer 2's canaries and ground-truth recall caught it.
5. **Explore the data.** *Price explorer* compares chains per product and shows which chain is cheapest in each
   category, city by city.

## Layer 1: evaluating the agent

| What's checked | How | Coverage |
|---|---|---|
| Every quoted price exists in the tool results the agent received | Deterministic: extract ₪ amounts, match to ±0.01 | 100% |
| No price stated without a price lookup; no chains outside the data | Deterministic | 100% |
| The priced product is what was asked for ("milk" ≠ milk chocolate) | Deterministic (head noun + catalog category) and judge | 100% |
| Right tool; reply in the user's language; off-topic and prompt injection declined; promos mentioned; data date cited | Deterministic | 100% |
| Helpfulness and groundedness (1–5) | LLM judge, OpenAI Structured Outputs, rubric, reasoning stored | Sampled |
| Regression before shipping | Fixed golden questions; expectations derived from the day's data | Every PR |

**Why mostly deterministic?** Prices are facts in a database. The most important question (*is this number
right?*) can be answered exactly, cheaply, on every request, with no judge bias. The judge covers what rules can't.

**Why golden *questions*, not golden answers?** Prices change daily, so stored expected answers rot within a day.

## Layer 2: evaluating the judge

A judge is a model too. If it silently goes lenient (a provider model update, a rubric edit), the agent looks
*better* while it isn't. Every verdict is stamped with the judge model and a rubric hash, and the judge gets its
own metrics and SLOs:

| Signal | How | SLO |
|---|---|---|
| Catches known-bad answers | Recall on answers the deterministic checks *know* are wrong: free, always-on labels | ≥ 80% (72h, Wilson 99%) |
| Still right on known cases | 24 canary replies with known verdicts (wrong price, invented chain, wrong product, complied injection, …), re-judged daily | ≥ 85% |
| Same verdict twice | Flip rate when a trace is re-judged | ≤ 15% |
| Agrees with humans | Cohen's κ from labels on the *Judge monitor* page | ≥ 0.6 (n ≥ 30) |
| Bias | Helpfulness vs. answer length (Spearman ρ); helpfulness gap Hebrew vs. English | reported |
| Score drift | PSI of judge scores with the agent version held fixed | info alert |
| Ops | Parse/API error rate, latency, cost | ≤ 5% errors |

## Monitoring and alert quality

- **SLOs** apply per *deployed* agent version over a 3h window: price accuracy ≥ 95%, grounded ≥ 95%, language ≥ 90%,
  scope ≥ 90%, right product ≥ 80%, p95 latency ≤ 8s, errors ≤ 5%. Rate SLOs need n ≥ 20 and fire only when the whole
  **99% Wilson interval** is past the SLO. Without that, a 95% SLO on 15 samples "breaches" on one miss: naive
  thresholds produced 46 alerts over the seeded month, most of them noise.
- **Anomaly rules** need both z > 3 *and* a ≥ 50% effect size. **Input drift** uses PSI on the topic and language
  mix. **Data freshness** checks each chain's feed age against a 36h SLA. **Budget** tracks daily LLM spend against
  a cap.
- **Alert quality is tested, not assumed.** `scripts/alert_quality.py` replays the seeded month under several random
  seeds and scores recall (every incident caught) and precision (every alert belongs to an incident). Result:
  **5/5 incidents caught on 5 of 5 seeds, 0 false positives.** The lenient-judge incident is caught by Layer 2 only
  (canaries on every seed; the ground-truth recall rule, which needs ≥10 known-bad answers in its 72h window, in
  2 of 5 seeds). Layer 1 never fires for it, which is the point.

The seeded month contains five incidents:
- a bad prompt deploy
- a stale price feed (Victory)
- an upstream latency spike
- a pre-holiday input-mix shift
- a lenient-judge regression, visible only to Layer 2

## Real-LLM quality loop

The golden set (60 fixed questions) runs against the live agent with `--live --record`, and each run lands in
*Version compare*. The first live run of v1 on `gpt-5-mini` found real problems in the agent:
- it asked clarifying questions instead of answering generic requests ("price of coffee in Tel Aviv")
- it hit the step limit on 6-item baskets by searching one item per turn
- baskets exceeded the latency SLO

It also exposed four kinds of evaluator bugs:
- refusals worded "I can’t…" with a curly apostrophe weren't recognized
- "מה מזג האוויר" (weather) was classified as a price question
- Hebrew street addresses counted against English answers in the language check
- a too-strict relevance rule failed "Eggs 18pcs medium" for "eggs"

Each bug got a fix and a regression test. Prompt **v3** fixes the agent issues. Results on the same 60 questions,
with the fixed evaluators:

| prompt | model | overall pass | price accuracy | correct tool use | language | cost / question | median latency |
|---|---|---|---|---|---|---|---|
| v1 | gpt-5-mini | 71.7% | 86.2% | 76.7% | 91.7% | $0.0021 | 5.1 s |
| **v3** | **gpt-5-mini** | **95.0%** ✅ | **97.6%** | **100%** | **100%** | $0.0024 | 6.8 s |
| v3 | gpt-5-nano | 65.0% | 97.4% | 97.7% | 75.0% | $0.0007 | 6.4 s |

- **v3 ships.** It clears every gate.
- **The honest trade-off:** answering decisively costs about 1.7 s more median latency, and 70% of answers meet
  the 8 s latency SLO (down from 80%). That's the next thing to work on.
- **gpt-5-nano is 3.4× cheaper**, but answers in the wrong language a quarter of the time, so it doesn't ship.

Total LLM spend for this whole loop (enrichment, four live golden-set runs, smoke tests): under $1.

## The data

Under the Israeli Price Transparency law (חוק קידום התחרות בענף המזון), chains publish full price, promotion
and store files daily. The ingest uses [`il-supermarket-scraper`](https://pypi.org/project/il-supermarket-scraper/)
for **Shufersal, Rami Levy, Victory, Yohananof, Osher Ad, Tiv Taam, Yellow and Hazi Hinam**. It keeps the cities
where at least 5 chains have stores (25 cities, 2 stores per chain per city) and products sold by 2+ chains.

The data is messy, and handling it is part of the work:
- **Encodings and tags vary.** Files come in UTF-16 with BOM or UTF-8, and tag casing varies by chain.
- **City is inconsistent.** It's a Hebrew name (many spellings), a CBS code, or `0` (Yohananof). Every value is
  resolved against the official CBS locality table from data.gov.il; city-less stores went from ~280 to ~20.
- **Timestamps are local.** File names carry Israel local time, converted to UTC.
- **Promotions need filtering.** They include coupons, club deals and ₪0 gifts, which are filtered out.
- **Servers are flaky.** FTP servers time out and files arrive truncated; the ingest retries.
- **English names and categories come from an LLM.** One `gpt-5-mini` pass (Structured Outputs, fixed category
  list) gave all 6,000 products an English name, brand and category for **$0.50**. It's cached by barcode, so the
  daily job only pays for new products.

**Storage.** The full snapshot (~9 MB) is published daily as the `snapshot-latest` GitHub Release asset, overwritten
so no history accumulates, and the app downloads it at startup. A 460 KB sample (2 cities) is committed for tests,
CI and offline use.

## Architecture

```
Chains' XML ─► daily ingest + LLM enrichment ─► snapshot (release asset) ─┐ ground truth
                                                                          ▼
Visitor ─► FastAPI /chat/stream ─► agent (OpenAI tools) ─► trace ─► Layer 1 evals ─► Layer 2 (judge) evals
             └ cost guard                                                 └──────► Postgres ─► monitors ─► alerts
                                                                                     └──── Streamlit dashboard
PR ─► CI: lint · tests · golden-set gate on the production prompt
```

| Path | What |
|---|---|
| `app/data/` | XML parsing, city normalization (`cities.py`), ingest, LLM enrichment, catalog (tools + ground truth), snapshot fetch/sample |
| `app/agent/` | OpenAI function-calling agent with streaming events, versioned prompts (`v1`, buggy `v2`, production `v3`), simulator |
| `app/evals/` | Deterministic evaluators, LLM judge, judge monitor (Layer 2), canaries, offline runner + golden set, calibration |
| `app/monitor/` | Aggregates, rules (Wilson SLOs, z-score, PSI, freshness, budget, judge rules), alert lifecycle |
| `app/traffic/` | Synthetic shoppers; 30-day backfill with five seeded incidents |
| `app/api/` | FastAPI service (SSE chat stream) and cost guard |
| `ui/` | Streamlit dashboard: guided tour, chat, price explorer, live monitor, traces, versions, judge monitor, data health |

**The simulator.** `app/agent/sim.py` is a deterministic stand-in for the LLM. It calls the *real* tools on the real
snapshot and injects each prompt version's known failure modes. It powers the backfill, CI and tests, and serves
the demo whenever there's no API key or the daily budget is spent. Its traces are tagged `model=sim`, and
backfilled ones `source=backfill`.

## Run it locally

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                            # OPENAI_API_KEY for the live agent (optional)
python -m app.traffic.backfill --days 30 --reset
uvicorn app.api.main:app --port 8000 &
python -m streamlit run ui/Home.py
```

Without a downloaded snapshot the app uses the committed sample. For full data, either run
`pip install -r requirements-ingest.txt && python -m app.data.ingest`, or set `SNAPSHOT_REPO=<owner>/<repo>` and run
`python -m app.data.fetch_snapshot`.

Other commands:

```bash
pytest -q                                                        # unit tests
python -m app.evals.offline_runner --version v3 --gate           # golden set, simulated agent
python -m app.evals.offline_runner --version v3 --live --record  # golden set, real agent (cost-capped)
python -m app.evals.canaries --run                               # Layer 2 canaries
python scripts/alert_quality.py --seeds 1 2 3                    # alert recall/precision across seeds
```

## Deploy (free tier)

1. **Database.** Create a free [Neon](https://neon.tech) Postgres.
2. **Space.** Create a Hugging Face Space with the *Docker* SDK. Add the secrets `OPENAI_API_KEY`, `DATABASE_URL`,
   `SNAPSHOT_REPO` and `CONTACT_WHATSAPP`, and optionally `DAILY_BUDGET_USD`, `AGENT_MODEL` and `JUDGE_MODEL`.
3. **GitHub.** Add the repo secrets `HF_TOKEN`, `DATABASE_URL` and `OPENAI_API_KEY`, and the variable
   `HF_SPACE=<you>/pricewatch-evals`.
   - Pushes to `main` deploy the Space.
   - The daily job ingests, enriches and publishes the snapshot.
   - Synthetic traffic runs every 30 minutes, and judge canaries run daily.

Cost: hosting and the database are free. LLM spend is capped by `DAILY_BUDGET_USD` (default $2/day); past the cap,
the demo switches to the simulator and says so.

## Known limitations

- **The simulated judge** derives its scores from the deterministic results plus noise, so in the seeded history
  it can't catch what those miss. The live judge can, and the *Judge monitor* page measures how well.
- **The product-relevance check is heuristic** (head noun + catalog category). It's conservative by design and
  backed up by the judge's relevance verdict.
- **Geo-blocking.** Chain servers can block or throttle non-Israeli IPs. If the daily ingest fails from a CI runner,
  the freshness monitor flags it, as designed.
- **The seeded history** is relative to the first boot; re-seed with `python -m app.traffic.backfill --reset`.

---

Built by **Sheer Wolff** · [LinkedIn](https://www.linkedin.com/in/sheerwolff) · [sheerwolff@gmail.com](mailto:sheerwolff@gmail.com)
