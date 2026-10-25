"""Offline regression suite: a fixed set of questions, run against a prompt version,
gated on pass rates. Runs in CI on every PR.

    python -m app.evals.offline_runner --version v1 --gate
    python -m app.evals.offline_runner --version v2           # see what a bad prompt scores

Golden *questions* are fixed (golden_set.jsonl). Golden *answers* are not stored: prices
change daily, so expectations are derived from the snapshot at run time — the
deterministic evaluators compare each answer to the tool results / database.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from app.agent.agent import run_agent
from app.config import settings
from app.data.catalog import get_catalog
from app.evals.deterministic import run_deterministic
from app.pipeline import to_trace

GOLDEN = Path(__file__).parent / "golden_set.jsonl"
RUNS = Path(__file__).resolve().parents[2] / "data" / "reference" / "golden_runs.jsonl"
CRITICAL = ["price_accuracy", "grounded_prices", "tool_use", "scope_handling", "language_match", "product_relevance"]
GATES = {"overall_pass_rate": 0.90, "price_accuracy": 0.95, "grounded_prices": 0.95, "scope_handling": 0.90}


def build_golden(n: int = 60, seed: int = 2026) -> list[dict]:
    from app.traffic.synthetic import Shopper

    shopper = Shopper(get_catalog(), random.Random(seed))
    mix = {"price_lookup": 0.55, "basket": 0.15, "stores": 0.05, "off_topic": 0.12, "injection": 0.13}
    return [{"id": f"g{i:03d}", "question": q.text, "intent": q.intent}
            for i, q in enumerate(shopper.ask(mix) for _ in range(n))]


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_suite(version: str, force_sim: bool, limit: int | None = None, max_cost: float = 1.0,
              save: bool = False, model: str | None = None) -> dict:
    """Runs the golden set. `max_cost` (USD) stops a live run early; `save` stores the traces
    (source="offline") so failures can be triaged in the Trace Explorer."""
    cases = load_golden()[:limit]
    per_eval = defaultdict(list)
    rows, cost, latencies = [], 0.0, []
    for i, case in enumerate(cases):
        if cost >= max_cost:
            print(f"stopping: cost ${cost:.3f} reached --max-cost ${max_cost:.2f}", file=sys.stderr)
            break
        res = run_agent(case["question"], version, force_sim=force_sim, intent=case["intent"], seed=i, model=model)
        cost += res.cost_usd if res.model != "sim" else 0.0
        latencies.append(res.latency_ms)
        trace = to_trace(res, "offline", f"golden-{version}")
        if save:
            from app.db import SessionLocal, init_db
            from app.pipeline import evaluate

            init_db()
            with SessionLocal() as s:
                s.add(trace)
                s.flush()
                for e in evaluate(trace, judge=False):
                    e.trace_id = trace.id
                    s.add(e)
                s.commit()
        outcomes = {o.evaluator: o for o in run_deterministic(trace)}
        for name, o in outcomes.items():
            if o.passed is not None:
                per_eval[name].append(o.passed)
        ok = all(outcomes[c].passed is not False for c in CRITICAL)
        rows.append({"id": case["id"], "question": case["question"], "passed": ok, "trace_id": trace.id,
                     "failed": [c for c in CRITICAL if outcomes[c].passed is False], "answer": res.answer,
                     "error": res.error})
    summary = {k: sum(v) / len(v) for k, v in per_eval.items() if v}
    summary["overall_pass_rate"] = sum(r["passed"] for r in rows) / len(rows)
    latencies.sort()
    return {"version": version, "mode": "sim" if force_sim else "live", "n": len(rows),
            "model": "sim" if force_sim else (model or settings.agent_model),
            "cost_usd": round(cost, 4), "latency_p50_ms": latencies[len(latencies) // 2] if latencies else None,
            "summary": summary, "rows": rows}


def report_markdown(result: dict, gate_failures: list[str]) -> str:
    lines = [f"### Offline eval: `{result['version']}` ({result['mode']}, n={result['n']}, "
             f"${result.get('cost_usd', 0):.3f})", "",
             "| metric | value | gate |", "|---|---|---|"]
    for k, v in sorted(result["summary"].items()):
        gate = GATES.get(k)
        mark = "" if gate is None else (f"≥ {gate:.0%} ❌" if k in gate_failures else f"≥ {gate:.0%} ✅")
        lines.append(f"| {k} | {v:.1%} | {mark} |")
    failed = [r for r in result["rows"] if not r["passed"]][:10]
    if failed:
        lines += ["", "<details><summary>Failing cases</summary>", ""]
        lines += [f"- **{r['id']}** {r['question']} → failed {', '.join(r['failed'])}" for r in failed]
        lines += ["", "</details>"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--live", action="store_true", help="use the OpenAI agent (needs OPENAI_API_KEY)")
    ap.add_argument("--gate", action="store_true", help="exit 1 if any gate fails")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-cost", type=float, default=1.0, help="USD cap for a live run")
    ap.add_argument("--save", action="store_true", help="store traces for triage in the Trace Explorer")
    ap.add_argument("--build", action="store_true", help="(re)generate golden_set.jsonl")
    ap.add_argument("--model", help="agent model override for a live run (model comparison)")
    ap.add_argument("--record", action="store_true", help="append the summary to data/reference/golden_runs.jsonl")
    args = ap.parse_args()
    if args.build:
        GOLDEN.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in build_golden()) + "\n",
                          encoding="utf-8")
        print(f"wrote {GOLDEN}")
        return
    result = run_suite(args.version, force_sim=not args.live, limit=args.limit, max_cost=args.max_cost,
                       save=args.save, model=args.model)
    if args.record:
        from datetime import UTC, datetime

        RUNS.parent.mkdir(parents=True, exist_ok=True)
        with RUNS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(UTC).isoformat(timespec="seconds"),
                                **{k: v for k, v in result.items() if k != "rows"}}) + "\n")
    failures = [k for k, g in GATES.items() if result["summary"].get(k, 1.0) < g]
    md = report_markdown(result, failures)
    print(md)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md + "\n")
    if args.gate and failures:
        print(f"\nGATE FAILED: {failures}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
