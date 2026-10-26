"""Run the analyst benchmark: golden questions -> analyst -> correctness scorers.

    python -m app.evals.bench --version a1 --live --record          # real model (cost-capped)
    python -m app.evals.bench --version a1 --live --include-drafts   # before the set is reviewed
    python -m app.evals.bench --selftest                             # CI without a key: references + scorers

Correctness is reported overall and sliced by question type, difficulty and language, each with a
95% Wilson interval, together with completion rate, cost and latency. `--gate` exits non-zero
when overall correctness is below GATE (or declines/assumptions below their own floors).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from app.analyst.semantic import get_semantic
from app.config import ROOT, settings
from app.evals import golden, scoring
from app.monitor.rules import wilson

RUNS = ROOT / "data" / "reference" / "bench_runs.jsonl"
GATES = {"overall": 0.80, "trap_unanswerable": 0.80, "trap_ambiguous": 0.75}


def run_one(item: dict, version: str, live: bool, model: str | None, run_id: str | None = None) -> dict:
    """Answer and score one golden item. With `run_id`, the answer is also stored as a trace (source 'golden')
    with a `golden_match` verdict, so it can be reviewed on the Review page."""
    sem = get_semantic()
    if live:
        from app.analyst.agent import run_analyst

        res = run_analyst(item["question"], version, model=model)
    else:
        from app.analyst.sim import run_sim

        res = run_sim(item["question"], version)
    s = scoring.score(item, res.answer, res.queries, sem)
    if run_id:
        save_trace(res, item, s, run_id)
    return {"id": item["id"], "qtype": item["qtype"], "difficulty": item["difficulty"], "language": item["language"],
            "answer_type": item["answer_type"], "passed": s.passed, "score": s.score, "details": s.details,
            "completed": res.completed, "error": res.error, "cost_usd": res.cost_usd if live else 0.0,
            "latency_ms": res.latency_ms, "steps": res.steps, "question": item["question"],
            "headline": (res.answer or {}).get("headline"), "trace_id": res.id}


def save_trace(res, item: dict, s, run_id: str) -> None:
    from app.analyst.pipeline import to_trace
    from app.data.catalog import get_catalog
    from app.db import EvalResult, SessionLocal

    trace = to_trace(res, "golden", f"golden:{run_id}", snapshot_date=get_catalog().snapshot_date,
                     topic=item["qtype"])
    with SessionLocal() as db:
        db.add(trace)
        db.flush()
        db.add(EvalResult(trace_id=trace.id, evaluator="golden_match", kind="deterministic", score=s.score,
                          passed=s.passed, details={**(s.details or {}), "golden_id": item["id"]}))
        db.commit()


def summarize(rows: list[dict]) -> dict:
    def rate(rs):
        n = len(rs)
        p = sum(r["passed"] for r in rs) / n if n else 0.0
        lo, hi = wilson(p, n) if n else (0.0, 0.0)
        return {"n": n, "rate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)]}

    out = {"overall": rate(rows)}
    for dim in ("qtype", "difficulty", "language"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[dim]].append(r)
        out[dim] = {k: rate(v) for k, v in sorted(groups.items())}
    lat = sorted(r["latency_ms"] for r in rows)
    completed = [r for r in rows if r["completed"]]
    out["completion_rate"] = round(len(completed) / len(rows), 3) if rows else 0.0
    out["cost_usd"] = round(sum(r["cost_usd"] for r in rows), 4)
    out["cost_per_completed"] = round(out["cost_usd"] / len(completed), 5) if completed else None
    out["latency_p50_ms"] = lat[len(lat) // 2] if lat else None
    out["latency_p95_ms"] = lat[int(len(lat) * 0.95) - 1] if lat else None
    return out


def gate_failures(summary: dict) -> list[str]:
    fails = []
    if summary["overall"]["rate"] < GATES["overall"]:
        fails.append(f"overall {summary['overall']['rate']:.0%} < {GATES['overall']:.0%}")
    for qtype in ("trap_unanswerable", "trap_ambiguous"):
        r = summary["qtype"].get(qtype)
        if r and r["rate"] < GATES[qtype]:
            fails.append(f"{qtype} {r['rate']:.0%} < {GATES[qtype]:.0%}")
    return fails


def run_bench(items: list[dict], version: str, live: bool, model: str | None = None, max_cost: float | None = None,
              workers: int = 4, on_progress=None, record: bool = True, include_drafts: bool = False,
              trigger: str = "cli", user_key: str | None = None, save_traces: bool = False) -> dict:
    """Run golden items through the analyst and score them. `on_progress(row)` fires per question; the run stops
    once spend passes `max_cost`. With `save_traces`, every answer is stored as a reviewable 'golden' trace.
    Appends the run to RUNS when `record` and returns it."""
    rows, spent = [], 0.0
    at = datetime.now(UTC).isoformat(timespec="seconds")
    run_id = at if save_traces else None
    with ThreadPoolExecutor(workers if live else 1) as pool:
        for r in pool.map(lambda it: run_one(it, version, live, model, run_id), items):
            rows.append(r)
            spent += r["cost_usd"]
            if on_progress:
                on_progress(r)
            if max_cost is not None and spent > max_cost:
                print(f"stopping: ${spent:.3f} > max cost ${max_cost:.3f}", file=sys.stderr)
                pool.shutdown(wait=False, cancel_futures=True)
                break
    summary = summarize(rows)
    rec = {"at": at, "version": version,
           "model": model or (settings.agent_model if live else "sim"), "live": live,
           "reviewed_only": not include_drafts, "include_drafts": include_drafts, "trigger": trigger,
           "user_key": user_key, "summary": summary, "gate_failures": gate_failures(summary), "rows": rows}
    if record:
        RUNS.parent.mkdir(parents=True, exist_ok=True)
        with RUNS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return rec


def latest_run(live_only: bool = False) -> dict | None:
    """The most recent recorded run (a live one if any exist, unless there are none and live_only is False)."""
    if not RUNS.exists():
        return None
    runs = [json.loads(x) for x in RUNS.read_text(encoding="utf-8").splitlines() if x.strip()]
    live = [r for r in runs if r.get("live")]
    pool = live if (live or live_only) else runs
    return pool[-1] if pool else None


def markdown(meta: dict, summary: dict, rows: list[dict], fails: list[str]) -> str:
    o = summary["overall"]
    lines = [f"### Analyst benchmark: `{meta['version']}` on `{meta['model']}` "
             f"({'reviewed' if meta['reviewed_only'] else 'includes unreviewed drafts'}, n={o['n']})", "",
             f"**Correctness {o['rate']:.1%}** (95% CI {o['ci'][0]:.0%}–{o['ci'][1]:.0%}) · completion "
             f"{summary['completion_rate']:.0%} · ${summary['cost_usd']:.3f} total · "
             f"${summary['cost_per_completed'] or 0:.4f}/completed · p50 {(summary['latency_p50_ms'] or 0) / 1000:.1f}s",
             "", "| slice | n | correct | 95% CI |", "|---|---|---|---|"]
    for dim in ("difficulty", "qtype", "language"):
        for k, v in summary[dim].items():
            lines.append(f"| {dim}: {k} | {v['n']} | {v['rate']:.0%} | {v['ci'][0]:.0%}–{v['ci'][1]:.0%} |")
    lines += ["", "**Gate:** " + ("✅ pass" if not fails else "❌ " + "; ".join(fails))]
    bad = [r for r in rows if not r["passed"]][:15]
    if bad:
        lines += ["", "<details><summary>Failures</summary>", ""]
        lines += [f"- **{r['id']}** ({r['qtype']}) {r['question']} → {r['headline'] or r['error']} · "
                  f"`{json.dumps(r['details'], ensure_ascii=False, default=str)[:160]}`" for r in bad]
        lines += ["", "</details>"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="a1")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--include-drafts", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-cost", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    statuses = ("approved", "draft") if args.include_drafts else ("approved",)
    items = golden.load(statuses)[: args.limit]
    if args.selftest:
        res = golden.selftest(items or golden.load(("approved", "draft")))
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(1 if res["problems"] else 0)
    if not items:
        print("no approved golden items (approve some on the Review page, or pass --include-drafts)", file=sys.stderr)
        sys.exit(2)
    spent = [0.0]

    def progress(r: dict) -> None:
        spent[0] += r["cost_usd"]
        print(f"{r['id']} {'✓' if r['passed'] else '✗'} ${spent[0]:.3f}", file=sys.stderr, flush=True)

    rec = run_bench(items, args.version, args.live, model=args.model, max_cost=args.max_cost, workers=args.workers,
                    on_progress=progress, record=args.record, include_drafts=args.include_drafts)
    md = markdown(rec, rec["summary"], rec["rows"], rec["gate_failures"])
    print(md)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8").write(md + "\n")
    if args.gate and rec["gate_failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
