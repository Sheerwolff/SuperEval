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


def run_one(item: dict, version: str, live: bool, model: str | None) -> dict:
    sem = get_semantic()
    if live:
        from app.analyst.agent import run_analyst

        res = run_analyst(item["question"], version, model=model)
    else:
        from app.analyst.sim import run_sim

        res = run_sim(item["question"], version)
    s = scoring.score(item, res.answer, res.queries, sem)
    return {"id": item["id"], "qtype": item["qtype"], "difficulty": item["difficulty"], "language": item["language"],
            "answer_type": item["answer_type"], "passed": s.passed, "score": s.score, "details": s.details,
            "completed": res.completed, "error": res.error, "cost_usd": res.cost_usd if live else 0.0,
            "latency_ms": res.latency_ms, "steps": res.steps, "question": item["question"],
            "headline": (res.answer or {}).get("headline"), "trace_id": res.id}


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
        print("no golden items (approve some in the review queue, or pass --include-drafts)", file=sys.stderr)
        sys.exit(2)
    model = args.model or (settings.agent_model if args.live else "sim")
    rows, spent = [], 0.0
    with ThreadPoolExecutor(args.workers if args.live else 1) as pool:
        for r in pool.map(lambda it: run_one(it, args.version, args.live, args.model), items):
            rows.append(r)
            spent += r["cost_usd"]
            print(f"{r['id']} {'✓' if r['passed'] else '✗'} ${spent:.3f}", file=sys.stderr, flush=True)
            if spent > args.max_cost:
                print(f"stopping: ${spent:.3f} > --max-cost", file=sys.stderr)
                break
    summary = summarize(rows)
    meta = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "version": args.version, "model": model,
            "live": args.live, "reviewed_only": not args.include_drafts}
    fails = gate_failures(summary)
    md = markdown(meta, summary, rows, fails)
    print(md)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8").write(md + "\n")
    if args.record:
        RUNS.parent.mkdir(parents=True, exist_ok=True)
        with RUNS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**meta, "summary": summary, "rows": rows}, ensure_ascii=False, default=str) + "\n")
    if args.gate and fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
