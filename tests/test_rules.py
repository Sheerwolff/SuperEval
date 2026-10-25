from datetime import datetime, timedelta

import pandas as pd

from app.monitor.rules import BudgetRule, Context, FreshnessRule, PSIDriftRule, ThresholdRule, ZScoreRule, psi, wilson

NOW = datetime(2026, 10, 25, 12)


def frames(n_traces, pass_rate, version="v1", start=timedelta(hours=2), latency=3000.0, topic="price_lookup"):
    ts = [NOW - start + timedelta(minutes=i * (start.total_seconds() / 60) / max(n_traces, 1))
          for i in range(n_traces)]
    traces = pd.DataFrame({"id": [f"t{i}" for i in range(n_traces)], "created_at": ts, "agent_version": version,
                           "latency_ms": latency, "cost_usd": 0.001, "error": None, "topic": topic, "language": "en"})
    n_pass = round(n_traces * pass_rate)
    evals = pd.DataFrame({"trace_id": traces.id, "created_at": ts, "evaluator": "price_accuracy",
                          "score": [1.0] * n_pass + [0.0] * (n_traces - n_pass),
                          "passed": [True] * n_pass + [False] * (n_traces - n_pass),
                          "agent_version": version, "topic": topic, "language": "en"})
    return traces, evals


def ctx(traces, evals, **kw):
    base = dict(now=NOW, traces=traces, evals=evals, freshness={}, spend_today=0.0, budget=0.0,
                freshness_sla_hours=36, deployed="v1")
    base.update(kw)
    return Context(**base)


def test_wilson_bounds():
    lo, hi = wilson(0.5, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    assert wilson(1.0, 10)[1] == 1.0


def test_threshold_ignores_single_miss_on_small_sample():
    tr, ev = frames(15, 14 / 15)  # 93% on n=15: below the 95% SLO, but not significantly
    assert ThresholdRule("price_accuracy", "<", 0.95).check(ctx(tr, ev)) == []


def test_threshold_fires_on_clear_regression_for_deployed_version_only():
    tr, ev = frames(30, 0.3, version="v2")
    rule = ThresholdRule("price_accuracy", "<", 0.95)
    fired = rule.check(ctx(tr, ev, deployed="v2"))
    assert len(fired) == 1 and fired[0].agent_version == "v2" and fired[0].severity == "critical"
    assert rule.check(ctx(tr, ev, deployed="v1")) == []  # rolled back -> resolves


def test_latency_threshold():
    tr, ev = frames(20, 1.0, latency=12000)
    assert ThresholdRule("latency_p95_ms", ">", 8000).check(ctx(tr, ev))


def test_zscore_needs_effect_size():
    hist_t, hist_e = frames(7 * 24 * 4, 1.0, start=timedelta(days=7), latency=5000)
    hist_t["latency_ms"] = 5000 + (hist_t.index % 7) * 50  # some natural variance
    cur_t, cur_e = frames(20, 1.0, start=timedelta(hours=2), latency=15000)
    tr = pd.concat([hist_t[hist_t.created_at < NOW - timedelta(hours=3)], cur_t])
    ev = pd.concat([hist_e[hist_e.created_at < NOW - timedelta(hours=3)], cur_e])
    assert ZScoreRule("latency_p95_ms", direction="up").check(ctx(tr, ev))
    cur_t["latency_ms"] = 5400  # statistically odd maybe, but not a 30% change
    tr = pd.concat([hist_t[hist_t.created_at < NOW - timedelta(hours=3)], cur_t])
    assert ZScoreRule("latency_p95_ms", direction="up").check(ctx(tr, ev)) == []


def test_psi():
    a = pd.Series(["x"] * 60 + ["y"] * 40)
    assert psi(a, a) < 1e-9
    assert psi(a, pd.Series(["x"] * 20 + ["y"] * 80)) > 0.2


def test_psi_rule_fires_on_mix_shift():
    base_t, base_e = frames(800, 1.0, start=timedelta(days=8), topic="price_lookup")
    cur_t, cur_e = frames(200, 1.0, start=timedelta(hours=20), topic="basket")
    tr = pd.concat([base_t[base_t.created_at < NOW - timedelta(hours=24)], cur_t])
    assert PSIDriftRule("topic").check(ctx(tr, base_e))


def test_freshness_and_budget():
    c = ctx(*frames(1, 1.0), freshness={"Victory": NOW - timedelta(hours=50), "Shufersal": NOW - timedelta(hours=3),
                                        "Yohananof": None})
    fired = {f.metric for f in FreshnessRule().check(c)}
    assert fired == {"freshness_Victory", "freshness_Yohananof"}
    assert BudgetRule().check(ctx(*frames(1, 1.0), spend_today=1.7, budget=2.0))
    assert BudgetRule().check(ctx(*frames(1, 1.0), spend_today=0.5, budget=2.0)) == []
