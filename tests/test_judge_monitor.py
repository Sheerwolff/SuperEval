from datetime import datetime, timedelta

import pandas as pd

from app.evals.judge_monitor import agreement, canary_runs, flip_rate, judge_frame_from, length_bias, recall_on_bad
from app.monitor.rules import CanaryRule, Context, JudgeRateRule

NOW = datetime(2026, 10, 25, 12)


def judged(n_bad: int, judge_catches: float, n_good: int = 100, repeat_flip: float = 0.0, lengths=None):
    """Synthetic eval rows: n_bad known-bad traces (price_accuracy failed), n_good known-good."""
    rows, traces = [], []
    for i in range(n_bad + n_good):
        tid, bad = f"t{i}", i < n_bad
        ts = NOW - timedelta(hours=1, minutes=i % 50)
        rows.append({"trace_id": tid, "created_at": ts, "evaluator": "grounding", "passed": not bad,
                     "details": None})
        caught = bad and (i < round(n_bad * judge_catches))
        ground = not caught
        rows.append({"trace_id": tid, "created_at": ts, "evaluator": "judge_groundedness", "passed": ground,
                     "details": {"judge_model": "m", "raw": 5 if ground else 2}})
        rows.append({"trace_id": tid, "created_at": ts, "evaluator": "judge_helpfulness", "passed": True,
                     "details": {"raw": 4 if lengths is None else lengths[i][1]}})
        if i % 2 == 0:
            flip = (i // 2) < round((n_bad + n_good) / 2 * repeat_flip)
            rows.append({"trace_id": tid, "created_at": ts, "evaluator": "judge_repeat",
                         "passed": (not ground) if flip else ground, "details": {}})
        traces.append({"id": tid, "agent_version": "v1", "language": "he" if i % 2 else "en",
                       "answer_len": 100 if lengths is None else lengths[i][0]})
    evals = pd.DataFrame(rows)
    evals["created_at"] = pd.to_datetime(evals.created_at)
    return judge_frame_from(evals, pd.DataFrame(traces))


def ctx(judge=None, canaries=None):
    empty = pd.DataFrame(columns=["created_at"])
    return Context(now=NOW, traces=empty, evals=empty, freshness={}, spend_today=0, budget=0,
                   freshness_sla_hours=36, judge=judge, canaries=canaries)


def test_healthy_judge_has_high_recall_and_agreement():
    j = judged(n_bad=20, judge_catches=0.95)
    assert recall_on_bad(j)[0] >= 0.9 and recall_on_bad(j)[1] == 20
    assert agreement(j)[0] > 0.95
    rule = JudgeRateRule("judge_recall_on_bad", "<", 0.80, window=timedelta(hours=72), min_n=10)
    assert rule.check(ctx(j)) == []


def test_lenient_judge_trips_layer2_but_not_on_few_samples():
    lenient = judged(n_bad=20, judge_catches=0.15)
    rule = JudgeRateRule("judge_recall_on_bad", "<", 0.80, window=timedelta(hours=72), min_n=10)
    fired = rule.check(ctx(lenient))
    assert len(fired) == 1 and fired[0].layer == "judge"
    assert rule.check(ctx(judged(n_bad=5, judge_catches=0.0))) == []  # below min_n


def test_flip_rate():
    j = judged(n_bad=0, judge_catches=1, n_good=100, repeat_flip=0.2)
    rate, n = flip_rate(j)
    assert n == 50 and abs(rate - 0.2) < 0.01


def test_length_bias_detects_reward_for_long_answers():
    lengths = [(50 + 10 * i, 1 + i // 20) for i in range(100)]  # longer -> higher helpfulness
    rho, _ = length_bias(judged(0, 1, n_good=100, lengths=lengths))
    assert rho > 0.9
    flat = [(50 + 10 * i, 3 + (i % 2)) for i in range(100)]
    assert abs(length_bias(judged(0, 1, n_good=100, lengths=flat))[0]) < 0.2


def test_canary_rule():
    def run(acc, run_id, hours_ago=2):
        n = 36
        return pd.DataFrame({"created_at": [NOW - timedelta(hours=hours_ago)] * n, "run_id": run_id,
                             "correct": [i < round(acc * n) for i in range(n)], "judge_model": "m"})

    assert CanaryRule().check(ctx(canaries=run(0.97, "a"))) == []
    fired = CanaryRule().check(ctx(canaries=pd.concat([run(0.97, "a", 26), run(0.70, "b")])))
    assert len(fired) == 1 and fired[0].layer == "judge" and "canary" in fired[0].message
    assert CanaryRule().check(ctx(canaries=run(0.5, "old", hours_ago=48))) == []  # stale run ignored
    assert len(canary_runs(run(0.97, "a"))) == 1
