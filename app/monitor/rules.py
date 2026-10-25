"""Alert rules. Each rule looks at the data as of `now` and returns zero or more firing
conditions; `alerts.run_monitors` turns those into open/resolved Alert rows."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from app.monitor.aggregates import METRIC_LABELS, METRICS, metric_value


@dataclass
class Firing:
    rule: str
    metric: str
    severity: str
    value: float | None
    threshold: float | None
    message: str
    agent_version: str | None = None
    layer: str = "agent"  # agent = layer 1, judge = layer 2

    @property
    def key(self) -> tuple:
        return (self.rule, self.metric, self.agent_version)


@dataclass
class Context:
    now: datetime
    traces: pd.DataFrame  # last 8 days
    evals: pd.DataFrame
    freshness: dict  # chain -> published_at (datetime | None)
    spend_today: float
    budget: float
    freshness_sla_hours: float
    judge: pd.DataFrame | None = None  # layer 2: one row per judged trace (judge_monitor.judge_frame_from)
    canaries: pd.DataFrame | None = None  # layer 2: canary results
    kappa: dict | None = None  # layer 2: judge vs human labels (calibration.agreement)


def _window(df: pd.DataFrame, now: datetime, start: timedelta, end: timedelta = timedelta(0)) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df.created_at >= now - start) & (df.created_at < now - end)]


@dataclass
class ThresholdRule:
    """Fires when a metric over the last `window` crosses a fixed SLO."""

    metric: str
    op: str  # "<" or ">"
    threshold: float
    window: timedelta = timedelta(hours=3)
    min_n: int = 20
    severity: str = "critical"
    confidence: float = 0.99  # for rate metrics: only fire when the breach is statistically clear

    @property
    def name(self) -> str:
        return f"slo:{self.metric}"

    def check(self, ctx: Context) -> list[Firing]:
        ev, tr = _window(ctx.evals, ctx.now, self.window), _window(ctx.traces, ctx.now, self.window)
        val, n = metric_value(self.metric, ev, tr)
        if val is None or n < self.min_n:
            return []
        if self.metric in METRICS and METRICS[self.metric][1] == "rate":
            # a 95% SLO on 15 samples "breaches" on a single miss; require the whole
            # confidence interval to sit on the wrong side of the SLO instead
            lo, hi = wilson(val, n, self.confidence)
            bad = hi < self.threshold if self.op == "<" else lo > self.threshold
            ci = f", {self.confidence:.0%} CI {lo:.0%}-{hi:.0%}"
        else:
            bad = val < self.threshold if self.op == "<" else val > self.threshold
            ci = ""
        if not bad:
            return []
        label = METRIC_LABELS.get(self.metric, self.metric)
        return [Firing(
            self.name, self.metric, self.severity, val, self.threshold,
            f"{label} is {_fmt(self.metric, val)} over the last {_dur(self.window)} "
            f"(SLO {'≥' if self.op == '<' else '≤'} {_fmt(self.metric, self.threshold)} breached, n={n}{ci})",
        )]


@dataclass
class ZScoreRule:
    """Anomaly vs. the metric's own recent history: current window vs. hourly baseline."""

    metric: str
    direction: str = "down"  # which way is bad
    z: float = 3.0
    window: timedelta = timedelta(hours=3)
    baseline: timedelta = timedelta(days=7)
    min_n: int = 10
    severity: str = "warn"
    min_rel_change: float = 0.5  # z alone is too twitchy on small windows; also require a real effect size

    @property
    def name(self) -> str:
        return f"anomaly:{self.metric}"

    def check(self, ctx: Context) -> list[Firing]:
        ev, tr = _window(ctx.evals, ctx.now, self.window), _window(ctx.traces, ctx.now, self.window)
        cur, n = metric_value(self.metric, ev, tr)
        if cur is None or n < self.min_n:
            return []
        hist_ev = _window(ctx.evals, ctx.now, self.baseline, self.window)
        hist_tr = _window(ctx.traces, ctx.now, self.baseline, self.window)
        vals = []
        if self.metric in ("latency_p95_ms", "cost_per_request", "error_rate"):
            for _, g in hist_tr.groupby(hist_tr.created_at.dt.floor(self.window)):
                v, k = metric_value(self.metric, hist_ev.iloc[0:0], g)
                if v is not None and k >= self.min_n:
                    vals.append(v)
        else:
            for _, g in hist_ev.groupby(hist_ev.created_at.dt.floor(self.window)):
                v, k = metric_value(self.metric, g, hist_tr.iloc[0:0])
                if v is not None and k >= self.min_n:
                    vals.append(v)
        if len(vals) < 20:
            return []
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) or 1e-9
        zval = (cur - mu) / sd
        bad = zval < -self.z if self.direction == "down" else zval > self.z
        if not bad or abs(cur - mu) < self.min_rel_change * abs(mu):
            return []
        label = METRIC_LABELS.get(self.metric, self.metric)
        return [Firing(
            self.name, self.metric, self.severity, cur, mu,
            f"{label} is {_fmt(self.metric, cur)} over the last {_dur(self.window)} vs. a 7-day baseline of "
            f"{_fmt(self.metric, mu)} "
            f"(z = {zval:+.1f})",
        )]


@dataclass
class PSIDriftRule:
    """Input drift: population stability index of a categorical feature (topic, language),
    last 24h vs. the previous 7 days. PSI > 0.2 is the classic 'significant shift' cut-off."""

    feature: str
    threshold: float = 0.2
    window: timedelta = timedelta(hours=24)
    baseline: timedelta = timedelta(days=7)
    min_n: int = 60
    severity: str = "info"

    @property
    def name(self) -> str:
        return f"drift:{self.feature}"

    def check(self, ctx: Context) -> list[Firing]:
        cur = _window(ctx.traces, ctx.now, self.window)
        base = _window(ctx.traces, ctx.now, self.baseline + self.window, self.window)
        if len(cur) < self.min_n or len(base) < 5 * self.min_n:  # a thin baseline makes PSI noisy
            return []
        value = psi(base[self.feature], cur[self.feature])
        if value <= self.threshold:
            return []
        shares = cur[self.feature].value_counts(normalize=True).round(2).to_dict()
        return [Firing(
            self.name, f"psi_{self.feature}", self.severity, value, self.threshold,
            f"Input distribution of '{self.feature}' shifted (PSI = {value:.2f}); last 24h mix: {shares}",
        )]


def wilson(p: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    z = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def psi(expected: pd.Series, actual: pd.Series, alpha: float = 1.0) -> float:
    """Population stability index with additive smoothing: a rare category missing from one quiet day
    shouldn't look like a shift (a tiny epsilon makes log(p/eps) explode)."""
    cats = sorted(set(expected.dropna()) | set(actual.dropna()))
    if not cats:
        return 0.0
    e, a = expected.value_counts(), actual.value_counts()
    ne, na, k = len(expected.dropna()), len(actual.dropna()), len(cats)
    total = 0.0
    for c in cats:
        pe = (e.get(c, 0) + alpha) / (ne + alpha * k)
        pa = (a.get(c, 0) + alpha) / (na + alpha * k)
        total += (pa - pe) * math.log(pa / pe)
    return total


@dataclass
class FreshnessRule:
    severity: str = "warn"

    name: str = "freshness"

    def check(self, ctx: Context) -> list[Firing]:
        out = []
        for chain, published in ctx.freshness.items():
            age = None if published is None else (ctx.now - published).total_seconds() / 3600
            if age is None or age > ctx.freshness_sla_hours:
                out.append(Firing(
                    self.name, f"freshness_{chain}", self.severity, age, ctx.freshness_sla_hours,
                    f"{chain} prices are {'missing' if age is None else f'{age:.0f}h old'} "
                    f"(SLA {ctx.freshness_sla_hours:.0f}h). Answers about {chain} may be stale.",
                ))
        return out


@dataclass
class BudgetRule:
    share: float = 0.8
    severity: str = "warn"
    name: str = "budget"

    def check(self, ctx: Context) -> list[Firing]:
        if ctx.budget <= 0 or ctx.spend_today < self.share * ctx.budget:
            return []
        return [Firing(
            self.name, "spend_today", self.severity, ctx.spend_today, ctx.budget,
            f"LLM spend today is ${ctx.spend_today:.2f} of the ${ctx.budget:.2f} daily budget",
        )]


# ---- Layer 2: rules about the judge ---------------------------------------------
@dataclass
class JudgeRateRule:
    """SLO on a judge-quality rate (Wilson-gated, like the agent SLOs)."""

    metric: str  # key of judge_monitor.JUDGE_METRICS
    op: str
    threshold: float
    window: timedelta
    min_n: int
    severity: str = "critical"
    confidence: float = 0.99

    @property
    def name(self) -> str:
        return f"judge:{self.metric}"

    def check(self, ctx: Context) -> list[Firing]:
        from app.evals.judge_monitor import JUDGE_METRICS

        if ctx.judge is None or ctx.judge.empty:
            return []
        label, fn = JUDGE_METRICS[self.metric]
        val, n = fn(_window(ctx.judge, ctx.now, self.window))
        if val is None or n < self.min_n:
            return []
        lo, hi = wilson(val, n, self.confidence)
        bad = hi < self.threshold if self.op == "<" else lo > self.threshold
        if not bad:
            return []
        return [Firing(
            self.name, self.metric, self.severity, val, self.threshold,
            f"Judge: {label.lower()} is {val:.0%} over the last {_dur(self.window)} "
            f"(SLO {self.op} {self.threshold:.0%}, n={n}, {self.confidence:.0%} CI {lo:.0%}-{hi:.0%}). "
            "Layer-1 judge scores may be unreliable.",
            layer="judge",
        )]


@dataclass
class CanaryRule:
    """The judge got known cases wrong in its latest canary run."""

    threshold: float = 0.80
    max_age: timedelta = timedelta(hours=36)
    severity: str = "critical"
    name: str = "judge:canaries"

    def check(self, ctx: Context) -> list[Firing]:
        from app.evals.judge_monitor import canary_runs

        if ctx.canaries is None or ctx.canaries.empty:
            return []
        runs = canary_runs(ctx.canaries[(ctx.canaries.created_at <= ctx.now)
                                        & (ctx.canaries.created_at >= ctx.now - self.max_age)])
        if runs.empty:
            return []
        last = runs.iloc[-1]
        if last.accuracy >= self.threshold:
            return []
        wrong = int(round((1 - last.accuracy) * last.checks))
        return [Firing(
            self.name, "canary_accuracy", self.severity, float(last.accuracy), self.threshold,
            f"Judge got {wrong} of {int(last.checks)} canary checks wrong ({last.accuracy:.0%} accuracy, "
            f"SLO ≥ {self.threshold:.0%}) with judge model {last.judge_model}. The judge changed, not the agent.",
            layer="judge",
        )]


@dataclass
class KappaRule:
    min_n: int = 30
    threshold: float = 0.6
    severity: str = "warn"
    name: str = "judge:human_kappa"

    def check(self, ctx: Context) -> list[Firing]:
        k = ctx.kappa or {}
        out = []
        for dim in ("helpful", "grounded"):
            r = k.get(dim)
            if k.get("n", 0) >= self.min_n and r and r["kappa"] is not None and r["kappa"] < self.threshold:
                out.append(Firing(
                    self.name, f"kappa_{dim}", self.severity, r["kappa"], self.threshold,
                    f"Judge vs human agreement on '{dim}' is κ = {r['kappa']:.2f} (n={k['n']}, target ≥ "
                    f"{self.threshold}). Revisit the rubric before trusting judge trends.",
                    layer="judge",
                ))
        return out


@dataclass
class JudgeScoreDriftRule:
    """Judge helpfulness distribution moved (the agent prompt is fixed, so the judge changed)."""

    threshold: float = 0.25
    window: timedelta = timedelta(hours=24)
    baseline: timedelta = timedelta(days=7)
    min_n: int = 120
    severity: str = "info"
    name: str = "judge:score_drift"

    def check(self, ctx: Context) -> list[Firing]:
        j = ctx.judge
        if j is None or j.empty:
            return []
        j = j[j.help_raw.notna()]
        cur = _window(j, ctx.now, self.window)
        base = _window(j, ctx.now, self.baseline + self.window, self.window)
        if len(cur) < self.min_n or len(base) < self.min_n:
            return []
        value = psi(base.help_raw.astype(int), cur.help_raw.astype(int))
        if value <= self.threshold:
            return []
        return [Firing(self.name, "psi_judge_helpfulness", self.severity, value, self.threshold,
                       f"Judge helpfulness scores shifted (PSI = {value:.2f}) with the agent unchanged.",
                       layer="judge")]


DEFAULT_RULES = [
    # layer 1: the analyst (Wilson-gated: fire only when the breach is statistically clear)
    ThresholdRule("grounding", "<", 0.98),
    ThresholdRule("completion", "<", 0.95),
    ThresholdRule("sql_health", "<", 0.95),
    ThresholdRule("method_lint", "<", 0.90),
    ThresholdRule("decline_check", "<", 0.90, severity="warn"),
    ThresholdRule("latency_p95_ms", ">", 20000, severity="warn"),
    ZScoreRule("judge_helpfulness"),
    ZScoreRule("latency_p95_ms", direction="up"),
    PSIDriftRule("topic", threshold=0.25),
    PSIDriftRule("language", threshold=0.25),
    FreshnessRule(),
    BudgetRule(),
    # layer 2: the judge
    JudgeRateRule("judge_recall_on_bad", "<", 0.80, window=timedelta(hours=72), min_n=10),
    JudgeRateRule("judge_flip_rate", ">", 0.15, window=timedelta(days=7), min_n=20, severity="warn"),
    JudgeRateRule("judge_error_rate", ">", 0.05, window=timedelta(hours=3), min_n=20, severity="warn"),
    CanaryRule(),
    KappaRule(),
]


def _fmt(metric: str, v: float) -> str:
    if metric.endswith("_ms"):
        return f"{v:,.0f} ms"
    if metric == "cost_per_request":
        return f"${v:.4f}"
    if metric.startswith("judge_"):
        return f"{v:.2f}"
    return f"{v:.0%}"


def _dur(td: timedelta) -> str:
    h = td.total_seconds() / 3600
    return f"{h:.0f}h" if h >= 1 else f"{td.total_seconds() / 60:.0f}m"
