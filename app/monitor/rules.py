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
    deployed: str | None = None  # version live at `now`; SLOs are judged on what is serving


def _window(df: pd.DataFrame, now: datetime, start: timedelta, end: timedelta = timedelta(0)) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df.created_at >= now - start) & (df.created_at < now - end)]


@dataclass
class ThresholdRule:
    """Fires when a metric over the last `window` crosses a fixed SLO, per agent version."""

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
        out = []
        # only the version that is serving: after a rollback the old version's alerts resolve
        if ctx.deployed:
            versions = [ctx.deployed]
        else:
            serving = _window(tr, ctx.now, timedelta(minutes=30))
            versions = sorted(set(serving.agent_version)) if len(serving) else []
        for version in versions:
            val, n = metric_value(self.metric, ev[ev.agent_version == version], tr[tr.agent_version == version])
            if val is None or n < self.min_n:
                continue
            is_rate = self.metric in METRICS and METRICS[self.metric][1] == "rate"
            if is_rate:
                # a 95% SLO on 15 samples "breaches" on a single miss; require the whole
                # confidence interval to sit on the wrong side of the SLO instead
                lo, hi = wilson(val, n, self.confidence)
                bad = hi < self.threshold if self.op == "<" else lo > self.threshold
                ci = f", {self.confidence:.0%} CI {lo:.0%}-{hi:.0%}"
            else:
                bad = val < self.threshold if self.op == "<" else val > self.threshold
                ci = ""
            if bad:
                label = METRIC_LABELS.get(self.metric, self.metric)
                out.append(Firing(
                    self.name, self.metric, self.severity, val, self.threshold,
                    f"{label} is {_fmt(self.metric, val)} over the last {_dur(self.window)} on {version} "
                    f"(SLO {self.op} {_fmt(self.metric, self.threshold)} breached, n={n}{ci})",
                    version,
                ))
        return out


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
    min_n: int = 120
    severity: str = "info"

    @property
    def name(self) -> str:
        return f"drift:{self.feature}"

    def check(self, ctx: Context) -> list[Firing]:
        cur = _window(ctx.traces, ctx.now, self.window)
        base = _window(ctx.traces, ctx.now, self.baseline + self.window, self.window)
        if len(cur) < self.min_n or len(base) < self.min_n:
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


def psi(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    cats = set(expected.dropna()) | set(actual.dropna())
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    total = 0.0
    for c in cats:
        pe, pa = max(e.get(c, 0.0), eps), max(a.get(c, 0.0), eps)
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


DEFAULT_RULES = [
    ThresholdRule("price_accuracy", "<", 0.95),
    ThresholdRule("grounded_prices", "<", 0.95),
    ThresholdRule("scope_handling", "<", 0.90, severity="warn"),
    ThresholdRule("language_match", "<", 0.90, severity="warn"),
    ThresholdRule("latency_p95_ms", ">", 8000, severity="warn"),
    ThresholdRule("error_rate", ">", 0.05),
    ZScoreRule("judge_helpfulness"),
    ZScoreRule("latency_p95_ms", direction="up"),
    PSIDriftRule("topic"),
    PSIDriftRule("language"),
    FreshnessRule(),
    BudgetRule(),
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
