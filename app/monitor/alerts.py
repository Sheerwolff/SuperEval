"""Evaluate rules and keep the alerts table in sync (open new, resolve cleared)."""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timedelta

from sqlalchemy import func, select

from app.config import settings
from app.db import Alert, Deployment, IngestRun, SessionLocal, Trace, utcnow
from app.evals.calibration import agreement
from app.evals.judge_monitor import load_canaries, load_judge_frame
from app.monitor.aggregates import load_evals, load_traces
from app.monitor.rules import DEFAULT_RULES, Context, Firing

log = logging.getLogger("monitor")


def freshness_as_of(session, now: datetime) -> dict:
    rows = session.execute(
        select(IngestRun.chain, func.max(IngestRun.latest_price_update))
        .where(IngestRun.ok.is_(True), IngestRun.created_at <= now)
        .group_by(IngestRun.chain)
    ).all()
    chains = {c for (c,) in session.execute(select(IngestRun.chain).distinct()).all()}
    out = {c: None for c in chains}
    for chain, published in rows:
        out[chain] = datetime.fromisoformat(published[:16]) if published else None
    return out


def spend_since(session, since: datetime, until: datetime | None = None) -> float:
    q = select(func.coalesce(func.sum(Trace.cost_usd), 0.0)).where(Trace.created_at >= since, Trace.model != "sim")
    if until is not None:
        q = q.where(Trace.created_at < until)
    return float(session.execute(q).scalar_one())


def deployed_as_of(session, now: datetime) -> str | None:
    dep = session.execute(
        select(Deployment).where(Deployment.created_at <= now)
        .order_by(Deployment.created_at.desc(), Deployment.id.desc()).limit(1)
    ).scalar_one_or_none()
    return dep.agent_version if dep else None


def build_context(session, now: datetime) -> Context:
    since = now - timedelta(days=8)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return Context(
        now=now,
        traces=load_traces(since, now, session=session),
        evals=load_evals(since, now, session=session),
        freshness=freshness_as_of(session, now),
        spend_today=spend_since(session, day_start, now),
        budget=settings.daily_budget_usd,
        freshness_sla_hours=settings.freshness_sla_hours,
        deployed=deployed_as_of(session, now),
        judge=load_judge_frame(now - timedelta(days=8), now, session=session),
        canaries=load_canaries(now - timedelta(days=2), now, session=session),
        kappa=agreement(),
    )


def run_monitors(now: datetime | None = None, notify: bool = True, ctx: Context | None = None) -> dict:
    """`ctx` lets the backfill pass pre-sliced in-memory frames instead of re-querying."""
    now = now or utcnow()
    with SessionLocal() as s:
        ctx = ctx or build_context(s, now)
        firing: dict[tuple, Firing] = {}
        for rule in DEFAULT_RULES:
            try:
                for f in rule.check(ctx):
                    firing[f.key] = f
            except Exception:
                log.exception("rule %s failed", getattr(rule, "name", rule))
        open_alerts = s.execute(select(Alert).where(Alert.status == "open")).scalars().all()
        open_by_key = {(a.rule, a.metric, a.agent_version): a for a in open_alerts}
        opened, resolved = [], []
        for key, f in firing.items():
            if key in open_by_key:
                a = open_by_key[key]
                a.value, a.message = f.value, f.message
                continue
            a = Alert(created_at=now, rule=f.rule, severity=f.severity, metric=f.metric, layer=f.layer,
                      agent_version=f.agent_version, value=f.value, threshold=f.threshold, message=f.message)
            s.add(a)
            opened.append(a)
        for key, a in open_by_key.items():
            if key not in firing:
                a.status, a.resolved_at = "resolved", now
                resolved.append(a)
        s.commit()
    if notify and settings.alert_webhook_url:
        for a in opened:
            _notify(a)
    return {"opened": len(opened), "resolved": len(resolved), "firing": len(firing)}


def _notify(alert: Alert) -> None:
    """Slack- and Discord-compatible payload."""
    text = f"[{alert.severity.upper()}] {alert.message}"
    body = json.dumps({"text": text, "content": text}).encode()
    req = urllib.request.Request(settings.alert_webhook_url, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        log.warning("alert webhook failed", exc_info=True)
