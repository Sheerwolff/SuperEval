"""Correctness scorers for the analyst benchmark: compare an analyst answer with the result of a
golden item's reference SQL, computed live on the current snapshot (golden *questions*, not
golden answers). One scorer per answer type:

  scalar      a claim (or headline number) within 1% of the reference value
  ranking     the reference top-1 entity is the first such entity named in the headline;
              Kendall tau reported when the agent's cited query ranks the same column
  set         F1 between the reference top-k entities and those named in the answer (pass >= 0.6)
  decline     the answer is declined
  assumption  an ambiguous question: answered (not declined) with at least one stated assumption
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.textutil import CHAIN_HE, CITY_HE

SCALAR_TOL = 0.01
SET_PASS_F1 = 0.6


@dataclass
class Score:
    passed: bool
    score: float
    details: dict = field(default_factory=dict)


def reference(item, semantic) -> tuple[list[str], list[list]] | None:
    """(columns, rows) of the item's reference SQL on the current snapshot, or None if it can't run."""
    if not item["reference_sql"]:
        return None
    r = semantic.run(item["reference_sql"], "ref")
    if r.error or not r.rows:
        return None
    return r.columns, r.rows


def _aliases(value: str) -> list[str]:
    v = str(value)
    out = [v.lower()]
    if v in CHAIN_HE:
        out.append(CHAIN_HE[v])
    if v in CITY_HE:
        out.append(CITY_HE[v])
    return out


def _first_position(text: str, value) -> int | None:
    low = text.lower()
    hits = [low.find(a) for a in _aliases(value) if low.find(a) >= 0]
    return min(hits) if hits else None


def _numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*\.?\d*", text or "")]


def _close(a: float, b: float) -> bool:
    for scale in (1, 100, 0.01):
        x = a * scale
        if abs(x - b) <= max(SCALAR_TOL * abs(b), 0.011):
            return True
    return False


def score_scalar(item, ans: dict, ref) -> Score:
    cols, rows = ref
    col = item["answer_key"].get("column", cols[0])
    expected = rows[0][cols.index(col)]
    candidates = [c["value"] for c in ans.get("claims", [])] + _numbers(ans.get("headline", ""))
    hit = next((v for v in candidates if v is not None and _close(float(v), float(expected))), None)
    return Score(hit is not None, float(hit is not None), {"expected": expected, "found": hit,
                                                          "candidates": candidates[:12]})


def score_ranking(item, ans: dict, ref, queries: list[dict]) -> Score:
    cols, rows = ref
    col = item["answer_key"].get("column", cols[0])
    order = [r[cols.index(col)] for r in rows]
    # entities tied with the top on the ranked value are all correct answers
    vcol = item["answer_key"].get("value_column")
    if vcol in cols:
        top = rows[0][cols.index(vcol)]
        accepted = [r[cols.index(col)] for r in rows if r[cols.index(vcol)] == top]
    else:
        accepted = [order[0]]
    headline = ans.get("headline", "")
    positions = {v: _first_position(headline, v) for v in order}
    if not any(p is not None for p in positions.values()):  # e.g. "four chains tie", named in the claims
        text = " ".join(c["text"] for c in ans.get("claims", []))
        positions = {v: _first_position(text, v) for v in order}
    named = sorted((p, v) for v, p in positions.items() if p is not None)
    first = named[0][1] if named else None
    tau = _kendall_from_queries(order, col, queries)
    return Score(first in accepted, float(first in accepted),
                 {"expected": accepted if len(accepted) > 1 else accepted[0], "first_named": first,
                  "kendall_tau": tau})


def score_set(item, ans: dict, ref) -> Score:
    cols, rows = ref
    col = item["answer_key"].get("column", cols[0])
    k = int(item["answer_key"].get("k", 5))
    expected = [str(r[cols.index(col)]) for r in rows[:k]]
    universe = [str(r[cols.index(col)]) for r in rows[: max(k * 4, 20)]]
    text = " ".join([ans.get("headline", "")] + [c["text"] for c in ans.get("claims", [])])
    named = {v for v in universe if _first_position(text, v) is not None}
    tp = len(named & set(expected))
    precision = tp / len(named) if named else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return Score(f1 >= SET_PASS_F1, round(f1, 3), {"expected": expected, "named": sorted(named)})


def _kendall_from_queries(order: list, col: str, queries: list[dict]) -> float | None:
    for q in queries:
        if q.get("error") or col not in (q.get("columns") or []):
            continue
        theirs = [r[q["columns"].index(col)] for r in q["rows"]]
        common = [v for v in order if v in theirs]
        if len(common) < 3:
            continue
        pos = {v: theirs.index(v) for v in common}
        agree = disagree = 0
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                if pos[common[i]] < pos[common[j]]:
                    agree += 1
                else:
                    disagree += 1
        return round((agree - disagree) / (agree + disagree), 3)
    return None


def score(item: dict, ans: dict | None, queries: list[dict], semantic) -> Score:
    """item: a GoldenItem as dict; ans: the analyst's structured answer (or None if it failed)."""
    at = item["answer_type"]
    if ans is None:
        return Score(False, 0.0, {"reason": "no answer"})
    if at == "decline":
        ok = bool(ans.get("declined"))
        return Score(ok, float(ok), {"declined": ans.get("declined")})
    if at == "assumption":
        ok = not ans.get("declined") and bool(ans.get("assumptions"))
        return Score(ok, float(ok), {"assumptions": ans.get("assumptions"), "declined": ans.get("declined")})
    if ans.get("declined"):
        return Score(False, 0.0, {"reason": "declined an answerable question"})
    ref = reference(item, semantic)
    if ref is None:
        return Score(False, 0.0, {"reason": "reference SQL returned nothing on this snapshot"})
    if at == "scalar":
        return score_scalar(item, ans, ref)
    if at == "ranking":
        return score_ranking(item, ans, ref, queries)
    if at == "set":
        return score_set(item, ans, ref)
    raise ValueError(f"unknown answer_type {at}")
