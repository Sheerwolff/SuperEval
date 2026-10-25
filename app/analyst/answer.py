"""The analyst's structured answer. Every number the user sees is a claim tied to a query,
which is what makes the answer evaluable (grounding re-executes the cited query)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str = Field(description="One sentence stating the fact, in the user's language")
    value: float = Field(description="The number, exactly as it appears in the query result")
    unit: str = Field(description="e.g. '₪', '%', 'products', 'stores', 'index'")
    query_id: str = Field(description="Id of the run_sql result the value comes from, e.g. 'q2'")


class ChartSpec(BaseModel):
    type: Literal["bar", "horizontal_bar", "line", "scatter"]
    query_id: str
    x: str = Field(description="Column for the x axis (category or number)")
    y: str = Field(description="Numeric column for the y axis")
    series: str | None = Field(description="Optional column to color by, else null")
    title: str


class Answer(BaseModel):
    headline: str = Field(description="The direct answer in one or two sentences, in the user's language")
    claims: list[Claim] = Field(description="Every number used in the answer, each tied to a query")
    method: str = Field(description="How it was computed, in plain language, one or two sentences")
    assumptions: list[str] = Field(description="Interpretations chosen for an ambiguous question; [] if none")
    caveats: list[str] = Field(description="Limits worth knowing (sample size, snapshot date, coverage); [] if none")
    chart: ChartSpec | None = Field(description="A chart of one query result when it helps, else null")
    declined: bool = Field(description="True when the data cannot answer the question or it is out of scope")


def render_markdown(a: Answer | dict, he: bool = False) -> str:
    """Plain-language rendering shown in chat (the structured form stays in the trace)."""
    a = a if isinstance(a, dict) else a.model_dump()
    parts = [a["headline"]]
    if a.get("claims") and not a.get("declined"):
        parts.append("\n".join(f"- {c['text']}" for c in a["claims"]))
    labels = ("שיטה", "הנחות", "הסתייגויות") if he else ("Method", "Assumptions", "Caveats")
    if a.get("method") and not a.get("declined"):
        parts.append(f"**{labels[0]}:** {a['method']}")
    if a.get("assumptions"):
        parts.append(f"**{labels[1]}:** " + "; ".join(a["assumptions"]))
    if a.get("caveats"):
        parts.append(f"**{labels[2]}:** " + "; ".join(a["caveats"]))
    return "\n\n".join(parts)
