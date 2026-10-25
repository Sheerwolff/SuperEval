"""The PriceWatch analyst: an OpenAI function-calling loop that answers analytical questions
by writing read-only SQL over the semantic layer and submitting a structured answer.

Tools: run_sql(sql) -> rows, submit_answer(Answer) -> ends the turn.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import openai
import yaml

from app.agent.textutil import detect_language
from app.analyst.answer import Answer, render_markdown
from app.analyst.semantic import Semantic, get_semantic
from app.config import cost_usd, settings

PROMPTS = Path(__file__).parent / "prompts"
MAX_STEPS = 10


@dataclass(frozen=True)
class AnalystVersion:
    version: str
    label: str
    description: str
    system_template: str
    max_steps: int = MAX_STEPS

    @property
    def prompt_hash(self) -> str:
        import hashlib

        return hashlib.sha256(self.system_template.encode()).hexdigest()[:12]


def load_versions() -> dict[str, AnalystVersion]:
    out = {}
    for path in sorted(PROMPTS.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        out[raw["version"]] = AnalystVersion(raw["version"], raw["label"], raw["description"].strip(), raw["system"],
                                             int(raw.get("max_steps", MAX_STEPS)))
    return out


@dataclass
class AnalystResult:
    question: str
    version: str
    model: str
    prompt_hash: str
    language: str
    answer: dict | None = None  # Answer.model_dump()
    queries: list[dict] = field(default_factory=list)  # QueryResult.to_dict()
    markdown: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    steps: int = 0
    error: str | None = None  # model_error | timeout | max_steps | no_answer: details
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)

    @property
    def completed(self) -> bool:
        return self.answer is not None and self.error is None


def tool_schemas() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": "run_sql",
            "description": ("Run ONE read-only DuckDB SELECT over the tables and macros described in the system prompt. "
                            "Returns columns, up to 200 rows, the total row count and a query id (q1, q2, ...)."),
            "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"],
                           "additionalProperties": False},
            "strict": True}},
        openai.pydantic_function_tool(Answer, name="submit_answer",
                                      description="Submit the final structured answer. Call exactly once, last."),
    ]


def _is_reasoning(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


class Analyst:
    def __init__(self, semantic: Semantic | None = None, model: str | None = None):
        self.sem = semantic or get_semantic()
        self.model = model or settings.agent_model
        self.client = openai.OpenAI(api_key=settings.openai_api_key, timeout=90)
        self.tools = tool_schemas()

    def run(self, question: str, version: AnalystVersion, history: list[dict] | None = None,
            on_event: Callable[[dict], None] | None = None) -> AnalystResult:
        emit = on_event or (lambda _e: None)
        res = AnalystResult(question=question, version=version.version, model=self.model,
                            prompt_hash=version.prompt_hash, language=detect_language(question))
        system = version.system_template.format(schema=self.sem.schema())
        messages = [{"role": "system", "content": system}, *(history or [])[-6:],
                    {"role": "user", "content": question}]
        extra = {"reasoning_effort": os.getenv("REASONING_EFFORT", "low")} if _is_reasoning(self.model) else {}
        start = time.perf_counter()
        try:
            for step in range(version.max_steps):
                res.steps = step + 1
                resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=self.tools,
                                                           tool_choice="required", **extra)
                if resp.usage:
                    res.input_tokens += resp.usage.prompt_tokens
                    res.output_tokens += resp.usage.completion_tokens
                msg = resp.choices[0].message
                if not msg.tool_calls:  # tool_choice=required makes this rare; nudge once
                    messages.append({"role": "user", "content": "Use run_sql, then submit_answer."})
                    continue
                messages.append(msg.model_dump(exclude_none=True))
                for call in msg.tool_calls:
                    name, raw = call.function.name, call.function.arguments or "{}"
                    if name == "submit_answer":
                        try:
                            res.answer = Answer.model_validate_json(raw).model_dump()
                        except Exception as exc:
                            content = json.dumps({"error": f"invalid answer: {exc}"})
                            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                            continue
                        emit({"type": "answer"})
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": "ok"})
                        continue
                    if name != "run_sql":
                        messages.append({"role": "tool", "tool_call_id": call.id,
                                         "content": json.dumps({"error": f"unknown tool {name}"})})
                        continue
                    sql = json.loads(raw).get("sql", "")
                    qid = f"q{len(res.queries) + 1}"
                    emit({"type": "tool_start", "name": "run_sql", "args": {"sql": sql}, "id": qid})
                    q = self.sem.run(sql, qid)
                    res.queries.append(q.to_dict())
                    emit({"type": "tool_end", "name": "run_sql", "id": qid, "args": {"sql": sql},
                          "summary": q.error or f"{q.row_count} row{'s' if q.row_count != 1 else ''}"})
                    payload = q.to_dict()
                    payload["rows"] = payload["rows"][:50]  # keep the context small; the trace keeps 200
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": json.dumps(payload, ensure_ascii=False, default=str)})
                if res.answer is not None:
                    break
            else:
                res.error = "max_steps: no answer submitted"
        except openai.APITimeoutError as exc:
            res.error = f"timeout: {exc}"
        except Exception as exc:
            res.error = f"model_error: {type(exc).__name__}: {exc}"
        res.latency_ms = (time.perf_counter() - start) * 1000
        if res.answer:
            res.markdown = render_markdown(res.answer, he=res.language == "he")
        elif not res.error:
            res.error = "no_answer: model stopped without submitting"
        if not res.markdown:
            res.markdown = ("מצטער, לא הצלחתי להשלים את הניתוח." if res.language == "he"
                            else "Sorry, I couldn't complete that analysis.")
        return res


def run_analyst(question: str, version: str, history: list[dict] | None = None, on_event=None,
                model: str | None = None) -> AnalystResult:
    return Analyst(model=model).run(question, load_versions()[version], history, on_event)
