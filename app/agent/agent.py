"""The PriceWatch agent: an OpenAI function-calling loop over the price catalog.

`run_agent` picks the live OpenAI agent when OPENAI_API_KEY is set and falls back to
the deterministic simulator otherwise (see sim.py), so the app always works.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from app.agent.textutil import classify_topic, detect_language
from app.agent.tools import TOOL_NAMES, run_tool, to_json, tool_schemas
from app.agent.versions import AgentVersion, get_version
from app.config import cost_usd, settings
from app.data.catalog import Catalog, get_catalog

MAX_STEPS = 6


@dataclass
class AgentResult:
    question: str
    answer: str
    agent_version: str
    model: str
    prompt_hash: str
    snapshot_date: str
    language: str
    topic: str
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIAgent:
    def __init__(self, catalog: Catalog, model: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key, timeout=60)
        self.catalog = catalog
        self.model = model or settings.agent_model
        self.tools = tool_schemas(catalog)

    def run(self, question: str, version: AgentVersion, history: list[dict] | None = None,
            on_event: Callable[[dict], None] | None = None) -> AgentResult:
        emit = on_event or (lambda _e: None)
        messages = [{"role": "system", "content": version.system_prompt(self.catalog)}]
        messages += (history or [])[-6:]
        messages.append({"role": "user", "content": question})
        result = AgentResult(
            question=question,
            answer="",
            agent_version=version.version,
            model=self.model,
            prompt_hash=version.prompt_hash,
            snapshot_date=self.catalog.snapshot_date,
            language=detect_language(question),
            topic=classify_topic(question),
        )
        extra = {}
        if _is_reasoning_model(self.model):
            extra["reasoning_effort"] = os.getenv("REASONING_EFFORT", "low")
        else:
            extra["temperature"] = version.temperature

        start = time.perf_counter()
        try:
            for _ in range(version.max_steps or MAX_STEPS):
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=self.tools, **extra
                )
                if resp.usage:
                    result.input_tokens += resp.usage.prompt_tokens
                    result.output_tokens += resp.usage.completion_tokens
                msg = resp.choices[0].message
                if not msg.tool_calls:
                    result.answer = msg.content or ""
                    break
                messages.append(msg.model_dump(exclude_none=True))
                for call in msg.tool_calls:
                    record = {"name": call.function.name, "raw_args": call.function.arguments}
                    try:
                        args = json.loads(call.function.arguments or "{}")
                        record["args"] = args
                        emit({"type": "tool_start", "name": call.function.name, "args": args})
                        if call.function.name not in TOOL_NAMES:
                            raise ValueError(f"unknown tool {call.function.name}")
                        out = run_tool(self.catalog, call.function.name, args)
                        record["result"] = out
                        content = to_json(out)
                    except Exception as exc:  # tool errors go back to the model and into the trace
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        content = json.dumps({"error": record["error"]})
                    result.tool_calls.append(record)
                    emit({"type": "tool_end", **summarize_call(record)})
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            else:
                result.error = "max_steps_exceeded"
                result.answer = result.answer or "Sorry, I couldn't finish that request."
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.answer = "Sorry, something went wrong on my side. Please try again."
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result


def summarize_call(record: dict) -> dict:
    """One-line human summary of a tool call, for the live 'thinking' view."""
    name, out = record.get("name"), record.get("result")
    if record.get("error"):
        return {"name": name, "summary": f"error: {record['error']}"}
    if name == "search_products":
        top = (out or [{}])[0]
        text = f"{len(out or [])} matches" + (f", top: {top.get('name_en') or top.get('name')}" if out else "")
    elif name == "get_prices":
        offers = (out or {}).get("offers") or []
        text = (f"{(out or {}).get('n_offers', len(offers))} offers, cheapest ₪{offers[0]['price']:.2f} at "
                f"{offers[0]['chain']}") if offers else "no prices found"
    elif name == "compare_basket":
        res = (out or {}).get("results") or []
        text = f"{len(res)} chains compared" + (f", cheapest {res[0]['chain']} ₪{res[0]['total']:.2f}" if res else "")
    elif name == "list_stores":
        text = f"{len(out or [])} stores"
    else:
        text = ""
    return {"name": name, "args": record.get("args"), "summary": text}


_live_agent: OpenAIAgent | None = None


def run_agent(
    question: str,
    version: str = "v1",
    history: list[dict] | None = None,
    force_sim: bool = False,
    on_event: Callable[[dict], None] | None = None,
    model: str | None = None,
    **sim_kwargs,
) -> AgentResult:
    global _live_agent
    catalog = get_catalog()
    agent_version = get_version(version)
    if settings.live_llm and not force_sim:
        if model and model != settings.agent_model:
            return OpenAIAgent(catalog, model).run(question, agent_version, history, on_event=on_event)
        if _live_agent is None:
            _live_agent = OpenAIAgent(catalog)
        return _live_agent.run(question, agent_version, history, on_event=on_event)
    from app.agent.sim import SimAgent

    res = SimAgent(catalog).run(question, agent_version, **sim_kwargs)
    if on_event:  # the simulator is instant; replay its steps so the UI looks the same
        for call in res.tool_calls:
            on_event({"type": "tool_start", "name": call["name"], "args": call.get("args")})
            on_event({"type": "tool_end", **summarize_call(call)})
    return res
