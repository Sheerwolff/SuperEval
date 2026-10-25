"""Exercises the live OpenAI agent loop against a fake client (no network, no key)."""
import json
from types import SimpleNamespace as NS

from app.agent.agent import OpenAIAgent
from app.agent.versions import get_version
from app.data.catalog import get_catalog
from app.evals.deterministic import run_deterministic
from app.pipeline import to_trace


class FakeMessage(NS):
    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in self.tool_calls or []]}


class FakeCompletions:
    """Turn 1: search. Turn 2: get_prices on the first hit. Turn 3: answer quoting the real price."""

    def __init__(self):
        self.calls = []

    def create(self, model, messages, tools, **kw):
        self.calls.append({"messages": messages, **kw})
        usage = NS(prompt_tokens=900, completion_tokens=40)
        step = len(self.calls)
        if step == 1:
            call = NS(id="c1", function=NS(name="search_products", arguments=json.dumps({"query": "חלב"})))
            msg = FakeMessage(content=None, tool_calls=[call])
        elif step == 2:
            hits = json.loads(messages[-1]["content"])
            args = {"item_code": hits[0]["item_code"], "city": "Haifa"}
            msg = FakeMessage(content=None, tool_calls=[NS(id="c2", function=NS(name="get_prices",
                                                                                 arguments=json.dumps(args)))])
        else:
            offer = json.loads(messages[-1]["content"])["offers"][0]
            msg = FakeMessage(content=f"The cheapest is at {offer['chain']} for ₪{offer['price']:.2f}, "
                                      f"prices as of today.", tool_calls=None)
        return NS(choices=[NS(message=msg)], usage=usage)


def test_openai_agent_loop_produces_grounded_trace():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent.catalog, agent.model = get_catalog(), "gpt-5-mini"
    agent.tools = []
    fake = FakeCompletions()
    agent.client = NS(chat=NS(completions=fake))

    res = agent.run("What's the cheapest milk in Haifa?", get_version("v1"))

    assert res.error is None
    assert [c["name"] for c in res.tool_calls] == ["search_products", "get_prices"]
    assert res.input_tokens == 2700 and res.output_tokens == 120 and res.cost_usd > 0
    assert fake.calls[0].get("reasoning_effort")  # reasoning model: effort sent, temperature not
    assert "temperature" not in fake.calls[0]
    outcomes = {o.evaluator: o.passed for o in run_deterministic(to_trace(res, "test", None))}
    assert outcomes["price_accuracy"] and outcomes["grounded_prices"] and outcomes["tool_use"]


def test_openai_agent_records_tool_errors():
    agent = OpenAIAgent.__new__(OpenAIAgent)
    agent.catalog, agent.model, agent.tools = get_catalog(), "gpt-4.1-mini", []

    class BadTool(FakeCompletions):
        def create(self, model, messages, tools, **kw):
            self.calls.append(kw)
            if len(self.calls) == 1:
                call = NS(id="x", function=NS(name="delete_db", arguments="{}"))
                return NS(choices=[NS(message=FakeMessage(content=None, tool_calls=[call]))], usage=None)
            return NS(choices=[NS(message=FakeMessage(content="Sorry.", tool_calls=None))], usage=None)

    fake = BadTool()
    agent.client = NS(chat=NS(completions=fake))
    res = agent.run("hi", get_version("v1"))
    assert "unknown tool" in res.tool_calls[0]["error"]
    assert fake.calls[0]["temperature"] == 0.2  # non-reasoning model gets the version's temperature
