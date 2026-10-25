"""Tool schemas exposed to the model, and their dispatch onto the catalog."""
from __future__ import annotations

import json

from app.data.catalog import Catalog


def tool_schemas(catalog: Catalog) -> list[dict]:
    city = {"type": "string", "enum": catalog.cities, "description": "City (English name)"}
    chain = {"type": "string", "enum": catalog.chains}
    return [
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": (
                    "Search the product catalog by Hebrew or English name. Every word must match, so use "
                    "few, specific words (e.g. 'milk 3%', 'במבה'). Returns item codes, Hebrew and English "
                    "names, category and size. Narrow with `category` when the word is ambiguous."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Product name or keywords, Hebrew or English"},
                        "category": {"type": "string", "enum": catalog.categories},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_prices",
                "description": "Prices (and promotions) for one product, cheapest first, optionally filtered.",
                "parameters": {
                    "type": "object",
                    "properties": {"item_code": {"type": "string"}, "city": city, "chain": chain},
                    "required": ["item_code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_basket",
                "description": "Total cost of a basket at the cheapest store of each chain in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"item_code": {"type": "string"}, "qty": {"type": "number"}},
                                "required": ["item_code"],
                            },
                        },
                        "city": city,
                    },
                    "required": ["items", "city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_stores",
                "description": "List covered stores, optionally by city and/or chain.",
                "parameters": {"type": "object", "properties": {"city": city, "chain": chain}},
            },
        },
    ]


TOOL_NAMES = {"search_products", "get_prices", "compare_basket", "list_stores"}


def run_tool(catalog: Catalog, name: str, args: dict):
    if name == "search_products":
        return catalog.search_products(args["query"], category=args.get("category"))
    if name == "get_prices":
        return catalog.get_prices(args["item_code"], args.get("city"), args.get("chain"))
    if name == "compare_basket":
        return catalog.compare_basket(args["items"], args["city"])
    if name == "list_stores":
        return catalog.list_stores(args.get("city"), args.get("chain"))
    raise ValueError(f"unknown tool {name}")


def to_json(result) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
