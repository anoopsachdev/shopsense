"""Query intent: turn a raw shopping query into structured fields with the intent model.

    "nike running shoes not white" ->
    {"product_type": "running shoes", "brand": "nike", "color": null,
     "attributes": [], "exclusions": ["white"]}

The model must answer in JSON matching INTENT_SCHEMA (google-genai response schema).
If the call fails or the answer doesn't parse, the extractor returns a neutral intent
(the whole query as product_type, nothing else) so search still works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from shopsense.llm.client import LLMClient, LLMError

INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "product_type": {"type": "string"},
        "brand": {"type": ["string", "null"]},
        "color": {"type": ["string", "null"]},
        "attributes": {"type": "array", "items": {"type": "string"}},
        "exclusions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product_type", "brand", "color", "attributes", "exclusions"],
}

SYSTEM_PROMPT = """You parse e-commerce search queries into JSON.
- product_type: the kind of product, in a few words (e.g. "running shoes", "phone case").
- brand: the brand of the product itself, only if the query names it, else null. A
  brand the product is compatible with is an attribute: "iphone 14 case" has brand
  null and attribute "iphone 14"; "otterbox iphone 14 case" has brand "otterbox".
- color: only a color the query names explicitly, else null.
- attributes: other wanted properties (size, material, model, quantity, audience).
- exclusions: things the shopper does NOT want, from "without X", "no X", "not X",
  "except X". Use the bare term X. A compound like "sugar free" or "wireless" describes
  the product: put it in attributes, not exclusions.
Use lowercase. Leave lists empty when nothing applies."""


@dataclass(frozen=True)
class QueryIntent:
    """Structured fields of one query. All text is lowercase."""

    product_type: str
    brand: str | None = None
    color: str | None = None
    attributes: tuple[str, ...] = field(default_factory=tuple)
    exclusions: tuple[str, ...] = field(default_factory=tuple)


def neutral_intent(query: str) -> QueryIntent:
    """The intent used when extraction fails: the query itself, no other fields."""
    return QueryIntent(product_type=query.strip().lower())


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected a string, got {value!r}")
    return value.strip().lower() or None


def _clean_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"expected a list, got {value!r}")
    cleaned = [_clean_text(item) for item in value]
    return tuple(dict.fromkeys(item for item in cleaned if item))  # dedupe, keep order


def parse_intent(text: str, query: str) -> QueryIntent:
    """Parse the model's JSON answer. Raises ValueError if it doesn't match the schema."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"not JSON: {text[:80]!r}") from err
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {payload!r}")
    missing = set(INTENT_SCHEMA["required"]) - set(payload)
    if missing:
        raise ValueError(f"missing fields {sorted(missing)}")
    return QueryIntent(
        product_type=_clean_text(payload["product_type"]) or neutral_intent(query).product_type,
        brand=_clean_text(payload["brand"]),
        color=_clean_text(payload["color"]),
        attributes=_clean_list(payload["attributes"]),
        exclusions=_clean_list(payload["exclusions"]),
    )


class QueryIntentExtractor:
    """Calls the intent model and keeps counts and cost for evaluation."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zero the call, failure and cost counters."""
        self.stats = {"calls": 0, "failures": 0, "cost_usd": 0.0}

    def extract(self, query: str) -> QueryIntent:
        """The query's intent, or `neutral_intent(query)` if the call or parsing fails.

        BudgetExceeded is not caught: it stops the run.
        """
        self.stats["calls"] += 1
        try:
            response = self.client.generate(
                f"Query: {query}", system=SYSTEM_PROMPT, response_schema=INTENT_SCHEMA
            )
        except LLMError:
            self.stats["failures"] += 1
            return neutral_intent(query)
        self.stats["cost_usd"] += response.cost_usd
        try:
            return parse_intent(response.text, query)
        except ValueError:
            self.stats["failures"] += 1
            return neutral_intent(query)
