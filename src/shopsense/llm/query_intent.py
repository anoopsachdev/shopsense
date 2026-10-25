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
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from shopsense.llm.client import LLMClient, LLMError
from shopsense.retrieval.base import Hit

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
    product_type = _clean_text(payload["product_type"]) or neutral_intent(query).product_type
    return QueryIntent(
        product_type=product_type,
        brand=_clean_text(payload["brand"]),
        color=_clean_text(payload["color"]),
        attributes=_clean_list(payload["attributes"]),
        exclusions=_clean_list(payload["exclusions"]),
    )


class QueryIntentExtractor:
    """Calls the intent model and keeps counts and cost for evaluation (thread-safe)."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self._lock = threading.Lock()
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zero the call, failure and cost counters."""
        self.stats = {"calls": 0, "failures": 0, "cost_usd": 0.0}

    def extract(self, query: str) -> QueryIntent:
        """The query's intent, or `neutral_intent(query)` if the call or parsing fails.

        BudgetExceeded is not caught: it stops the run.
        """
        try:
            response = self.client.generate(
                f"Query: {query}", system=SYSTEM_PROMPT, response_schema=INTENT_SCHEMA
            )
        except LLMError:
            self._record(0.0, failed=True)
            return neutral_intent(query)
        try:
            intent = parse_intent(response.text, query)
        except ValueError:
            self._record(response.cost_usd, failed=True)
            return neutral_intent(query)
        self._record(response.cost_usd, failed=False)
        return intent

    def _record(self, cost: float, failed: bool) -> None:
        with self._lock:
            self.stats["calls"] += 1
            self.stats["cost_usd"] += cost
            self.stats["failures"] += int(failed)


def mentions(term: str, text: str) -> bool:
    """True if `term` appears in `text` as whole words, ignoring case."""
    return re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text.lower()) is not None


class IntentBooster:
    """Pipeline stage: re-orders retrieved hits using the query's intent.

    Drops hits whose title mentions an exclusion, then moves hits that match the
    intent's brand and color up: hits matching both first, then one, then neither,
    keeping retriever order within each group.
    """

    def __init__(self, extractor: QueryIntentExtractor) -> None:
        self.extractor = extractor
        self._fields: dict[str, tuple[str, str, str]] = {}

    def extract(self, query: str) -> QueryIntent:
        """The query's intent (see QueryIntentExtractor.extract)."""
        return self.extractor.extract(query)

    @property
    def stats(self) -> dict[str, float]:
        """The extractor's call, failure and cost counters."""
        return self.extractor.stats

    def reset_stats(self) -> None:
        """Zero the extractor's counters."""
        self.extractor.reset_stats()

    def index(self, catalog: pd.DataFrame) -> None:
        """Keep each product's title, brand and color."""
        def text(column: str) -> list[str]:
            if column not in catalog:
                return [""] * len(catalog)
            return catalog[column].fillna("").astype(str).tolist()

        self._fields = dict(
            zip(
                catalog["product_id"],
                zip(text("product_title"), text("product_brand"), text("product_color")),
            )
        )

    def apply(
        self, query: str, hits: Sequence[Hit], intent: QueryIntent | None = None
    ) -> list[Hit]:
        """Hits without excluded products, brand/color matches first.

        Extracts the query's intent unless `intent` is given.
        """
        intent = self.extractor.extract(query) if intent is None else intent

        def excluded(hit: Hit) -> bool:
            title = self._fields[hit.product_id][0]
            return any(mentions(term, title) for term in intent.exclusions)

        kept = [hit for hit in hits if not excluded(hit)]

        def matches(hit: Hit) -> int:
            title, brand, color = self._fields[hit.product_id]
            brand_match = intent.brand is not None and (
                mentions(intent.brand, brand) or mentions(intent.brand, title)
            )
            color_match = intent.color is not None and (
                mentions(intent.color, color) or mentions(intent.color, title)
            )
            return int(brand_match) + int(color_match)

        return sorted(kept, key=lambda hit: -matches(hit))  # stable
