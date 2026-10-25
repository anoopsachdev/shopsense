"""Listwise LLM reranking: the rerank model orders all candidates in one call.

The prompt lists the query and up to `len(hits)` products, numbered 1..n and shortened
to their first `text_chars` characters (title first). The model answers with the
numbers only, comma-separated: "3,1,2,...". Numbers it leaves out keep their retriever
order after the ones it ranked.

The answer used to be JSON ({"ranking": [...]} under a response schema). JSON mode
pretty-printed it one number per line, ~102 output tokens for 20 products against ~50
for the plain list, so the plain list is used and parsed strictly instead.

If the call times out or errors, or the answer is unusable, the fallback reranker
(the cross-encoder by default) orders the same hits; the fallback is counted and the
query recorded. Safe to call from several threads.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence

import pandas as pd

from shopsense.llm.client import LLMClient, LLMError
from shopsense.rerank.base import Reranker
from shopsense.rerank.cross_encoder import CrossEncoderReranker
from shopsense.retrieval.base import Hit, catalog_texts

DEFAULT_TEXT_CHARS = 300

# The whole answer must be numbers separated by commas (spaces allowed).
ANSWER_PATTERN = re.compile(r"\s*\d+(?:\s*,\s*\d+)*\s*")

SYSTEM_PROMPT = """You rank products for an e-commerce search query.
Order the products from most to least relevant:
1. Exact: the product is what the query asks for, matching every stated detail.
2. Substitute: close but differs in a detail (brand, size, color, model), yet could
   still serve.
3. Complement: goes with what was asked for (an accessory or part), but isn't it.
4. Irrelevant: none of the above.
Respect what the query excludes ("without", "no", "not").
Answer with the product numbers only, best first, comma-separated, using every number
exactly once, e.g. 3,1,2. No other text."""


def build_prompt(query: str, texts: Sequence[str]) -> str:
    """The user prompt: the query, then the numbered product texts."""
    lines = [f"Query: {query}", "", "Products:"]
    lines += [f"[{i}] {text}" for i, text in enumerate(texts, start=1)]
    return "\n".join(lines)


def parse_ranking(text: str, n: int) -> list[int]:
    """0-based positions in the model's order, then any it left out in input order.

    Out-of-range and repeated numbers are ignored. Raises ValueError if the answer
    isn't a comma-separated list of numbers or ranks none of the n products.
    """
    if not ANSWER_PATTERN.fullmatch(text):
        raise ValueError(f"expected comma-separated numbers, got {text[:80]!r}")
    numbers = [int(part) for part in text.split(",")]
    ranked = [number - 1 for number in numbers if 1 <= number <= n]
    ranked = list(dict.fromkeys(ranked))
    if not ranked:
        raise ValueError("ranking names none of the products")
    return ranked + [i for i in range(n) if i not in set(ranked)]


class LLMReranker(Reranker):
    """Orders hits with one listwise call to the rerank model, falling back on failure."""

    def __init__(
        self,
        client: LLMClient,
        fallback: Reranker | None = None,
        text_chars: int = DEFAULT_TEXT_CHARS,
    ) -> None:
        self.client = client
        self.fallback = fallback or CrossEncoderReranker()
        self.text_chars = text_chars
        self._texts: dict[str, str] = {}
        self._lock = threading.Lock()
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zero the call, fallback and cost counters and forget recorded fallbacks."""
        self.stats = {"calls": 0, "fallbacks": 0, "cost_usd": 0.0}
        self.fallback_queries: list[str] = []

    def index(self, catalog: pd.DataFrame) -> None:
        """Keep shortened product texts, and index the fallback reranker."""
        texts = catalog_texts(catalog)
        self._texts = {
            pid: text[: self.text_chars] for pid, text in zip(catalog["product_id"], texts)
        }
        self.fallback.index(catalog)

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        """Hits in the model's order, scored n..1; the fallback's order if the call fails.

        BudgetExceeded is not caught: it stops the run.
        """
        return self.rerank_with_status(query, hits)[0]

    def rerank_with_status(self, query: str, hits: Sequence[Hit]) -> tuple[list[Hit], bool]:
        """`rerank`, plus whether this call fell back to the fallback reranker."""
        if not hits:
            return [], False
        prompt = build_prompt(query, [self._texts[hit.product_id] for hit in hits])
        cost = 0.0
        try:
            response = self.client.generate(prompt, system=SYSTEM_PROMPT)
            cost = response.cost_usd
            order = parse_ranking(response.text, len(hits))
        except (LLMError, ValueError):
            self._record(cost, fell_back_on=query)
            return self.fallback.rerank(query, hits), True
        self._record(cost)
        n = len(hits)
        reranked = [Hit(hits[i].product_id, float(n - rank)) for rank, i in enumerate(order)]
        return reranked, False

    def _record(self, cost: float, fell_back_on: str | None = None) -> None:
        with self._lock:
            self.stats["calls"] += 1
            self.stats["cost_usd"] += cost
            if fell_back_on is not None:
                self.stats["fallbacks"] += 1
                self.fallback_queries.append(fell_back_on)
