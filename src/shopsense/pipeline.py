"""Compose a retriever, an optional intent stage and an optional reranker from a config.

A config names its stages, so swapping a stage is a config change:

    {"retriever": "hybrid", "intent": "llm", "reranker": "llm", "rerank_depth": 20}

`PIPELINES` holds the named configs that scripts/evaluate.py runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from shopsense.llm import client as llm_client
from shopsense.llm.query_intent import IntentBooster, QueryIntent, QueryIntentExtractor
from shopsense.rerank.base import Reranker
from shopsense.rerank.cross_encoder import CrossEncoderReranker
from shopsense.rerank.llm import LLMReranker
from shopsense.retrieval.base import Hit, Retriever
from shopsense.retrieval.bm25 import BM25Retriever
from shopsense.retrieval.dense import DEFAULT_VECTORS_PATH, DenseRetriever
from shopsense.retrieval.hybrid import HybridRetriever

DEFAULT_RERANK_DEPTH = 50
FAST_RERANK_DEPTH = 20
# The rerank model took 20-40 s per 20-product call in testing, so it gets a long
# timeout and one retry before falling back to the cross-encoder.
LLM_RERANK_TIMEOUT_S = 60.0
LLM_INTENT_TIMEOUT_S = 15.0

RETRIEVERS: dict[str, Callable[[], Retriever]] = {
    "bm25": BM25Retriever,
    "dense": lambda: DenseRetriever(vectors_path=DEFAULT_VECTORS_PATH),
    "hybrid": lambda: HybridRetriever(
        [BM25Retriever(), DenseRetriever(vectors_path=DEFAULT_VECTORS_PATH)]
    ),
}

RERANKERS: dict[str, Callable[[], Reranker]] = {
    "cross_encoder": CrossEncoderReranker,
    # Falls back to the cross-encoder over the same top-20 hits: hybrid+cross_encoder_fast.
    "llm": lambda: LLMReranker(
        llm_client.make_client("rerank", timeout_s=LLM_RERANK_TIMEOUT_S, max_retries=1),
        fallback=CrossEncoderReranker(),
    ),
}

INTENTS: dict[str, Callable[[], IntentBooster]] = {
    "llm": lambda: IntentBooster(
        QueryIntentExtractor(llm_client.make_client("intent", timeout_s=LLM_INTENT_TIMEOUT_S))
    ),
}

PIPELINES: dict[str, dict[str, Any]] = {
    "bm25": {"retriever": "bm25"},
    "dense": {"retriever": "dense"},
    "hybrid": {"retriever": "hybrid"},
    "hybrid+cross_encoder": {
        "retriever": "hybrid",
        "reranker": "cross_encoder",
        "rerank_depth": DEFAULT_RERANK_DEPTH,
    },
    # Same stages, but the cross-encoder only re-orders the top 20: ~2.5x fewer pairs.
    "hybrid+cross_encoder_fast": {
        "retriever": "hybrid",
        "reranker": "cross_encoder",
        "rerank_depth": FAST_RERANK_DEPTH,
    },
    "hybrid+llm": {
        "retriever": "hybrid",
        "reranker": "llm",
        "rerank_depth": FAST_RERANK_DEPTH,
    },
    "intent+hybrid+llm": {
        "retriever": "hybrid",
        "intent": "llm",
        "reranker": "llm",
        "rerank_depth": FAST_RERANK_DEPTH,
    },
}

CONFIG_KEYS = {"retriever", "intent", "reranker", "rerank_depth"}


@dataclass(frozen=True)
class SearchDetails:
    """One search's hits, its parsed intent and whether the LLM reranker fell back.

    `intent` is None unless the pipeline has an intent stage.
    """

    hits: list[Hit]
    intent: QueryIntent | None = None
    fell_back: bool = False


class Pipeline(Retriever):
    """A retriever, an optional intent stage over its hits, and an optional reranker
    over the top `rerank_depth` hits.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        rerank_depth: int = DEFAULT_RERANK_DEPTH,
        intent: IntentBooster | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_depth = rerank_depth
        self.intent = intent

    @property
    def uses_llm(self) -> bool:
        """True if any stage calls an LLM (and so costs money per query)."""
        return self.intent is not None or isinstance(self.reranker, LLMReranker)

    def llm_stats(self) -> dict[str, float]:
        """Summed LLM calls, cost and fallbacks of the LLM stages since the last reset."""
        totals = {"calls": 0, "cost_usd": 0.0, "rerank_calls": 0, "fallbacks": 0}
        for stage in (self.intent, self.reranker):
            stats = getattr(stage, "stats", None)
            if stats is None:
                continue
            totals["calls"] += stats["calls"]
            totals["cost_usd"] += stats["cost_usd"]
            if "fallbacks" in stats:
                totals["rerank_calls"] += stats["calls"]
                totals["fallbacks"] += stats["fallbacks"]
        return totals

    def reset_llm_stats(self) -> None:
        """Zero the LLM stages' counters."""
        for stage in (self.intent, self.reranker):
            if hasattr(stage, "reset_stats"):
                stage.reset_stats()

    def index(self, catalog: pd.DataFrame) -> None:
        """Index the catalog in every stage."""
        self.retriever.index(catalog)
        if self.intent is not None:
            self.intent.index(catalog)
        if self.reranker is not None:
            self.reranker.index(catalog)

    def search(
        self, query: str, k: int, candidate_ids: Sequence[str] | None = None
    ) -> list[Hit]:
        """Top-k hits.

        The intent stage, if any, drops and re-orders the retrieved hits. With a
        reranker, the top `rerank_depth` hits are then re-ordered by it and any deeper
        hits follow (keeping their earlier scores).
        """
        return self.search_details(query, k, candidate_ids).hits

    def search_details(
        self, query: str, k: int, candidate_ids: Sequence[str] | None = None
    ) -> SearchDetails:
        """`search`, plus this query's parsed intent and LLM fallback flag."""
        if self.reranker is None and self.intent is None:
            return SearchDetails(self.retriever.search(query, k, candidate_ids))
        hits = self.retriever.search(query, max(k, self.rerank_depth), candidate_ids)
        intent = None
        if self.intent is not None:
            intent = self.intent.extract(query)
            hits = self.intent.apply(query, hits, intent)
        if self.reranker is None:
            return SearchDetails(hits[:k], intent)
        top = hits[: self.rerank_depth]
        if hasattr(self.reranker, "rerank_with_status"):
            reranked, fell_back = self.reranker.rerank_with_status(query, top)
        else:
            reranked, fell_back = self.reranker.rerank(query, top), False
        return SearchDetails((reranked + hits[self.rerank_depth :])[:k], intent, fell_back)


def build_pipeline(
    config: Mapping[str, Any],
    retrievers: Mapping[str, Callable[[], Retriever]] = RETRIEVERS,
    rerankers: Mapping[str, Callable[[], Reranker]] = RERANKERS,
    intents: Mapping[str, Callable[[], IntentBooster]] = INTENTS,
) -> Pipeline:
    """Build a pipeline from a config dict. Raises ValueError for unknown keys or names."""
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"unknown config keys {sorted(unknown)}; allowed: {sorted(CONFIG_KEYS)}"
        )
    retriever = _lookup(retrievers, config.get("retriever"), "retriever")()
    reranker = None
    if config.get("reranker") is not None:
        reranker = _lookup(rerankers, config["reranker"], "reranker")()
    intent = None
    if config.get("intent") is not None:
        intent = _lookup(intents, config["intent"], "intent")()
    depth = config.get("rerank_depth", DEFAULT_RERANK_DEPTH)
    return Pipeline(retriever, reranker, depth, intent)


def get_pipeline(name: str) -> Pipeline:
    """Build one of the named pipelines in `PIPELINES`."""
    return build_pipeline(_lookup(PIPELINES, name, "pipeline"))


def build_shared_pipelines(names: Sequence[str]) -> dict[str, Pipeline]:
    """Build named pipelines that share stage objects, for a server that loads each once.

    They share one BM25 index, one set of dense vectors, one cross-encoder (also the
    LLM reranker's fallback) and one LLM client per role.
    """
    made: dict[str, Any] = {}

    def once(key: str, factory: Callable[[], Any]) -> Callable[[], Any]:
        def get() -> Any:
            if key not in made:
                made[key] = factory()
            return made[key]

        return get

    bm25 = once("bm25", BM25Retriever)
    dense = once("dense", lambda: DenseRetriever(vectors_path=DEFAULT_VECTORS_PATH))
    cross_encoder = once("cross_encoder", CrossEncoderReranker)
    retrievers = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": once("hybrid", lambda: HybridRetriever([bm25(), dense()])),
    }
    rerankers = {
        "cross_encoder": cross_encoder,
        "llm": once(
            "llm_reranker",
            lambda: LLMReranker(
                llm_client.make_client("rerank", timeout_s=LLM_RERANK_TIMEOUT_S, max_retries=1),
                fallback=cross_encoder(),
            ),
        ),
    }
    intents = {"llm": once("intent", INTENTS["llm"])}
    return {
        name: build_pipeline(_lookup(PIPELINES, name, "pipeline"), retrievers, rerankers, intents)
        for name in names
    }


def _lookup(registry: Mapping[str, Any], name: Any, kind: str) -> Any:
    if name not in registry:
        raise ValueError(f"unknown {kind} {name!r}; choose from {sorted(registry)}")
    return registry[name]
