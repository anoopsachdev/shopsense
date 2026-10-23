"""Compose a retriever and an optional reranker from a config dict.

A config names its stages, so swapping a stage is a config change:

    {"retriever": "hybrid", "reranker": "cross_encoder", "rerank_depth": 50}

`PIPELINES` holds the named configs that scripts/evaluate.py runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from shopsense.rerank.base import Reranker
from shopsense.rerank.cross_encoder import CrossEncoderReranker
from shopsense.retrieval.base import Hit, Retriever
from shopsense.retrieval.bm25 import BM25Retriever
from shopsense.retrieval.dense import DEFAULT_VECTORS_PATH, DenseRetriever
from shopsense.retrieval.hybrid import HybridRetriever

DEFAULT_RERANK_DEPTH = 50

RETRIEVERS: dict[str, Callable[[], Retriever]] = {
    "bm25": BM25Retriever,
    "dense": lambda: DenseRetriever(vectors_path=DEFAULT_VECTORS_PATH),
    "hybrid": lambda: HybridRetriever(
        [BM25Retriever(), DenseRetriever(vectors_path=DEFAULT_VECTORS_PATH)]
    ),
}

RERANKERS: dict[str, Callable[[], Reranker]] = {
    "cross_encoder": CrossEncoderReranker,
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
}

CONFIG_KEYS = {"retriever", "reranker", "rerank_depth"}


class Pipeline(Retriever):
    """A retriever, optionally followed by a reranker over its top `rerank_depth` hits."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        rerank_depth: int = DEFAULT_RERANK_DEPTH,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_depth = rerank_depth

    def index(self, catalog: pd.DataFrame) -> None:
        """Index the catalog in every stage."""
        self.retriever.index(catalog)
        if self.reranker is not None:
            self.reranker.index(catalog)

    def search(
        self, query: str, k: int, candidate_ids: Sequence[str] | None = None
    ) -> list[Hit]:
        """Top-k hits.

        With a reranker, the top `rerank_depth` hits are re-ordered by it and any deeper
        hits follow in retriever order (keeping the retriever's scores).
        """
        if self.reranker is None:
            return self.retriever.search(query, k, candidate_ids)
        hits = self.retriever.search(query, max(k, self.rerank_depth), candidate_ids)
        reranked = self.reranker.rerank(query, hits[: self.rerank_depth])
        return (reranked + hits[self.rerank_depth :])[:k]


def build_pipeline(
    config: Mapping[str, Any],
    retrievers: Mapping[str, Callable[[], Retriever]] = RETRIEVERS,
    rerankers: Mapping[str, Callable[[], Reranker]] = RERANKERS,
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
    return Pipeline(retriever, reranker, config.get("rerank_depth", DEFAULT_RERANK_DEPTH))


def get_pipeline(name: str) -> Pipeline:
    """Build one of the named pipelines in `PIPELINES`."""
    return build_pipeline(_lookup(PIPELINES, name, "pipeline"))


def _lookup(registry: Mapping[str, Any], name: Any, kind: str) -> Any:
    if name not in registry:
        raise ValueError(f"unknown {kind} {name!r}; choose from {sorted(registry)}")
    return registry[name]
