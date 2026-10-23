"""Hybrid retrieval: reciprocal rank fusion (RRF) of several retrievers' rankings."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from shopsense.retrieval.base import Hit, Retriever

DEFAULT_RRF_K = 60
DEFAULT_DEPTH = 100


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], rrf_k: int = DEFAULT_RRF_K
) -> list[Hit]:
    """Fuse ranked lists of product IDs: score(p) = sum over lists of 1 / (rrf_k + rank).

    Ranks start at 1; a product missing from a list gets nothing from it. Returns every
    product best first, with ties in order of first appearance.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, product_id in enumerate(ranking, start=1):
            scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (rrf_k + rank)
    # sorted() is stable, so ties keep insertion (first-appearance) order.
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return [Hit(product_id, score) for product_id, score in ordered]


class HybridRetriever(Retriever):
    """Runs each retriever to `depth` results and fuses the rankings with RRF."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = DEFAULT_RRF_K,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.depth = depth

    def index(self, catalog: pd.DataFrame) -> None:
        """Index the catalog in every underlying retriever."""
        for retriever in self.retrievers:
            retriever.index(catalog)

    def search(
        self, query: str, k: int, candidate_ids: Sequence[str] | None = None
    ) -> list[Hit]:
        """Top-k fused hits. Each retriever contributes its top max(k, depth) results."""
        depth = max(k, self.depth)
        rankings = [
            [hit.product_id for hit in retriever.search(query, depth, candidate_ids)]
            for retriever in self.retrievers
        ]
        return reciprocal_rank_fusion(rankings, self.rrf_k)[:k]
