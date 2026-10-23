"""BM25 retrieval over product title, brand, color and bullet points (rank-bm25)."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from shopsense.retrieval.base import Hit, Retriever, catalog_texts


def tokenize(text: str) -> list[str]:
    """Lowercase and split on anything that isn't a letter or digit."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(Retriever):
    """Okapi BM25 with rank-bm25's default parameters (k1=1.5, b=0.75)."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._product_ids: list[str] = []
        self._position: dict[str, int] = {}

    def index(self, catalog: pd.DataFrame) -> None:
        """Tokenize each product's text and build the BM25 index."""
        self._product_ids = catalog["product_id"].tolist()
        self._position = {pid: i for i, pid in enumerate(self._product_ids)}
        corpus = [tokenize(text) for text in catalog_texts(catalog)]
        self._bm25 = BM25Okapi(corpus)

    def search(
        self, query: str, k: int, candidate_ids: Sequence[str] | None = None
    ) -> list[Hit]:
        """Top-k products by BM25 score. Ties (including all-zero scores) keep catalog order.

        Candidates missing from the index are ignored.
        """
        if self._bm25 is None:
            raise RuntimeError("call index() before search()")
        tokens = tokenize(query)
        if candidate_ids is None:
            positions = np.arange(len(self._product_ids))
            scores = self._bm25.get_scores(tokens)
        else:
            positions = np.array(
                [self._position[pid] for pid in candidate_ids if pid in self._position],
                dtype=int,
            )
            scores = np.asarray(self._bm25.get_batch_scores(tokens, positions.tolist()))
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self._product_ids[positions[i]], float(scores[i])) for i in order]
