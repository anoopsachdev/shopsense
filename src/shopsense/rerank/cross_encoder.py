"""Cross-encoder reranking with an MS MARCO MiniLM model (sentence-transformers, CPU)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from shopsense.rerank.base import Reranker
from shopsense.retrieval.base import Hit, catalog_texts

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Tokens per (query, product text) pair. The model allows 512; 256 roughly halves the
# cost per pair and still covers the title, brand, color and the start of the bullets.
DEFAULT_MAX_LENGTH = 256

# Scores (query, product text) pairs; higher means more relevant.
PairScorer = Callable[[list[tuple[str, str]]], Sequence[float]]


def load_cross_encoder(
    model_name: str = DEFAULT_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 32,
) -> PairScorer:
    """Load a sentence-transformers CrossEncoder on CPU and return its scoring function.

    Pairs longer than `max_length` tokens are truncated.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device="cpu", max_length=max_length)
    return lambda pairs: model.predict(pairs, batch_size=batch_size, show_progress_bar=False)


class CrossEncoderReranker(Reranker):
    """Scores each (query, product text) pair jointly and sorts by that score."""

    def __init__(
        self,
        scorer: PairScorer | None = None,
        model_name: str = DEFAULT_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._scorer = scorer
        self._texts: dict[str, str] = {}

    def index(self, catalog: pd.DataFrame) -> None:
        """Keep each product's text (same fields as the retrievers)."""
        self._texts = dict(zip(catalog["product_id"], catalog_texts(catalog)))

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        """Hits re-sorted by cross-encoder score, best first. Ties keep the input order."""
        if not hits:
            return []
        if self._scorer is None:
            self._scorer = load_cross_encoder(self.model_name, self.max_length)
        pairs = [(query, self._texts[hit.product_id]) for hit in hits]
        scores = np.asarray(self._scorer(pairs), dtype=float)
        order = np.argsort(-scores, kind="stable")
        return [Hit(hits[i].product_id, float(scores[i])) for i in order]
