"""The interface every reranker implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

from shopsense.retrieval.base import Hit


class Reranker(ABC):
    """Re-orders a retriever's hits for a query, usually with a slower, better model."""

    @abstractmethod
    def index(self, catalog: pd.DataFrame) -> None:
        """Keep whatever the reranker needs from the catalog (e.g. product text)."""

    @abstractmethod
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        """Return the same products, best first, scored by the reranker."""
