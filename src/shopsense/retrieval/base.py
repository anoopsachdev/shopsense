"""The interface every first-stage retriever implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Hit:
    """One retrieved product and its retriever score (higher is better)."""

    product_id: str
    score: float


class Retriever(ABC):
    """Indexes a product catalog and returns the best-matching products for a query."""

    @abstractmethod
    def index(self, catalog: pd.DataFrame) -> None:
        """Build the index from catalog rows (one per product, with a product_id column)."""

    @abstractmethod
    def search(
        self, query: str, k: int, candidate_ids: Sequence[str] | None = None
    ) -> list[Hit]:
        """Return up to `k` hits for `query`, best first.

        If `candidate_ids` is given, only those products are scored (the re-ranking
        setting); otherwise the whole catalog is searched.
        """
