"""The interface every first-stage retriever implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

# The product fields every stage reads, in this order (title first, so truncation keeps it).
TEXT_FIELDS = ["product_title", "product_brand", "product_color", "product_bullet_point"]


def product_text(row: pd.Series) -> str:
    """Join the product's text fields, skipping missing ones."""
    return " ".join(str(row[field]) for field in TEXT_FIELDS if pd.notna(row.get(field)))


def catalog_texts(catalog: pd.DataFrame) -> list[str]:
    """`product_text` for every catalog row, in catalog order."""
    return [product_text(row) for _, row in catalog.iterrows()]


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
