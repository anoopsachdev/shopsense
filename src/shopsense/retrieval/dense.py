"""Dense retrieval: bge-small embeddings (fastembed, ONNX on CPU) and cosine similarity.

Product vectors are embedded once by scripts/build_index.py and saved to data/, so
evaluation only embeds queries.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from shopsense.retrieval.base import Hit, Retriever, catalog_texts

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_VECTORS_PATH = Path("data") / "dense_bge-small-en-v1.5.npz"


class Encoder(Protocol):
    """Turns text into vectors. Real runs use FastEmbedEncoder; tests use a fake."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEncoder:
    """fastembed text embedding model, loaded on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """One row per text."""
        vectors = self._load().passage_embed(texts, batch_size=self.batch_size)
        return np.vstack(list(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        """A single query vector."""
        return next(iter(self._load().query_embed(text)))


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row (or a single vector) to unit length, leaving zero vectors as zeros."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def save_vectors(path: Path, product_ids: list[str], vectors: np.ndarray) -> None:
    """Save product IDs and their vectors to one .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, product_ids=np.array(product_ids), vectors=vectors.astype(np.float32))


def load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    """Load (product_ids, vectors) saved by `save_vectors`."""
    with np.load(path) as saved:
        return saved["product_ids"].tolist(), saved["vectors"]


class DenseRetriever(Retriever):
    """Cosine similarity between the query vector and every product vector."""

    def __init__(self, encoder: Encoder | None = None, vectors_path: Path | None = None) -> None:
        self.encoder = encoder or FastEmbedEncoder()
        self.vectors_path = vectors_path
        self._product_ids: list[str] = []
        self._position: dict[str, int] = {}
        self._vectors: np.ndarray | None = None

    def index(self, catalog: pd.DataFrame) -> None:
        """Load saved vectors for this catalog, or embed the catalog (and save, if a path is set).

        Raises ValueError if the saved vectors were built for a different catalog.
        """
        product_ids = catalog["product_id"].tolist()
        if self.vectors_path is not None and self.vectors_path.exists():
            saved_ids, vectors = load_vectors(self.vectors_path)
            if saved_ids != product_ids:
                raise ValueError(
                    f"{self.vectors_path} was built for a different catalog; "
                    "rerun scripts/build_index.py"
                )
        else:
            vectors = normalize(self.encoder.embed_documents(catalog_texts(catalog)))
            if self.vectors_path is not None:
                save_vectors(self.vectors_path, product_ids, vectors)
        self._product_ids = product_ids
        self._position = {pid: i for i, pid in enumerate(product_ids)}
        self._vectors = vectors

    def search(
        self, query: str, k: int, candidate_ids: Sequence[str] | None = None
    ) -> list[Hit]:
        """Top-k products by cosine similarity. Ties keep catalog order.

        Candidates missing from the index are ignored.
        """
        if self._vectors is None:
            raise RuntimeError("call index() before search()")
        if candidate_ids is None:
            positions = np.arange(len(self._product_ids))
        else:
            positions = np.array(
                [self._position[pid] for pid in candidate_ids if pid in self._position],
                dtype=int,
            )
        query_vector = normalize(np.asarray(self.encoder.embed_query(query), dtype=np.float32))
        scores = self._vectors[positions] @ query_vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self._product_ids[positions[i]], float(scores[i])) for i in order]
