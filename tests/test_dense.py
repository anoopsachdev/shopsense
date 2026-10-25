import numpy as np
import pandas as pd
import pytest

from shopsense.retrieval.dense import (
    DenseRetriever,
    load_vectors,
    normalize,
    save_vectors,
)

VOCAB = ["shoes", "jacket", "coffee", "lamp"]

CATALOG = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3", "p4"],
        "product_title": ["running shoes", "rain jacket", "coffee mug", "desk lamp"],
    }
)


class FakeEncoder:
    """Bag-of-words vectors over a tiny vocabulary; counts how often documents are embedded."""

    def __init__(self):
        self.document_calls = 0

    def _vector(self, text):
        return np.array([text.split().count(word) for word in VOCAB], dtype=np.float32)

    def embed_documents(self, texts):
        self.document_calls += 1
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def retriever():
    dense = DenseRetriever(encoder=FakeEncoder(), allow_build=True)
    dense.index(CATALOG)
    return dense


def test_normalize_rows_and_zero_vectors():
    result = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert result.tolist() == [[0.6, 0.8], [0.0, 0.0]]


def test_vectors_round_trip(tmp_path):
    path = tmp_path / "v.npz"
    save_vectors(path, ["a", "b"], np.eye(2))
    ids, vectors = load_vectors(path)
    assert ids == ["a", "b"]
    assert vectors.dtype == np.float32 and vectors.tolist() == [[1, 0], [0, 1]]


def test_search_ranks_most_similar_first(retriever):
    hits = retriever.search("coffee", k=2)
    assert hits[0].product_id == "p3"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == 0.0


def test_search_with_candidates_only_scores_candidates(retriever):
    hits = retriever.search("lamp", k=10, candidate_ids=["p1", "p4", "missing"])
    assert [hit.product_id for hit in hits] == ["p4", "p1"]


def test_index_saves_then_reuses_vectors(tmp_path):
    path = tmp_path / "v.npz"
    encoder = FakeEncoder()
    DenseRetriever(encoder=encoder, vectors_path=path, allow_build=True).index(CATALOG)
    reloaded = DenseRetriever(encoder=encoder, vectors_path=path)
    reloaded.index(CATALOG)
    assert encoder.document_calls == 1
    assert reloaded.search("jacket", k=1)[0].product_id == "p2"


def test_index_rejects_stale_vectors_by_default(tmp_path):
    path = tmp_path / "v.npz"
    DenseRetriever(encoder=FakeEncoder(), vectors_path=path, allow_build=True).index(CATALOG)
    encoder = FakeEncoder()
    with pytest.raises(ValueError, match="different catalog.*build_index.py"):
        DenseRetriever(encoder=encoder, vectors_path=path).index(CATALOG.iloc[:2])
    assert encoder.document_calls == 0


def test_index_rejects_missing_vectors_by_default(tmp_path):
    encoder = FakeEncoder()
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        DenseRetriever(encoder=encoder, vectors_path=tmp_path / "missing.npz").index(CATALOG)
    assert encoder.document_calls == 0


def test_index_without_a_path_needs_allow_build():
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        DenseRetriever(encoder=FakeEncoder()).index(CATALOG)


def test_allow_build_rebuilds_stale_vectors(tmp_path):
    path = tmp_path / "v.npz"
    DenseRetriever(encoder=FakeEncoder(), vectors_path=path, allow_build=True).index(CATALOG)
    smaller = CATALOG.iloc[:2]
    DenseRetriever(encoder=FakeEncoder(), vectors_path=path, allow_build=True).index(smaller)
    assert load_vectors(path)[0] == ["p1", "p2"]


def test_default_retriever_does_not_build():
    assert DenseRetriever(encoder=FakeEncoder()).allow_build is False


def test_search_before_index_raises():
    with pytest.raises(RuntimeError):
        DenseRetriever(encoder=FakeEncoder()).search("shoes", k=1)


def test_indexing_the_same_catalog_again_is_a_no_op():
    encoder = FakeEncoder()
    dense = DenseRetriever(encoder=encoder, allow_build=True)
    dense.index(CATALOG)
    dense.index(CATALOG)
    assert encoder.document_calls == 1
