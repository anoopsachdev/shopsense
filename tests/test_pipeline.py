import pandas as pd
import pytest

from shopsense.pipeline import (
    PIPELINES,
    Pipeline,
    build_pipeline,
    get_pipeline,
)
from shopsense.rerank.base import Reranker
from shopsense.rerank.cross_encoder import CrossEncoderReranker
from shopsense.retrieval.base import Hit, Retriever
from shopsense.retrieval.bm25 import BM25Retriever
from shopsense.retrieval.dense import DenseRetriever
from shopsense.retrieval.hybrid import HybridRetriever


class ListRetriever(Retriever):
    """Returns products a1..a5 in order (restricted to candidates if given)."""

    def __init__(self):
        self.indexed = False
        self.asked_k = None

    def index(self, catalog):
        self.indexed = True

    def search(self, query, k, candidate_ids=None):
        self.asked_k = k
        ids = [f"a{i}" for i in range(1, 6)]
        ids = [p for p in ids if candidate_ids is None or p in candidate_ids]
        return [Hit(p, 10.0 - i) for i, p in enumerate(ids[:k])]


class ReverseReranker(Reranker):
    """Reverses whatever it is given."""

    def __init__(self):
        self.indexed = False
        self.seen = None

    def index(self, catalog):
        self.indexed = True

    def rerank(self, query, hits):
        self.seen = [hit.product_id for hit in hits]
        return [Hit(hit.product_id, float(i)) for i, hit in enumerate(reversed(hits))]


FAKE_RETRIEVERS = {"list": ListRetriever}
FAKE_RERANKERS = {"reverse": ReverseReranker}


def build(config):
    return build_pipeline(config, retrievers=FAKE_RETRIEVERS, rerankers=FAKE_RERANKERS)


def test_build_retriever_only_pipeline():
    pipeline = build({"retriever": "list"})
    assert isinstance(pipeline.retriever, ListRetriever)
    assert pipeline.reranker is None
    assert [hit.product_id for hit in pipeline.search("q", k=2)] == ["a1", "a2"]


def test_build_pipeline_with_reranker_and_depth():
    pipeline = build({"retriever": "list", "reranker": "reverse", "rerank_depth": 3})
    assert isinstance(pipeline.reranker, ReverseReranker)
    assert pipeline.rerank_depth == 3


def test_reranker_sees_only_top_depth_and_deeper_hits_follow():
    pipeline = build({"retriever": "list", "reranker": "reverse", "rerank_depth": 3})
    hits = pipeline.search("q", k=5)
    assert pipeline.reranker.seen == ["a1", "a2", "a3"]
    assert [hit.product_id for hit in hits] == ["a3", "a2", "a1", "a4", "a5"]


def test_pipeline_retrieves_at_least_rerank_depth():
    pipeline = build({"retriever": "list", "reranker": "reverse", "rerank_depth": 4})
    assert len(pipeline.search("q", k=2)) == 2
    assert pipeline.retriever.asked_k == 4


def test_pipeline_passes_candidates_through():
    pipeline = build({"retriever": "list", "reranker": "reverse"})
    hits = pipeline.search("q", k=10, candidate_ids=["a2", "a4"])
    assert [hit.product_id for hit in hits] == ["a4", "a2"]


def test_index_reaches_every_stage():
    pipeline = build({"retriever": "list", "reranker": "reverse"})
    pipeline.index(pd.DataFrame({"product_id": []}))
    assert pipeline.retriever.indexed and pipeline.reranker.indexed


@pytest.mark.parametrize(
    "config, message",
    [
        ({"retriever": "nope"}, "unknown retriever 'nope'"),
        ({}, "unknown retriever None"),
        ({"retriever": "list", "reranker": "nope"}, "unknown reranker 'nope'"),
        ({"retriever": "list", "rerank_k": 5}, "unknown config keys"),
    ],
)
def test_build_pipeline_rejects_bad_configs(config, message):
    with pytest.raises(ValueError, match=message):
        build(config)


def test_named_pipelines_build_with_the_real_stages():
    # Building is cheap: models load lazily, on first index or search.
    assert list(PIPELINES) == [
        "bm25",
        "dense",
        "hybrid",
        "hybrid+cross_encoder",
        "hybrid+cross_encoder_fast",
    ]
    assert isinstance(get_pipeline("bm25").retriever, BM25Retriever)
    assert isinstance(get_pipeline("dense").retriever, DenseRetriever)
    hybrid = get_pipeline("hybrid").retriever
    assert isinstance(hybrid, HybridRetriever)
    assert [type(r) for r in hybrid.retrievers] == [BM25Retriever, DenseRetriever]
    reranked = get_pipeline("hybrid+cross_encoder")
    assert isinstance(reranked, Pipeline)
    assert isinstance(reranked.reranker, CrossEncoderReranker)
    assert reranked.rerank_depth == 50
    fast = get_pipeline("hybrid+cross_encoder_fast")
    assert isinstance(fast.retriever, HybridRetriever)
    assert isinstance(fast.reranker, CrossEncoderReranker)
    assert fast.rerank_depth == 20


def test_get_pipeline_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown pipeline"):
        get_pipeline("bm26")
