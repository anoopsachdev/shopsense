import pandas as pd
import pytest

from shopsense import pipeline as pipeline_module
from shopsense.llm.client import LLMResponse
from shopsense.llm.query_intent import IntentBooster
from shopsense.pipeline import (
    PIPELINES,
    Pipeline,
    build_pipeline,
    get_pipeline,
)
from shopsense.rerank.llm import LLMReranker
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


class DropA2Intent:
    """Fake intent stage: drops a2, moves a4 first; counts calls like the real one."""

    def __init__(self):
        self.indexed = False
        self.stats = {"calls": 0, "failures": 0, "cost_usd": 0.0}

    def index(self, catalog):
        self.indexed = True

    def reset_stats(self):
        self.stats = {"calls": 0, "failures": 0, "cost_usd": 0.0}

    def apply(self, query, hits):
        self.stats["calls"] += 1
        self.stats["cost_usd"] += 0.5
        kept = [hit for hit in hits if hit.product_id != "a2"]
        return sorted(kept, key=lambda hit: hit.product_id != "a4")


FAKE_INTENTS = {"drop_a2": DropA2Intent}
ZERO_STATS = {"calls": 0, "cost_usd": 0.0, "rerank_calls": 0, "fallbacks": 0}


def build(config):
    return build_pipeline(
        config, retrievers=FAKE_RETRIEVERS, rerankers=FAKE_RERANKERS, intents=FAKE_INTENTS
    )


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
        "hybrid+llm",
        "intent+hybrid+llm",
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


def test_intent_stage_runs_between_retriever_and_reranker():
    config = {"retriever": "list", "intent": "drop_a2", "reranker": "reverse", "rerank_depth": 3}
    pipeline = build(config)
    hits = pipeline.search("q", k=5)
    assert pipeline.reranker.seen == ["a4", "a1", "a3"]
    assert [hit.product_id for hit in hits] == ["a3", "a1", "a4", "a5"]


def test_intent_stage_without_reranker():
    pipeline = build({"retriever": "list", "intent": "drop_a2"})
    assert [hit.product_id for hit in pipeline.search("q", k=2)] == ["a4", "a1"]


def test_index_reaches_the_intent_stage():
    pipeline = build({"retriever": "list", "intent": "drop_a2"})
    pipeline.index(pd.DataFrame({"product_id": []}))
    assert pipeline.intent.indexed


def test_unknown_intent_is_rejected():
    with pytest.raises(ValueError, match="unknown intent 'nope'"):
        build({"retriever": "list", "intent": "nope"})


def test_llm_stats_sum_intent_and_reranker_and_reset():
    class CountingReranker(ReverseReranker):
        stats = {"calls": 2, "fallbacks": 1, "cost_usd": 0.25}

        def reset_stats(self):
            self.stats = {"calls": 0, "fallbacks": 0, "cost_usd": 0.0}

    pipeline = Pipeline(ListRetriever(), CountingReranker(), 3, DropA2Intent())
    pipeline.search("q", k=3)
    assert pipeline.llm_stats() == {"calls": 3, "cost_usd": 0.75, "rerank_calls": 2, "fallbacks": 1}
    pipeline.reset_llm_stats()
    assert pipeline.llm_stats() == ZERO_STATS


def test_llm_stats_are_zero_without_llm_stages():
    pipeline = build({"retriever": "list", "reranker": "reverse"})
    assert pipeline.llm_stats() == ZERO_STATS
    assert not pipeline.uses_llm


def test_llm_pipelines_build_with_their_clients(monkeypatch):
    made = []

    class FakeClient:
        def generate(self, prompt, system=None, response_schema=None):
            return LLMResponse("{}", 0, 0, 0.0, 0.0, 0.0, False)

    def fake_make_client(role, **kwargs):
        made.append((role, kwargs))
        return FakeClient()

    monkeypatch.setattr(pipeline_module.llm_client, "make_client", fake_make_client)
    plain = get_pipeline("hybrid+llm")
    assert isinstance(plain.reranker, LLMReranker)
    assert isinstance(plain.reranker.fallback, CrossEncoderReranker)
    assert plain.rerank_depth == 20 and plain.intent is None and plain.uses_llm
    with_intent = get_pipeline("intent+hybrid+llm")
    assert isinstance(with_intent.intent, IntentBooster)
    assert isinstance(with_intent.retriever, HybridRetriever)
    assert made == [
        ("rerank", {"timeout_s": 60.0, "max_retries": 1}),
        ("rerank", {"timeout_s": 60.0, "max_retries": 1}),
        ("intent", {"timeout_s": 15.0}),
    ]


def test_get_pipeline_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown pipeline"):
        get_pipeline("bm26")
