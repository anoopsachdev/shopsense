import pandas as pd
import pytest

from shopsense.retrieval.base import Hit, Retriever
from shopsense.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


class FixedRetriever(Retriever):
    """Returns a fixed ranking; records what it was asked."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.indexed = None
        self.calls = []

    def index(self, catalog):
        self.indexed = catalog

    def search(self, query, k, candidate_ids=None):
        self.calls.append((query, k, candidate_ids))
        ranking = [p for p in self.ranking if candidate_ids is None or p in candidate_ids]
        return [Hit(p, 1.0) for p in ranking[:k]]


def test_rrf_hand_computed_example():
    # a: 1/61                = 0.0163934
    # b: 1/62 + 1/61         = 0.0161290 + 0.0163934 = 0.0325224
    # c: 1/63 + 1/62         = 0.0158730 + 0.0161290 = 0.0320020
    # d:        1/63         = 0.0158730
    hits = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], rrf_k=60)
    assert [hit.product_id for hit in hits] == ["b", "c", "a", "d"]
    assert [hit.score for hit in hits] == pytest.approx(
        [0.0325224, 0.0320020, 0.0163934, 0.0158730], abs=1e-6
    )


def test_rrf_ties_keep_first_appearance_order():
    # x and y both score 1/(k+1) + 1/(k+2).
    hits = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], rrf_k=60)
    assert [hit.product_id for hit in hits] == ["x", "y"]
    assert hits[0].score == pytest.approx(hits[1].score)


def test_rrf_smaller_k_rewards_top_ranks_more():
    assert reciprocal_rank_fusion([["a"]], rrf_k=0)[0].score == 1.0
    assert reciprocal_rank_fusion([["a"]], rrf_k=60)[0].score == pytest.approx(1 / 61)


def test_rrf_of_no_rankings_is_empty():
    assert reciprocal_rank_fusion([]) == []


def test_hybrid_indexes_every_retriever():
    first, second = FixedRetriever([]), FixedRetriever([])
    catalog = pd.DataFrame({"product_id": ["p1"]})
    HybridRetriever([first, second]).index(catalog)
    assert first.indexed is catalog and second.indexed is catalog


def test_hybrid_fuses_and_truncates_to_k():
    hybrid = HybridRetriever([FixedRetriever(["a", "b", "c"]), FixedRetriever(["b", "c", "d"])])
    assert [hit.product_id for hit in hybrid.search("q", k=2)] == ["b", "c"]


def test_hybrid_asks_each_retriever_for_depth_results_and_passes_candidates():
    first = FixedRetriever(["a", "b"])
    hybrid = HybridRetriever([first], depth=100)
    hits = hybrid.search("q", k=5, candidate_ids=["b"])
    assert first.calls == [("q", 100, ["b"])]
    assert [hit.product_id for hit in hits] == ["b"]
