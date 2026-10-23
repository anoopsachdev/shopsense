import pandas as pd
import pytest

from shopsense.retrieval.base import Hit
from shopsense.retrieval.bm25 import BM25Retriever, product_text, tokenize

CATALOG = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3", "p4"],
        "product_title": ["Red running shoes", "Blue rain jacket", "Coffee mug", "Desk lamp"],
        "product_brand": ["Acme", "Stormy", None, "Brightco"],
        "product_color": ["red", "navy", "white", None],
        "product_bullet_point": ["Lightweight mesh", None, "Holds 12oz of coffee", "LED bulb"],
        "product_description": ["shoes shoes shoes", None, None, None],
    }
)


@pytest.fixture
def retriever():
    bm25 = BM25Retriever()
    bm25.index(CATALOG)
    return bm25


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Men's Running-Shoes, size 10!") == ["men", "s", "running", "shoes", "size", "10"]


def test_product_text_joins_fields_and_skips_missing_and_description():
    assert product_text(CATALOG.iloc[2]) == "Coffee mug white Holds 12oz of coffee"


def test_search_ranks_matching_product_first(retriever):
    hits = retriever.search("running shoes", k=2)
    assert len(hits) == 2
    assert hits[0].product_id == "p1"
    assert hits[0].score > hits[1].score


def test_search_matches_brand_and_bullet_points(retriever):
    assert retriever.search("stormy", k=1)[0].product_id == "p2"
    assert retriever.search("led", k=1)[0].product_id == "p4"


def test_search_returns_at_most_k_hits(retriever):
    assert len(retriever.search("coffee", k=10)) == 4


def test_search_with_candidates_only_scores_candidates(retriever):
    hits = retriever.search("coffee", k=10, candidate_ids=["p4", "p3", "missing"])
    assert [hit.product_id for hit in hits] == ["p3", "p4"]
    assert hits[1].score == 0.0


def test_search_ties_keep_catalog_order(retriever):
    hits = retriever.search("nothing matches", k=3)
    assert hits == [Hit("p1", 0.0), Hit("p2", 0.0), Hit("p3", 0.0)]


def test_search_before_index_raises():
    with pytest.raises(RuntimeError):
        BM25Retriever().search("shoes", k=1)
