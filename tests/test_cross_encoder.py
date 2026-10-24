import pandas as pd
import pytest

from shopsense.rerank import cross_encoder
from shopsense.rerank.cross_encoder import CrossEncoderReranker
from shopsense.retrieval.base import Hit

CATALOG = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3"],
        "product_title": ["red shoes", "blue shoes", "coffee mug"],
        "product_brand": ["Acme", None, None],
    }
)


def word_overlap_scorer(pairs):
    """Fake cross-encoder: number of query words found in the product text."""
    return [len(set(query.split()) & set(text.lower().split())) for query, text in pairs]


@pytest.fixture
def reranker():
    reranker = CrossEncoderReranker(scorer=word_overlap_scorer)
    reranker.index(CATALOG)
    return reranker


def test_rerank_sorts_by_cross_encoder_score(reranker):
    hits = [Hit("p3", 9.0), Hit("p2", 8.0), Hit("p1", 7.0)]
    reranked = reranker.rerank("red shoes", hits)
    assert reranked == [Hit("p1", 2.0), Hit("p2", 1.0), Hit("p3", 0.0)]


def test_rerank_ties_keep_input_order(reranker):
    reranked = reranker.rerank("shoes", [Hit("p2", 0.0), Hit("p1", 0.0)])
    assert [hit.product_id for hit in reranked] == ["p2", "p1"]


def test_rerank_scores_product_text_with_all_fields():
    seen = []
    reranker = CrossEncoderReranker(scorer=lambda pairs: seen.extend(pairs) or [0.0])
    reranker.index(CATALOG)
    reranker.rerank("q", [Hit("p1", 1.0)])
    assert seen == [("q", "red shoes Acme")]


def test_rerank_of_no_hits_does_not_load_a_model():
    assert CrossEncoderReranker().rerank("q", []) == []


def test_max_length_defaults_to_256_and_reaches_the_loader(monkeypatch):
    loaded = []

    def fake_loader(model_name, max_length):
        loaded.append((model_name, max_length))
        return word_overlap_scorer

    monkeypatch.setattr(cross_encoder, "load_cross_encoder", fake_loader)
    cases = [(CrossEncoderReranker(), 256), (CrossEncoderReranker(max_length=128), 128)]
    for reranker, expected in cases:
        reranker.index(CATALOG)
        reranker.rerank("shoes", [Hit("p1", 0.0)])
        assert loaded[-1] == (cross_encoder.DEFAULT_MODEL, expected)
