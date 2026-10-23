import math

import pytest

from shopsense.metrics import dcg, gain, mean, ndcg_at_k, recall_at_k

LABELS = {"a": "E", "b": "S", "c": "I", "d": "C"}


def test_gain_maps_labels_and_unlabelled():
    assert [gain(label) for label in ["E", "S", "C", "I", None]] == [3, 2, 1, 0, 0]


def test_dcg_discounts_by_log2_rank():
    # 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2618595 + 0.5
    assert dcg([3, 2, 1]) == pytest.approx(4.7618595, abs=1e-6)


def test_ndcg_hand_computed_example_1():
    # Ranking c, a, b has gains 0, 3, 2.
    #   DCG@3  = 0/1 + 3/log2(3) + 2/log2(4) = 1.8927893 + 1.0 = 2.8927893
    # Ideal top 3 of the labels is E, S, C with gains 3, 2, 1.
    #   IDCG@3 = 3/1 + 2/log2(3) + 1/log2(4) = 3 + 1.2618595 + 0.5 = 4.7618595
    #   NDCG@3 = 2.8927893 / 4.7618595 = 0.6074915
    assert ndcg_at_k(["c", "a", "b", "x"], LABELS, k=3) == pytest.approx(0.6074915, abs=1e-6)


def test_ndcg_hand_computed_example_2_unlabelled_product_is_irrelevant():
    # "x" is unlabelled, so gain 0 at rank 1; "a" (Exact, gain 3) is at rank 2.
    #   DCG@2  = 0/1 + 3/log2(3) = 1.8927893
    #   IDCG@2 = 3/1 = 3 (the query has only one labelled product)
    #   NDCG@2 = 1.8927893 / 3 = 0.6309298
    assert ndcg_at_k(["x", "a"], {"a": "E"}, k=2) == pytest.approx(0.6309298, abs=1e-6)


def test_ndcg_is_one_for_ideal_ranking():
    assert ndcg_at_k(["a", "b", "d", "c"], LABELS, k=10) == pytest.approx(1.0)


def test_ndcg_is_zero_when_query_has_no_relevant_product():
    assert ndcg_at_k(["a"], {"a": "I"}, k=10) == 0.0


def test_recall_counts_products_with_positive_gain():
    # Relevant are a (E), b (S) and d (C); c is Irrelevant.
    assert recall_at_k(["c", "a"], LABELS, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(["c", "a", "b", "d"], LABELS, k=4) == 1.0
    assert recall_at_k(["a", "b", "d"], LABELS, k=1) == pytest.approx(1 / 3)


def test_recall_is_zero_when_query_has_no_relevant_product():
    assert recall_at_k(["a"], {"a": "I"}, k=10) == 0.0


def test_mean():
    assert mean([0.5, 1.0]) == 0.75
    assert mean([]) == 0.0
    assert math.isclose(mean([1 / 3] * 3), 1 / 3)
