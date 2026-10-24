import math

import pytest

from shopsense.metrics import (
    ESCI_GRADED_GAINS,
    ESCI_OFFICIAL_GAINS,
    GAIN_SETS,
    dcg,
    gain,
    mean,
    ndcg_at_k,
    recall_at_k,
)

G = ESCI_GRADED_GAINS

LABELS = {"a": "E", "b": "S", "c": "I", "d": "C"}


def test_gain_defaults_to_official_gains():
    assert [gain(label) for label in ["E", "S", "C", "I", None]] == [1.0, 0.1, 0.01, 0, 0]


def test_gain_with_graded_gains():
    assert [gain(label, G) for label in ["E", "S", "C", "I", None]] == [3, 2, 1, 0, 0]


def test_gain_sets_put_official_first():
    assert list(GAIN_SETS) == ["official", "graded"]
    assert GAIN_SETS["official"] is ESCI_OFFICIAL_GAINS


def test_dcg_discounts_by_log2_rank():
    # 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2618595 + 0.5
    assert dcg([3, 2, 1]) == pytest.approx(4.7618595, abs=1e-6)


def test_ndcg_hand_computed_example_1_graded_gains():
    # Ranking c, a, b has gains 0, 3, 2.
    #   DCG@3  = 0/1 + 3/log2(3) + 2/log2(4) = 1.8927893 + 1.0 = 2.8927893
    # Ideal top 3 of the labels is E, S, C with gains 3, 2, 1.
    #   IDCG@3 = 3/1 + 2/log2(3) + 1/log2(4) = 3 + 1.2618595 + 0.5 = 4.7618595
    #   NDCG@3 = 2.8927893 / 4.7618595 = 0.6074915
    assert ndcg_at_k(["c", "a", "b", "x"], LABELS, k=3, gains=G) == pytest.approx(
        0.6074915, abs=1e-6
    )


def test_ndcg_hand_computed_example_1_official_gains():
    # Same ranking c, a, b with official gains 0, 1.0, 0.1.
    #   DCG@3  = 0/1 + 1.0/log2(3) + 0.1/log2(4) = 0.6309298 + 0.05 = 0.6809298
    # Ideal top 3 is E, S, C with gains 1.0, 0.1, 0.01.
    #   IDCG@3 = 1.0/1 + 0.1/log2(3) + 0.01/log2(4) = 1 + 0.0630930 + 0.005 = 1.0680930
    #   NDCG@3 = 0.6809298 / 1.0680930 = 0.6375193
    assert ndcg_at_k(["c", "a", "b", "x"], LABELS, k=3) == pytest.approx(0.6375193, abs=1e-6)


def test_ndcg_hand_computed_example_2_unlabelled_product_is_irrelevant():
    # "x" is unlabelled, so gain 0 at rank 1; "a" (Exact, gain 3) is at rank 2.
    #   DCG@2  = 0/1 + 3/log2(3) = 1.8927893
    #   IDCG@2 = 3/1 = 3 (the query has only one labelled product)
    #   NDCG@2 = 1.8927893 / 3 = 0.6309298
    # With a single Exact product the gain scale cancels, so official gains agree.
    assert ndcg_at_k(["x", "a"], {"a": "E"}, k=2, gains=G) == pytest.approx(0.6309298, abs=1e-6)
    assert ndcg_at_k(["x", "a"], {"a": "E"}, k=2) == pytest.approx(0.6309298, abs=1e-6)


def test_official_gains_give_substitutes_little_credit():
    # Only a Substitute in the top slot, Exact at rank 2.
    #   graded:   DCG = 2 + 3/log2(3) = 3.8927893, IDCG = 3 + 2/log2(3) = 4.2618595 -> 0.9134
    #   official: DCG = 0.1 + 1/log2(3) = 0.7309298, IDCG = 1 + 0.1/log2(3) = 1.0630930 -> 0.6876
    labels = {"a": "E", "b": "S"}
    assert ndcg_at_k(["b", "a"], labels, k=2, gains=G) == pytest.approx(0.9134, abs=1e-4)
    assert ndcg_at_k(["b", "a"], labels, k=2) == pytest.approx(0.6876, abs=1e-4)


def test_ndcg_is_one_for_ideal_ranking():
    assert ndcg_at_k(["a", "b", "d", "c"], LABELS, k=10) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b", "d", "c"], LABELS, k=10, gains=G) == pytest.approx(1.0)


def test_ndcg_is_zero_when_query_has_no_relevant_product():
    assert ndcg_at_k(["a"], {"a": "I"}, k=10) == 0.0


def test_recall_counts_products_with_positive_gain():
    # Relevant are a (E), b (S) and d (C); c is Irrelevant. Both gain sets agree on this.
    assert recall_at_k(["c", "a"], LABELS, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(["c", "a", "b", "d"], LABELS, k=4) == 1.0
    assert recall_at_k(["a", "b", "d"], LABELS, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(["c", "a"], LABELS, k=2, gains=G) == pytest.approx(1 / 3)


def test_recall_is_zero_when_query_has_no_relevant_product():
    assert recall_at_k(["a"], {"a": "I"}, k=10) == 0.0


def test_mean():
    assert mean([0.5, 1.0]) == 0.75
    assert mean([]) == 0.0
    assert math.isclose(mean([1 / 3] * 3), 1 / 3)
