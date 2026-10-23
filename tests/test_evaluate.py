import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from shopsense.retrieval.bm25 import BM25Retriever

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

CATALOG = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3"],
        "product_title": ["red shoes", "blue shoes", "coffee mug"],
    }
)
QUERIES = {1: "red shoes", 2: "coffee"}
LABELS = {1: {"p1": "E", "p2": "S"}, 2: {"p3": "E", "p1": "I"}}


@pytest.fixture
def bm25():
    retriever = BM25Retriever()
    retriever.index(CATALOG)
    return retriever


def test_evaluate_full_catalog_perfect_rankings(bm25):
    result = evaluate.evaluate_full_catalog(bm25, QUERIES, LABELS)
    assert result["ndcg@10"] == pytest.approx(1.0)
    assert result["recall@50"] == pytest.approx(1.0)
    assert result["mean_latency_ms"] >= 0


def test_evaluate_reranking_scores_only_labelled_candidates(bm25):
    # Query 2's candidates are p3 (E) and p1 (I); BM25 puts p3 first, so NDCG is 1.
    assert evaluate.evaluate_reranking(bm25, QUERIES, LABELS) == {"ndcg@10": pytest.approx(1.0)}


def test_evaluate_reranking_penalises_wrong_order(bm25):
    # "blue" ranks p2 (I) above p1 (E): DCG = 0 + 3/log2(3), IDCG = 3, NDCG = 0.6309298.
    result = evaluate.evaluate_reranking(bm25, {1: "blue"}, {1: {"p1": "E", "p2": "I"}})
    assert result["ndcg@10"] == pytest.approx(0.6309298, abs=1e-6)
