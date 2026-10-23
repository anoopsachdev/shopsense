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
    assert 0 <= result["p50_ms"] <= result["p95_ms"]


def test_evaluate_reranking_scores_only_labelled_candidates(bm25):
    # Query 2's candidates are p3 (E) and p1 (I); BM25 puts p3 first, so NDCG is 1.
    result = evaluate.evaluate_reranking(bm25, QUERIES, LABELS)
    assert result["ndcg@10"] == pytest.approx(1.0)
    assert 0 <= result["p50_ms"] <= result["p95_ms"]


def test_evaluate_reranking_penalises_wrong_order(bm25):
    # "blue" ranks p2 (I) above p1 (E): DCG = 0 + 3/log2(3), IDCG = 3, NDCG = 0.6309298.
    result = evaluate.evaluate_reranking(bm25, {1: "blue"}, {1: {"p1": "E", "p2": "I"}})
    assert result["ndcg@10"] == pytest.approx(0.6309298, abs=1e-6)


def test_latency_percentiles_in_milliseconds():
    # numpy linear interpolation over 0.001..0.020 s: p50 = 10.5 ms, p95 = 19.05 ms.
    seconds = [i / 1000 for i in range(1, 21)]
    p50, p95 = evaluate.latency_percentiles(seconds)
    assert p50 == pytest.approx(10.5)
    assert p95 == pytest.approx(19.05)
    assert evaluate.latency_percentiles([]) == (0.0, 0.0)


def test_parse_pipeline_names():
    assert evaluate.parse_pipeline_names("all") == [
        "bm25",
        "dense",
        "hybrid",
        "hybrid+cross_encoder",
    ]
    assert evaluate.parse_pipeline_names("bm25, hybrid") == ["bm25", "hybrid"]
    with pytest.raises(ValueError, match="unknown pipelines"):
        evaluate.parse_pipeline_names("bm25,nope")
    with pytest.raises(ValueError):
        evaluate.parse_pipeline_names("")


def test_format_table_has_a_row_per_pipeline():
    full = {"ndcg@10": 0.5, "recall@50": 0.6, "p50_ms": 1.0, "p95_ms": 2.0}
    rerank = {"ndcg@10": 0.8, "p50_ms": 0.5, "p95_ms": 0.75}
    table = evaluate.format_table({"bm25": {"full": full, "rerank": rerank}}).splitlines()
    assert table[0].startswith("| Pipeline | Full NDCG@10 | Full Recall@50 |")
    assert table[2] == "| bm25 | 0.5000 | 0.6000 | 1.0 | 2.0 | 0.8000 | 0.5 | 0.8 |"
