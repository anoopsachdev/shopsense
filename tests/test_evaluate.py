import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from shopsense.metrics import ESCI_GRADED_GAINS, ESCI_OFFICIAL_GAINS
from shopsense.retrieval.bm25 import BM25Retriever

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
sys.modules["evaluate"] = evaluate  # @dataclass looks its module up in sys.modules
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


def test_run_full_catalog_records_rankings_and_latency(bm25):
    run = evaluate.run_full_catalog(bm25, QUERIES)
    assert run.rankings[1][:2] == ["p1", "p2"]
    assert run.rankings[2][0] == "p3"
    assert 0 <= run.p50_ms <= run.p95_ms


def test_run_reranking_ranks_only_labelled_candidates(bm25):
    run = evaluate.run_reranking(bm25, QUERIES, LABELS)
    assert run.rankings == {1: ["p1", "p2"], 2: ["p3", "p1"]}


def test_score_run_perfect_rankings():
    run = evaluate.Run({1: ["p1", "p2"], 2: ["p3", "p1"]}, p50_ms=1.0, p95_ms=2.0)
    scores = evaluate.score_run(run, LABELS, ESCI_OFFICIAL_GAINS, with_recall=True)
    assert scores == {
        "ndcg@10": pytest.approx(1.0),
        "recall@50": pytest.approx(1.0),
        "p50_ms": 1.0,
        "p95_ms": 2.0,
    }
    assert "recall@50" not in evaluate.score_run(run, LABELS, ESCI_OFFICIAL_GAINS, False)


def test_score_run_depends_on_the_gain_set():
    # Substitute p2 above Exact p1 (see test_metrics for the hand computation).
    run = evaluate.Run({1: ["p2", "p1"]}, p50_ms=0.0, p95_ms=0.0)
    labels = {1: {"p1": "E", "p2": "S"}}
    official = evaluate.score_run(run, labels, ESCI_OFFICIAL_GAINS, with_recall=False)
    graded = evaluate.score_run(run, labels, ESCI_GRADED_GAINS, with_recall=False)
    assert official["ndcg@10"] == pytest.approx(0.6876, abs=1e-4)
    assert graded["ndcg@10"] == pytest.approx(0.9134, abs=1e-4)


def test_score_runs_scores_both_settings_per_pipeline():
    full = evaluate.Run({1: ["p1", "p2"]}, p50_ms=5.0, p95_ms=6.0)
    rerank = evaluate.Run({1: ["p2", "p1"]}, p50_ms=1.0, p95_ms=1.5)
    labels = {1: {"p1": "E", "p2": "I"}}
    runs = {"bm25": {"full": full, "rerank": rerank}}
    result = evaluate.score_runs(runs, labels, ESCI_OFFICIAL_GAINS)
    # Rerank puts the Irrelevant p2 first: DCG = 1/log2(3), IDCG = 1, NDCG = 0.6309298.
    assert result["bm25"]["full"]["ndcg@10"] == pytest.approx(1.0)
    assert result["bm25"]["full"]["p50_ms"] == 5.0
    assert result["bm25"]["rerank"]["ndcg@10"] == pytest.approx(0.6309298, abs=1e-6)


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
        "hybrid+cross_encoder_fast",
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
