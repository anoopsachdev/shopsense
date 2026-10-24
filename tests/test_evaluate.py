import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from shopsense.metrics import ESCI_GRADED_GAINS, ESCI_OFFICIAL_GAINS
from shopsense.retrieval.base import Hit
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


def run_of(rankings, seconds=0.0, **kwargs):
    """A Run with the same latency for every query."""
    return evaluate.Run(rankings, {qid: seconds for qid in rankings}, **kwargs)


def test_score_run_perfect_rankings():
    run = run_of({1: ["p1", "p2"], 2: ["p3", "p1"]}, seconds=0.002)
    scores = evaluate.score_run(run, LABELS, ESCI_OFFICIAL_GAINS, with_recall=True)
    assert scores == {
        "ndcg@10": pytest.approx(1.0),
        "recall@50": pytest.approx(1.0),
        "p50_ms": pytest.approx(2.0),
        "p95_ms": pytest.approx(2.0),
        "cost_per_1k": 0.0,
        "fallback_rate": 0.0,
    }
    assert "recall@50" not in evaluate.score_run(run, LABELS, ESCI_OFFICIAL_GAINS, False)


def test_score_run_depends_on_the_gain_set():
    # Substitute p2 above Exact p1 (see test_metrics for the hand computation).
    run = run_of({1: ["p2", "p1"]})
    labels = {1: {"p1": "E", "p2": "S"}}
    official = evaluate.score_run(run, labels, ESCI_OFFICIAL_GAINS, with_recall=False)
    graded = evaluate.score_run(run, labels, ESCI_GRADED_GAINS, with_recall=False)
    assert official["ndcg@10"] == pytest.approx(0.6876, abs=1e-4)
    assert graded["ndcg@10"] == pytest.approx(0.9134, abs=1e-4)


def test_score_runs_scores_both_settings_per_pipeline():
    full = run_of({1: ["p1", "p2"]}, seconds=0.005)
    rerank = run_of({1: ["p2", "p1"]}, seconds=0.001)
    labels = {1: {"p1": "E", "p2": "I"}}
    runs = {"bm25": {"full": full, "rerank": rerank}}
    result = evaluate.score_runs(runs, labels, ESCI_OFFICIAL_GAINS)
    # Rerank puts the Irrelevant p2 first: DCG = 1/log2(3), IDCG = 1, NDCG = 0.6309298.
    assert result["bm25"]["full"]["ndcg@10"] == pytest.approx(1.0)
    assert result["bm25"]["full"]["p50_ms"] == pytest.approx(5.0)
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
        "hybrid+llm",
        "intent+hybrid+llm",
    ]
    assert evaluate.parse_pipeline_names("bm25, hybrid") == ["bm25", "hybrid"]
    with pytest.raises(ValueError, match="unknown pipelines"):
        evaluate.parse_pipeline_names("bm25,nope")
    with pytest.raises(ValueError):
        evaluate.parse_pipeline_names("")


FULL_SCORES = {
    "ndcg@10": 0.5,
    "recall@50": 0.6,
    "p50_ms": 1.0,
    "p95_ms": 2.0,
    "cost_per_1k": 1.234,
    "fallback_rate": 0.05,
}
RERANK_SCORES = {
    "ndcg@10": 0.8, "p50_ms": 0.5, "p95_ms": 0.75, "cost_per_1k": 0.5, "fallback_rate": 0.0,
}


def test_format_table_has_a_row_per_pipeline():
    results = {"bm25": {"full": FULL_SCORES, "rerank": RERANK_SCORES}}
    table = evaluate.format_table(results).splitlines()
    assert table[0].startswith("| Pipeline | Full NDCG@10 | Full Recall@50 |")
    assert table[2] == "| bm25 | 0.5000 | 0.6000 | 1.0 | 2.0 | 0.8000 | 0.5 | 0.8 |"


def test_format_table_with_llm_columns():
    results = {"x": {"full": FULL_SCORES, "rerank": RERANK_SCORES}}
    table = evaluate.format_table(results, with_llm=True).splitlines()
    assert "Full $/1K q | Full fallback" in table[0]
    assert table[2] == (
        "| x | 0.5000 | 0.6000 | 1.0 | 2.0 | 1.23 | 5.0% | 0.8000 | 0.5 | 0.8 | 0.50 | 0.0% |"
    )


def test_run_cost_per_1k_fallback_rate_and_subset():
    rankings = {1: ["a"], 2: ["b"], 3: ["c"], 4: ["d"]}
    run = run_of(rankings, cost_usd=0.02, rerank_calls=4, fallbacks=1)
    assert run.cost_per_1k == pytest.approx(5.0)
    assert run.fallback_rate == 0.25
    sub = run.subset([2, 4, 99])
    assert sub.rankings == {2: ["b"], 4: ["d"]}
    assert sub.cost_usd == pytest.approx(0.01)
    assert run_of({}).cost_per_1k == 0.0 and run_of({}).fallback_rate == 0.0


class StatsRetriever:
    """Retriever that reports LLM stats like a Pipeline does."""

    def __init__(self):
        self.resets = 0
        self.reranker = type("R", (), {"fallback_queries": ["coffee"]})()

    def reset_llm_stats(self):
        self.resets += 1

    def llm_stats(self):
        return {"calls": 3, "cost_usd": 0.003, "rerank_calls": 2, "fallbacks": 1}

    def search(self, query, k, candidate_ids=None):
        return [Hit("p1", 1.0)]


def test_runs_collect_llm_stats_and_fallback_queries():
    retriever = StatsRetriever()
    run = evaluate.run_full_catalog(retriever, QUERIES)
    assert retriever.resets == 1
    assert (run.cost_usd, run.rerank_calls, run.fallbacks) == (0.003, 2, 1)
    assert run.fallback_queries == frozenset({"coffee"})


def test_workers_give_the_same_rankings(bm25):
    assert (
        evaluate.run_reranking(bm25, QUERIES, LABELS, workers=4).rankings
        == evaluate.run_reranking(bm25, QUERIES, LABELS).rankings
    )


def test_query_words_and_negation():
    assert evaluate.query_words("Sugar-Free gum, 12 pack") == ["sugar-free", "gum", "12", "pack"]
    negated = [
        "dress without sleeves",
        "no sugar candy",
        "sugar-free gum",
        "non-slip mat",
        "gluten free bread",
        "case not otterbox",
    ]
    assert all(evaluate.is_negation(q) for q in negated)
    assert not any(evaluate.is_negation(q) for q in ["nothing bundt cake", "nonstick pan"])


def test_query_slices():
    queries = {1: "nike shoes", 2: "red dress without sleeves for women", 3: "acme coffee mug"}
    labels = {1: {"p1": "E"}, 2: {"p2": "E"}, 3: {"p3": "S", "p4": "E"}}
    brands = {"p1": "Nike", "p2": "Generic", "p3": "Acme", "p4": "Other"}
    slices = evaluate.query_slices(queries, labels, brands)
    assert slices == {
        "all": [1, 2, 3],
        "short": [1],
        "long": [2],
        "brand": [1],  # query 3 names Acme, but its Exact product is by Other
        "negation": [2],
    }


def test_format_slices():
    runs = {"x": run_of({1: ["p1"], 2: ["p9"]})}
    labels = {1: {"p1": "E"}, 2: {"p2": "E"}}
    slices = {"all": [1, 2], "a": [1], "none": []}
    table = evaluate.format_slices(runs, labels, ESCI_OFFICIAL_GAINS, slices).splitlines()
    assert table[0] == "| Pipeline | all (n=2) | a (n=1) | none (n=0) |"
    assert table[2] == "| x | 0.5000 | 1.0000 | - |"


def test_regressions_lists_only_worse_queries_worst_first():
    labels = {1: {"p1": "E"}, 2: {"p2": "E"}, 3: {"p3": "E"}}
    queries = {1: "a", 2: "b", 3: "c"}
    baseline = run_of({1: ["p1"], 2: ["p2"], 3: ["x", "p3"]})
    misses = [f"x{i}" for i in range(10)]
    run = run_of({1: ["x", "p1"], 2: misses, 3: ["p3"]}, fallback_queries=frozenset({"a"}))
    rows = evaluate.regressions(baseline, run, queries, labels, ESCI_OFFICIAL_GAINS, n=5)
    assert [row["query_id"] for row in rows] == [2, 1]
    assert rows[0]["delta"] == pytest.approx(-1.0)
    assert rows[1]["delta"] == pytest.approx(0.6309298 - 1.0, abs=1e-6)
    assert [row["fell_back"] for row in rows] == [False, True]
    table = evaluate.format_regressions(rows, "llm").splitlines()
    assert table[2] == "| 2 | b | 1.000 | 0.000 | -1.000 |  |"
    assert table[3].endswith("| yes |")
