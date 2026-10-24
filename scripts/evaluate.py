"""Evaluate named pipelines on the eval queries in two settings.

Full catalog: search all catalog products (~2,500 queries' worth) and report NDCG@10
and Recall@50.
Re-ranking only: rank each query's own labelled candidates (~20 products) and report
NDCG@10, which is the original ESCI Task 1 setup.
Both settings report p50/p95 latency per query (search only; indexing is excluded).

NDCG is reported with the official ESCI gains (E 1.0, S 0.1, C 0.01, I 0; the headline)
and the graded gains (E 3, S 2, C 1, I 0). Each pipeline searches once; its rankings are
scored under each gain set.

Run scripts/download_data.py first, and scripts/build_index.py for dense pipelines.

    python scripts/evaluate.py --pipelines all
    python scripts/evaluate.py --pipelines bm25,hybrid --gains official
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shopsense import data
from shopsense.metrics import GAIN_SETS, gain, mean, ndcg_at_k, recall_at_k
from shopsense.pipeline import PIPELINES, get_pipeline
from shopsense.retrieval.base import Retriever

NDCG_K = 10
RECALL_K = 50


def latency_percentiles(seconds: Sequence[float]) -> tuple[float, float]:
    """(p50, p95) of per-query latencies, in milliseconds. (0, 0) for no queries."""
    if not seconds:
        return 0.0, 0.0
    p50, p95 = np.percentile(np.asarray(seconds) * 1000, [50, 95])
    return float(p50), float(p95)


@dataclass(frozen=True)
class Run:
    """One pipeline's ranked product IDs per query, and its search latency."""

    rankings: dict[int, list[str]]
    p50_ms: float
    p95_ms: float


def run_full_catalog(retriever: Retriever, queries: Mapping[int, str]) -> Run:
    """Search the whole index for each query (top max(NDCG_K, RECALL_K))."""
    rankings, latencies = {}, []
    for query_id, text in queries.items():
        start = time.perf_counter()
        hits = retriever.search(text, k=max(NDCG_K, RECALL_K))
        latencies.append(time.perf_counter() - start)
        rankings[query_id] = [hit.product_id for hit in hits]
    return Run(rankings, *latency_percentiles(latencies))


def run_reranking(
    retriever: Retriever,
    queries: Mapping[int, str],
    labels: Mapping[int, Mapping[str, str]],
) -> Run:
    """Rank only each query's own labelled candidates."""
    rankings, latencies = {}, []
    for query_id, text in queries.items():
        candidates = list(labels[query_id])
        start = time.perf_counter()
        hits = retriever.search(text, k=len(candidates), candidate_ids=candidates)
        latencies.append(time.perf_counter() - start)
        rankings[query_id] = [hit.product_id for hit in hits]
    return Run(rankings, *latency_percentiles(latencies))


def score_run(
    run: Run,
    labels: Mapping[int, Mapping[str, str]],
    gains: Mapping[str, float],
    with_recall: bool,
) -> dict[str, float]:
    """Mean NDCG@10 (and Recall@50 if `with_recall`) of a run, plus its latency."""
    items = run.rankings.items()
    scores = {f"ndcg@{NDCG_K}": mean([ndcg_at_k(r, labels[q], NDCG_K, gains) for q, r in items])}
    if with_recall:
        recalls = [recall_at_k(r, labels[q], RECALL_K, gains) for q, r in items]
        scores[f"recall@{RECALL_K}"] = mean(recalls)
    return {**scores, "p50_ms": run.p50_ms, "p95_ms": run.p95_ms}


def score_runs(
    runs: Mapping[str, Mapping[str, Run]],
    labels: Mapping[int, Mapping[str, str]],
    gains: Mapping[str, float],
) -> dict[str, dict[str, dict[str, float]]]:
    """Score every pipeline's {"full": Run, "rerank": Run} under one gain set."""
    return {
        name: {
            "full": score_run(pair["full"], labels, gains, with_recall=True),
            "rerank": score_run(pair["rerank"], labels, gains, with_recall=False),
        }
        for name, pair in runs.items()
    }


def parse_pipeline_names(arg: str) -> list[str]:
    """'all' or a comma-separated list of names from PIPELINES. Raises ValueError otherwise."""
    if arg == "all":
        return list(PIPELINES)
    names = [name.strip() for name in arg.split(",") if name.strip()]
    unknown = [name for name in names if name not in PIPELINES]
    if unknown or not names:
        raise ValueError(f"unknown pipelines {unknown}; choose from {list(PIPELINES)} or 'all'")
    return names


def format_table(results: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str:
    """One Markdown table: a row per pipeline, full-catalog then re-ranking-only columns.

    `results` maps pipeline name to {"full": ..., "rerank": ...} as returned by
    `score_runs`.
    """
    header = (
        f"| Pipeline | Full NDCG@{NDCG_K} | Full Recall@{RECALL_K} | Full p50 ms | Full p95 ms "
        f"| Rerank NDCG@{NDCG_K} | Rerank p50 ms | Rerank p95 ms |"
    )
    lines = [header, "|" + "---|" * 8]
    for name, result in results.items():
        full, rerank = result["full"], result["rerank"]
        lines.append(
            f"| {name} | {full[f'ndcg@{NDCG_K}']:.4f} | {full[f'recall@{RECALL_K}']:.4f} "
            f"| {full['p50_ms']:.1f} | {full['p95_ms']:.1f} "
            f"| {rerank[f'ndcg@{NDCG_K}']:.4f} | {rerank['p50_ms']:.1f} | {rerank['p95_ms']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=data.DATA_DIR)
    parser.add_argument(
        "--pipelines", default="all", help=f"'all' or comma-separated: {', '.join(PIPELINES)}"
    )
    parser.add_argument(
        "--gains",
        choices=list(GAIN_SETS),
        help="print only this gain set's table (default: all, official first)",
    )
    args = parser.parse_args()
    names = parse_pipeline_names(args.pipelines)
    gain_names = [args.gains] if args.gains else list(GAIN_SETS)

    con = data.connect(args.data_dir)
    sample = data.load_query_sample(args.data_dir / data.EVAL_QUERY_IDS_FILE)
    catalog = data.build_catalog(con, sample.catalog_query_ids)
    queries = data.load_queries(con, sample.eval_ids)
    labels = data.load_labels(con, sample.eval_ids)
    no_relevant = sum(
        1 for qid in sample.eval_ids if all(gain(label) == 0 for label in labels[qid].values())
    )
    print(f"eval queries: {len(queries)} ({no_relevant} with no relevant product)")
    print(f"catalog: {len(catalog):,} products from {len(sample.catalog_query_ids):,} queries")

    runs = {}
    for name in names:
        pipeline = get_pipeline(name)
        start = time.perf_counter()
        pipeline.index(catalog)
        print(f"{name}: indexed in {time.perf_counter() - start:.1f}s", flush=True)
        runs[name] = {
            "full": run_full_catalog(pipeline, queries),
            "rerank": run_reranking(pipeline, queries, labels),
        }
        print(f"{name}: done", flush=True)
    for gain_name in gain_names:
        gains = GAIN_SETS[gain_name]
        described = ", ".join(f"{label} {value:g}" for label, value in gains.items())
        headline = " (headline)" if gain_name == next(iter(GAIN_SETS)) else ""
        print(f"\nNDCG gains: {gain_name}{headline}: {described}\n")
        print(format_table(score_runs(runs, labels, gains)))


if __name__ == "__main__":
    main()
