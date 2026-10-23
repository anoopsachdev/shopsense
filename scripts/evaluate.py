"""Evaluate named pipelines on the eval queries in two settings.

Full catalog: search all catalog products (~2,500 queries' worth) and report NDCG@10
and Recall@50.
Re-ranking only: rank each query's own labelled candidates (~20 products) and report
NDCG@10, which is the original ESCI Task 1 setup.
Both settings report p50/p95 latency per query (search only; indexing is excluded).

Run scripts/download_data.py first, and scripts/build_index.py for dense pipelines.

    python scripts/evaluate.py --pipelines all
    python scripts/evaluate.py --pipelines bm25,hybrid
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from shopsense import data
from shopsense.metrics import gain, mean, ndcg_at_k, recall_at_k
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


def evaluate_full_catalog(
    retriever: Retriever,
    queries: Mapping[int, str],
    labels: Mapping[int, Mapping[str, str]],
) -> dict[str, float]:
    """Mean NDCG@10 and Recall@50, plus p50/p95 latency, searching the whole index."""
    ndcgs, recalls, latencies = [], [], []
    for query_id, text in queries.items():
        start = time.perf_counter()
        hits = retriever.search(text, k=max(NDCG_K, RECALL_K))
        latencies.append(time.perf_counter() - start)
        ranked = [hit.product_id for hit in hits]
        ndcgs.append(ndcg_at_k(ranked, labels[query_id], NDCG_K))
        recalls.append(recall_at_k(ranked, labels[query_id], RECALL_K))
    p50, p95 = latency_percentiles(latencies)
    return {
        f"ndcg@{NDCG_K}": mean(ndcgs),
        f"recall@{RECALL_K}": mean(recalls),
        "p50_ms": p50,
        "p95_ms": p95,
    }


def evaluate_reranking(
    retriever: Retriever,
    queries: Mapping[int, str],
    labels: Mapping[int, Mapping[str, str]],
) -> dict[str, float]:
    """Mean NDCG@10, plus p50/p95 latency, when each query ranks only its own candidates."""
    ndcgs, latencies = [], []
    for query_id, text in queries.items():
        candidates = list(labels[query_id])
        start = time.perf_counter()
        hits = retriever.search(text, k=len(candidates), candidate_ids=candidates)
        latencies.append(time.perf_counter() - start)
        ndcgs.append(ndcg_at_k([hit.product_id for hit in hits], labels[query_id], NDCG_K))
    p50, p95 = latency_percentiles(latencies)
    return {f"ndcg@{NDCG_K}": mean(ndcgs), "p50_ms": p50, "p95_ms": p95}


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

    `results` maps pipeline name to {"full": ..., "rerank": ...} as returned by the
    evaluate_* functions.
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
    args = parser.parse_args()
    names = parse_pipeline_names(args.pipelines)

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

    results = {}
    for name in names:
        pipeline = get_pipeline(name)
        start = time.perf_counter()
        pipeline.index(catalog)
        print(f"{name}: indexed in {time.perf_counter() - start:.1f}s", flush=True)
        results[name] = {
            "full": evaluate_full_catalog(pipeline, queries, labels),
            "rerank": evaluate_reranking(pipeline, queries, labels),
        }
        print(f"{name}: done", flush=True)
    print()
    print(format_table(results))


if __name__ == "__main__":
    main()
