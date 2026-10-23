"""Evaluate BM25 on the eval queries in two settings.

Full catalog: search all catalog products (~2,500 queries' worth) and report NDCG@10
and Recall@50.
Re-ranking only: rank each query's own labelled candidates (~20 products) and report
NDCG@10, which is the original ESCI Task 1 setup.

Run scripts/download_data.py first.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path

from shopsense import data
from shopsense.metrics import gain, mean, ndcg_at_k, recall_at_k
from shopsense.retrieval.base import Retriever
from shopsense.retrieval.bm25 import BM25Retriever

NDCG_K = 10
RECALL_K = 50


def evaluate_full_catalog(
    retriever: Retriever,
    queries: Mapping[int, str],
    labels: Mapping[int, Mapping[str, str]],
) -> dict[str, float]:
    """Mean NDCG@10, Recall@50 and search latency when searching the whole index."""
    ndcgs, recalls, latencies = [], [], []
    for query_id, text in queries.items():
        start = time.perf_counter()
        hits = retriever.search(text, k=max(NDCG_K, RECALL_K))
        latencies.append(time.perf_counter() - start)
        ranked = [hit.product_id for hit in hits]
        ndcgs.append(ndcg_at_k(ranked, labels[query_id], NDCG_K))
        recalls.append(recall_at_k(ranked, labels[query_id], RECALL_K))
    return {
        f"ndcg@{NDCG_K}": mean(ndcgs),
        f"recall@{RECALL_K}": mean(recalls),
        "mean_latency_ms": 1000 * mean(latencies),
    }


def evaluate_reranking(
    retriever: Retriever,
    queries: Mapping[int, str],
    labels: Mapping[int, Mapping[str, str]],
) -> dict[str, float]:
    """Mean NDCG@10 when each query ranks only its own labelled candidates."""
    ndcgs = []
    for query_id, text in queries.items():
        candidates = list(labels[query_id])
        hits = retriever.search(text, k=len(candidates), candidate_ids=candidates)
        ndcgs.append(ndcg_at_k([hit.product_id for hit in hits], labels[query_id], NDCG_K))
    return {f"ndcg@{NDCG_K}": mean(ndcgs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data.DATA_DIR)
    args = parser.parse_args()

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

    start = time.perf_counter()
    bm25 = BM25Retriever()
    bm25.index(catalog)
    print(f"BM25 index built in {time.perf_counter() - start:.1f}s")

    full = evaluate_full_catalog(bm25, queries, labels)
    print(
        f"BM25 full catalog:    NDCG@{NDCG_K} = {full[f'ndcg@{NDCG_K}']:.4f}   "
        f"Recall@{RECALL_K} = {full[f'recall@{RECALL_K}']:.4f}   "
        f"latency = {full['mean_latency_ms']:.1f} ms/query"
    )
    rerank = evaluate_reranking(bm25, queries, labels)
    print(f"BM25 re-ranking only: NDCG@{NDCG_K} = {rerank[f'ndcg@{NDCG_K}']:.4f}")


if __name__ == "__main__":
    main()
