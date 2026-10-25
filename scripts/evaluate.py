"""Evaluate named pipelines on the eval queries in two settings.

Full catalog: search all catalog products (~2,500 queries' worth) and report NDCG@10
and Recall@50.
Re-ranking only: rank each query's own labelled candidates (~20 products) and report
NDCG@10, which is the original ESCI Task 1 setup.
Both settings report p50/p95 latency per query (search only; indexing is excluded).

NDCG is reported with the official ESCI gains (E 1.0, S 0.1, C 0.01, I 0; the headline)
and the graded gains (E 3, S 2, C 1, I 0). Each pipeline searches once; its rankings are
scored under each gain set.

Pipelines that call an LLM run only on a fixed 200-query subset of the eval set
(data.llm_eval_ids), several queries at a time, and stop if the logged LLM spend
reaches the budget. For them the script also prints cost per 1K queries, fallback
rate, NDCG by query slice, and the queries that regressed most against
hybrid+cross_encoder_fast.

Run scripts/download_data.py first, and scripts/build_index.py for dense pipelines.

    python scripts/evaluate.py --pipelines all
    python scripts/evaluate.py --pipelines bm25,hybrid --gains official
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shopsense import data
from shopsense.llm.client import BudgetExceeded
from shopsense.llm.query_intent import mentions
from shopsense.metrics import GAIN_SETS, gain, mean, ndcg_at_k, recall_at_k
from shopsense.pipeline import PIPELINES, get_pipeline
from shopsense.retrieval.base import Hit, Retriever

NDCG_K = 10
RECALL_K = 50
BASELINE = "hybrid+cross_encoder_fast"
DEFAULT_LLM_WORKERS = 8
NUM_REGRESSIONS = 20

SHORT_MAX_WORDS = 2
LONG_MIN_WORDS = 6
NEGATION_WORDS = {"no", "not", "without", "non", "except", "excluding", "free"}
GENERIC_BRANDS = {"generic", "unbranded", "unknown", "n/a", "na", "none"}


def latency_percentiles(seconds: Sequence[float]) -> tuple[float, float]:
    """(p50, p95) of per-query latencies, in milliseconds. (0, 0) for no queries."""
    if not seconds:
        return 0.0, 0.0
    p50, p95 = np.percentile(np.asarray(seconds) * 1000, [50, 95])
    return float(p50), float(p95)


@dataclass(frozen=True)
class Run:
    """One pipeline's ranked product IDs and latency per query, plus its LLM usage."""

    rankings: dict[int, list[str]]
    latencies: dict[int, float]
    cost_usd: float = 0.0
    rerank_calls: int = 0
    fallbacks: int = 0
    fallback_queries: frozenset[str] = field(default_factory=frozenset)

    @property
    def p50_ms(self) -> float:
        return latency_percentiles(list(self.latencies.values()))[0]

    @property
    def p95_ms(self) -> float:
        return latency_percentiles(list(self.latencies.values()))[1]

    @property
    def cost_per_1k(self) -> float:
        """List-price LLM cost per 1,000 queries (cache hits included at full price)."""
        return 1000 * self.cost_usd / len(self.rankings) if self.rankings else 0.0

    @property
    def fallback_rate(self) -> float:
        """Share of LLM rerank calls that fell back to the cross-encoder."""
        return self.fallbacks / self.rerank_calls if self.rerank_calls else 0.0

    def subset(self, query_ids: Sequence[int]) -> Run:
        """The same run restricted to `query_ids`; cost is scaled by the share kept."""
        kept = [qid for qid in query_ids if qid in self.rankings]
        share = len(kept) / len(self.rankings) if self.rankings else 0.0
        return Run(
            rankings={qid: self.rankings[qid] for qid in kept},
            latencies={qid: self.latencies[qid] for qid in kept},
            cost_usd=self.cost_usd * share,
        )


def _run(
    retriever: Retriever,
    query_ids: Sequence[int],
    search: Callable[[int], list[Hit]],
    workers: int,
) -> Run:
    """Time `search` per query (in `workers` threads) and collect the pipeline's LLM stats."""
    if hasattr(retriever, "reset_llm_stats"):
        retriever.reset_llm_stats()

    def timed(query_id: int) -> tuple[int, list[str], float]:
        start = time.perf_counter()
        hits = search(query_id)
        return query_id, [hit.product_id for hit in hits], time.perf_counter() - start

    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(timed, query_ids))
    else:
        results = [timed(qid) for qid in query_ids]
    stats = retriever.llm_stats() if hasattr(retriever, "llm_stats") else {}
    reranker = getattr(retriever, "reranker", None)
    return Run(
        rankings={qid: ranked for qid, ranked, _ in results},
        latencies={qid: seconds for qid, _, seconds in results},
        cost_usd=stats.get("cost_usd", 0.0),
        rerank_calls=stats.get("rerank_calls", 0),
        fallbacks=stats.get("fallbacks", 0),
        fallback_queries=frozenset(getattr(reranker, "fallback_queries", ())),
    )


def run_full_catalog(
    retriever: Retriever, queries: Mapping[int, str], workers: int = 1
) -> Run:
    """Search the whole index for each query (top max(NDCG_K, RECALL_K))."""
    k = max(NDCG_K, RECALL_K)
    return _run(retriever, list(queries), lambda qid: retriever.search(queries[qid], k), workers)


def run_reranking(
    retriever: Retriever,
    queries: Mapping[int, str],
    labels: Mapping[int, Mapping[str, str]],
    workers: int = 1,
) -> Run:
    """Rank only each query's own labelled candidates."""

    def search(query_id: int) -> list[Hit]:
        candidates = list(labels[query_id])
        return retriever.search(queries[query_id], k=len(candidates), candidate_ids=candidates)

    return _run(retriever, list(queries), search, workers)


def per_query_ndcg(
    run: Run, labels: Mapping[int, Mapping[str, str]], gains: Mapping[str, float]
) -> dict[int, float]:
    """NDCG@10 of each query in the run."""
    return {q: ndcg_at_k(ranked, labels[q], NDCG_K, gains) for q, ranked in run.rankings.items()}


def score_run(
    run: Run,
    labels: Mapping[int, Mapping[str, str]],
    gains: Mapping[str, float],
    with_recall: bool,
) -> dict[str, float]:
    """Mean NDCG@10 (and Recall@50 if `with_recall`), latency, LLM cost and fallback rate."""
    scores = {f"ndcg@{NDCG_K}": mean(list(per_query_ndcg(run, labels, gains).values()))}
    if with_recall:
        recalls = [recall_at_k(r, labels[q], RECALL_K, gains) for q, r in run.rankings.items()]
        scores[f"recall@{RECALL_K}"] = mean(recalls)
    return {
        **scores,
        "p50_ms": run.p50_ms,
        "p95_ms": run.p95_ms,
        "cost_per_1k": run.cost_per_1k,
        "fallback_rate": run.fallback_rate,
    }


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


def format_table(
    results: Mapping[str, Mapping[str, Mapping[str, float]]], with_llm: bool = False
) -> str:
    """One Markdown table: a row per pipeline, full-catalog then re-ranking-only columns.

    `results` maps pipeline name to {"full": ..., "rerank": ...} as returned by
    `score_runs`. `with_llm` adds cost per 1K queries and fallback rate per setting.
    """
    columns = [
        "Pipeline",
        f"Full NDCG@{NDCG_K}",
        f"Full Recall@{RECALL_K}",
        "Full p50 ms",
        "Full p95 ms",
    ]
    columns += ["Full $/1K q", "Full fallback"] if with_llm else []
    columns += [f"Rerank NDCG@{NDCG_K}", "Rerank p50 ms", "Rerank p95 ms"]
    columns += ["Rerank $/1K q", "Rerank fallback"] if with_llm else []
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for name, result in results.items():
        full, rerank = result["full"], result["rerank"]
        cells = [
            name,
            f"{full[f'ndcg@{NDCG_K}']:.4f}",
            f"{full[f'recall@{RECALL_K}']:.4f}",
            f"{full['p50_ms']:.1f}",
            f"{full['p95_ms']:.1f}",
        ]
        if with_llm:
            cells += [f"{full['cost_per_1k']:.2f}", f"{full['fallback_rate']:.1%}"]
        cells += [
            f"{rerank[f'ndcg@{NDCG_K}']:.4f}",
            f"{rerank['p50_ms']:.1f}",
            f"{rerank['p95_ms']:.1f}",
        ]
        if with_llm:
            cells += [f"{rerank['cost_per_1k']:.2f}", f"{rerank['fallback_rate']:.1%}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def query_words(query: str) -> list[str]:
    """Lowercase words, keeping hyphenated words like "sugar-free" whole."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", query.lower())


def is_negation(query: str) -> bool:
    """True if the query says what it doesn't want: no/not/without/..., or X-free/non-X."""
    return any(
        word in NEGATION_WORDS or word.endswith("-free") or word.startswith("non-")
        for word in query_words(query)
    )


def query_slices(
    queries: Mapping[int, str],
    labels: Mapping[int, Mapping[str, str]],
    brands: Mapping[str, str],
) -> dict[str, list[int]]:
    """Query IDs per slice.

    short: at most 2 words. long: at least 6 words. brand: the query names the brand of
    one of its Exact products (`brands` maps product ID to brand). negation: see
    `is_negation`. Slices overlap; "all" holds every query.
    """

    def names_exact_brand(qid: int) -> bool:
        exact_brands = {
            brands.get(pid, "").strip() for pid, label in labels[qid].items() if label == "E"
        }
        return any(
            len(b) >= 2 and b.lower() not in GENERIC_BRANDS and mentions(b, queries[qid])
            for b in exact_brands
        )

    return {
        "all": list(queries),
        "short": [q for q, text in queries.items() if len(query_words(text)) <= SHORT_MAX_WORDS],
        "long": [q for q, text in queries.items() if len(query_words(text)) >= LONG_MIN_WORDS],
        "brand": [q for q in queries if names_exact_brand(q)],
        "negation": [q for q, text in queries.items() if is_negation(text)],
    }


def format_slices(
    runs: Mapping[str, Run],
    labels: Mapping[int, Mapping[str, str]],
    gains: Mapping[str, float],
    slices: Mapping[str, Sequence[int]],
) -> str:
    """Markdown table of mean NDCG@10 per pipeline (rows) and slice (columns)."""
    titles = [f"{name} (n={len(ids)})" for name, ids in slices.items()]
    header = "| Pipeline | " + " | ".join(titles) + " |"
    lines = [header, "|" + "---|" * (len(slices) + 1)]
    for name, run in runs.items():
        per_query = per_query_ndcg(run, labels, gains)
        cells = [
            f"{mean([per_query[q] for q in ids if q in per_query]):.4f}" if ids else "-"
            for ids in slices.values()
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def regressions(
    baseline: Run,
    run: Run,
    queries: Mapping[int, str],
    labels: Mapping[int, Mapping[str, str]],
    gains: Mapping[str, float],
    n: int = NUM_REGRESSIONS,
) -> list[dict]:
    """The `n` queries where `run` lost the most NDCG@10 against `baseline`, worst first.

    Only queries that got worse are listed.
    """
    before = per_query_ndcg(baseline, labels, gains)
    after = per_query_ndcg(run, labels, gains)
    rows = [
        {
            "query_id": qid,
            "query": queries[qid],
            "baseline": before[qid],
            "ndcg": after[qid],
            "delta": after[qid] - before[qid],
            "fell_back": queries[qid] in run.fallback_queries,
        }
        for qid in after
        if qid in before and after[qid] < before[qid]
    ]
    return sorted(rows, key=lambda row: (row["delta"], row["query_id"]))[:n]


def format_regressions(rows: Sequence[dict], name: str) -> str:
    """Markdown table of `regressions` output."""
    lines = [
        f"| Query ID | Query | {BASELINE} | {name} | Δ NDCG@{NDCG_K} | Fell back |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        query = row["query"].replace("|", "\\|")
        lines.append(
            f"| {row['query_id']} | {query} | {row['baseline']:.3f} | {row['ndcg']:.3f} "
            f"| {row['delta']:+.3f} | {'yes' if row['fell_back'] else ''} |"
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
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=DEFAULT_LLM_WORKERS,
        help="queries in flight at once for LLM pipelines",
    )
    args = parser.parse_args()
    names = parse_pipeline_names(args.pipelines)
    gain_names = [args.gains] if args.gains else list(GAIN_SETS)
    headline = GAIN_SETS[next(iter(GAIN_SETS))]

    con = data.connect(args.data_dir)
    sample = data.load_query_sample(args.data_dir / data.EVAL_QUERY_IDS_FILE)
    catalog = data.build_catalog(con, sample.catalog_query_ids)
    queries = data.load_queries(con, sample.eval_ids)
    labels = data.load_labels(con, sample.eval_ids)
    llm_ids = data.llm_eval_ids(sample)
    llm_queries = {qid: queries[qid] for qid in llm_ids}
    no_relevant = sum(
        1 for qid in sample.eval_ids if all(gain(label) == 0 for label in labels[qid].values())
    )
    print(f"eval queries: {len(queries)} ({no_relevant} with no relevant product)")
    print(f"LLM subset: {len(llm_ids)} queries")
    print(f"catalog: {len(catalog):,} products from {len(sample.catalog_query_ids):,} queries")

    runs: dict[str, dict[str, Run]] = {}
    llm_names: list[str] = []
    llm_client = None
    budget_hit = False
    started_at = time.time()
    for name in names:
        pipeline = get_pipeline(name)
        uses_llm = getattr(pipeline, "uses_llm", False)
        if uses_llm and budget_hit:
            print(f"{name}: skipped, LLM budget reached", flush=True)
            continue
        start = time.perf_counter()
        pipeline.index(catalog)
        print(f"{name}: indexed in {time.perf_counter() - start:.1f}s", flush=True)
        if uses_llm:
            llm_client = getattr(pipeline.reranker, "client", None) or llm_client
            try:
                runs[name] = {
                    "full": run_full_catalog(pipeline, llm_queries, args.llm_workers),
                    "rerank": run_reranking(pipeline, llm_queries, labels, args.llm_workers),
                }
            except BudgetExceeded as err:
                print(f"{name}: STOPPED, {err}", flush=True)
                budget_hit = True
                continue
            llm_names.append(name)
        else:
            runs[name] = {
                "full": run_full_catalog(pipeline, queries),
                "rerank": run_reranking(pipeline, queries, labels),
            }
        print(f"{name}: done", flush=True)
    plain_names = [name for name in runs if name not in llm_names]

    for gain_name in gain_names:
        gains = GAIN_SETS[gain_name]
        described = ", ".join(f"{label} {value:g}" for label, value in gains.items())
        tag = " (headline)" if gains is headline else ""
        if plain_names:
            print(
                f"\n## All {len(queries)} eval queries. NDCG gains: {gain_name}{tag}: "
                f"{described}\n"
            )
            print(format_table(score_runs({n: runs[n] for n in plain_names}, labels, gains)))
        if llm_names:
            subset_runs = {
                n: {setting: run.subset(llm_ids) for setting, run in runs[n].items()}
                for n in plain_names
            }
            subset_runs.update({n: runs[n] for n in llm_names})
            print(
                f"\n## LLM subset, {len(llm_ids)} queries. NDCG gains: {gain_name}{tag}: "
                f"{described}\n"
            )
            print(format_table(score_runs(subset_runs, labels, gains), with_llm=True))

    if not llm_names:
        return
    full_runs = {n: runs[n]["full"].subset(llm_ids) for n in plain_names}
    full_runs.update({n: runs[n]["full"] for n in llm_names})
    brands = dict(zip(catalog["product_id"], catalog["product_brand"].fillna("")))
    slices = query_slices(llm_queries, labels, brands)
    print("\n## NDCG@10 by query slice (LLM subset, full catalog, official gains)\n")
    print(format_slices(full_runs, labels, headline, slices))
    if BASELINE in full_runs:
        for name in llm_names:
            rows = regressions(full_runs[BASELINE], full_runs[name], llm_queries, labels, headline)
            print(f"\n## {len(rows)} biggest regressions of {name} vs {BASELINE} (full catalog)\n")
            print(format_regressions(rows, name))
    else:
        print(f"\n(regressions skipped: run {BASELINE} too)")
    if llm_client is not None:
        usage = llm_client.usage_since(started_at)
        print(
            f"\nLLM calls this run: {usage['calls']} ({usage['cached']} cache hits); "
            f"list-price cost ${usage['cost_usd']:.4f}, billed ${usage['billed_usd']:.4f}; "
            f"total logged spend ${llm_client.spent_usd():.4f} of ${llm_client.budget_usd:.2f}"
        )


if __name__ == "__main__":
    main()
