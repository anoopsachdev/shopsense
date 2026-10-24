# ShopSense

LLM-assisted product search on Amazon's [ESCI shopping-queries dataset](https://github.com/amazon-science/esci-data),
scored against human relevance labels on quality, latency and cost per query.

Status: BM25, dense (bge-small), hybrid (RRF), cross-encoder and Gemini (query intent and listwise reranking)
pipelines, all scored by one harness. The API is not built yet.

## Results

500 held-out test queries (US, Task 1 small version, seed 42), official ESCI gains (see below). LLM pipelines
are scored on a fixed 200-query subset, so they are compared there with the fast cross-encoder on the same queries.

**Full catalog** (the main setting): each query searches all 48,862 catalog products, which are the products
labelled for its own query plus 2,499 others. This is the setting with room to improve.

| Pipeline | NDCG@10 | Recall@50 | p50 latency | $ / 1K queries |
|---|---|---|---|---|
| BM25 | 0.468 | 0.603 | 64 ms | – |
| Dense (bge-small) | 0.506 | 0.663 | 43 ms | – |
| Hybrid (RRF of BM25 + dense) | 0.520 | 0.687 | 103 ms | – |
| Hybrid + cross-encoder (top 50) | 0.574 | 0.687 | 1,156 ms | – |
| Hybrid + cross-encoder (top 20, "fast") | 0.564 | 0.687 | 632 ms | – |
| *200-query subset:* hybrid + cross-encoder fast | 0.565 | 0.680 | 625 ms | – |
| *200-query subset:* hybrid + Gemini rerank (top 20) | 0.600 | 0.680 | ~53 s | 1.14 |
| *200-query subset:* intent + hybrid + Gemini rerank | **0.607** | 0.663 | ~53 s* | 1.42 |

\* Its measured p50 (3.1 s) mostly reflects cache hits shared with the plain Gemini pipeline; uncached calls
take as long as the plain pipeline's.

The Gemini reranker is the most accurate, but at about 50 s per call and 24–28% fallbacks to the cross-encoder
(mostly HTTP 500s), it's not usable for live search as configured. See `NOTES.md` for the slices, regressions
and the three findings.

**Re-ranking only** (the original ESCI Task 1 setup): each query ranks only its own ~20 labelled candidates.
Most candidates are Exact matches, so there is less headroom.

| Pipeline | NDCG@10 |
|---|---|
| BM25 | 0.706 |
| Dense | 0.734 |
| Hybrid | 0.729 |
| Hybrid + cross-encoder (top 50) | 0.744 |
| Hybrid + cross-encoder fast | 0.745 |
| *200-query subset:* hybrid + cross-encoder fast | 0.750 |
| *200-query subset:* hybrid + Gemini rerank | 0.801 |
| *200-query subset:* intent + hybrid + Gemini rerank | 0.801 |

### How scores are computed

- **NDCG gains (headline: official).** The headline numbers use the **official ESCI gains**: Exact 1.0,
  Substitute 0.1, Complement 0.01, Irrelevant 0. These come from the esci-data repo's own Task 1 ranking scripts
  (`ranking/prepare_trec_eval_files.py`, `ranking/train.py`); the repo README itself states no gains. They put scores
  on the same gain scale as published ESCI results (the query set and catalog still differ), and they give
  substitutes almost no credit, so a ranking only scores well if it puts Exact matches first.
  We also report **graded gains** (E 3, S 2, C 1, I 0, the original CLAUDE.md fallback), which give substitutes
  two-thirds credit and so score higher. `python scripts/evaluate.py` prints both tables, official first, and
  `--gains official|graded` picks one. Every pipeline is scored with the same gains (`ESCI_OFFICIAL_GAINS` and
  `ESCI_GRADED_GAINS` in `src/shopsense/metrics.py`). Unlabelled products count as 0 under both.
- **Recall@50:** the share of a query's Exact, Substitute and Complement products found in the top 50. It's the same
  under both gain sets, since every gain except Irrelevant's is positive.
- **Latency:** median search time per query on a laptop CPU, indexing excluded. Non-LLM pipelines run one query
  at a time; LLM pipelines run 8 queries at a time, and their time is mostly waiting on Vertex AI. p95 and
  re-ranking-only latencies are in `NOTES.md`.
- **Cost:** list price per 1,000 queries from the `.env` prices, counting cache hits as paid.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py   # US-locale, Task 1 (small_version) parquet into data/
python scripts/build_index.py     # embed the catalog once (about 2 h on a laptop CPU)
cp .env.example .env              # only for the LLM pipelines: fill in project, models and prices
```

LLM pipelines call Gemini on Vertex AI with Application Default Credentials (`gcloud auth application-default login`).

## Commands

- Tests: `pytest -q`
- Evaluate: `python scripts/evaluate.py --pipelines all` (or e.g. `--pipelines bm25,hybrid`)
