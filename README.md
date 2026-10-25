# ShopSense

LLM-assisted product search on Amazon's [ESCI shopping-queries dataset](https://github.com/amazon-science/esci-data),
scored against human relevance labels on quality, latency and cost per query.

Status: BM25, dense (bge-small), hybrid (RRF), cross-encoder and Gemini (query intent and listwise reranking)
pipelines, all scored by one harness, and served by a FastAPI app with a one-page search UI.

## Live demo

**https://shopsense-819934511305.us-central1.run.app**: the search page, on Cloud Run (us-central1, 2 vCPU,
2 GiB, scales to zero).

The page loads in about a second even on a cold instance. The models load in the background, and only the first
search after a cold start waits, up to about 1.5 minutes, with a "warming up" note. A Cloud Scheduler job calls
`/health?wait=true` every 10 minutes, so an instance is usually already warm.

- API: [`/search?q=red+shoes`](https://shopsense-819934511305.us-central1.run.app/search?q=red+shoes),
  [`/pipelines`](https://shopsense-819934511305.us-central1.run.app/pipelines),
  [`/health`](https://shopsense-819934511305.us-central1.run.app/health)
- The 5 non-LLM pipelines serve publicly; the default is `hybrid+cross_encoder_fast`.

| Pipeline (live) | NDCG@10, full catalog* | Recall@50* | Live server latency, median of 5** |
|---|---|---|---|
| bm25 | 0.468 | 0.603 | 121 ms |
| dense | 0.506 | 0.663 | 218 ms |
| hybrid | 0.520 | 0.687 | 310 ms |
| hybrid+cross_encoder | 0.574 | 0.687 | 4,374 ms |
| hybrid+cross_encoder_fast (default) | 0.564 | 0.687 | 1,691 ms |

\* Official ESCI gains, 500 held-out queries, from the offline evaluation in `NOTES.md`.
\*\* `latency_ms` returned by the live service for 5 queries on a warm instance (2026-10-25), excluding network
time. Cloud Run's 2 vCPUs run the cross-encoder 3–4× slower than the laptop the evaluation ran on.

**The LLM pipelines are offline-only.** `hybrid+llm` and `intent+hybrid+llm` score best (0.600 and 0.607
NDCG@10 against 0.565 for the fast cross-encoder on the same 200 queries). But the Gemini reranker took about 50 s per
call, 24–28% of calls fell back to the cross-encoder (mostly Vertex HTTP 500s), and each costs about $1.1–1.4
per 1,000 queries. The spend cap is a per-container log, so it doesn't protect a public endpoint. The service
therefore has no Vertex AI settings: `/pipelines` lists both as unavailable, and calling them returns 503. They
run locally with `.env` and `scripts/evaluate.py`.

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

## API and web page

```bash
PYTHONPATH=src python scripts/build_index.py --catalog-only   # data/catalog.parquet, if missing
PYTHONPATH=src uvicorn shopsense.api:app --port 8080
```

Open http://localhost:8080 for the search page, or call the API directly:

- `GET /search?q=red+shoes&pipeline=hybrid+cross_encoder_fast&k=10`: results, the parsed intent (for
  `intent+hybrid+llm`), latency and whether the LLM reranker fell back to the cross-encoder.
- `GET /pipelines`: every pipeline, its config and whether it is available.
- `GET /health`

Indexes and models load once at startup (about 1 GB of RAM); the catalog is never re-embedded. LLM pipelines
need the `.env` settings and Application Default Credentials, and are rate-limited per IP
(`SHOPSENSE_LLM_RATE_PER_MIN`, default 10).

## Docker

```bash
docker build -t shopsense .
docker run --rm -p 8080:8080 shopsense
```

The image holds the catalog, the dense vectors and both models, and runs uvicorn on `$PORT`. It never contains `.env`; pass
the LLM settings as environment variables if you want the LLM pipelines.

## Commands

- Tests: `pytest -q`
- Evaluate: `PYTHONPATH=src python scripts/evaluate.py --pipelines all` (or e.g. `--pipelines bm25,hybrid`)
- API: `PYTHONPATH=src uvicorn shopsense.api:app --reload`
