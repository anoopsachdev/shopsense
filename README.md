# ShopSense

LLM-assisted product search on Amazon's [ESCI shopping-queries dataset](https://github.com/amazon-science/esci-data),
scored against human relevance labels on quality, latency and cost per query.

Status: BM25, dense (bge-small), hybrid (RRF) and cross-encoder pipelines, all scored by one harness. LLM stages and the API are not built yet.

## Results

500 held-out test queries (US, Task 1 small version, seed 42).

**Full catalog** (the main setting): each query searches all 48,862 catalog products, which are the products
labelled for its own query plus 2,499 others. This is the setting with room to improve.

| Pipeline | NDCG@10 | Recall@50 | p50 latency |
|---|---|---|---|
| BM25 | 0.527 | 0.603 | 50 ms |
| Dense (bge-small) | 0.563 | 0.663 | 18 ms |
| Hybrid (RRF of BM25 + dense) | 0.583 | 0.687 | 65 ms |
| Hybrid + cross-encoder (top 50) | **0.630** | 0.687 | 1,194 ms |

**Re-ranking only** (the original ESCI Task 1 setup): each query ranks only its own ~20 labelled candidates.
Most candidates are Exact matches, so BM25 already scores high and there is little headroom.

| Pipeline | NDCG@10 |
|---|---|
| BM25 | 0.829 |
| Dense | 0.852 |
| Hybrid | 0.845 |
| Hybrid + cross-encoder | 0.850 |

### How scores are computed

- **NDCG gains:** Exact 3, Substitute 2, Complement 1, Irrelevant 0, with unlabelled products counting as 0. The esci-data README
  gives no Task 1 gains, so this is the fallback from CLAUDE.md. Note that it gives substitutes two-thirds credit, which
  pushes scores up. The esci-data repo's own evaluation script (`ranking/prepare_trec_eval_files.py`) uses
  E 1.0, S 0.1, C 0.01, I 0. The numbers here are therefore **not** comparable to the README's published 0.83 baseline.
  All pipelines use the same gains (`ESCI_GAINS` in `src/shopsense/metrics.py`).
- **Recall@50:** the share of a query's Exact, Substitute and Complement products found in the top 50.
- **Latency:** median search time per query on a laptop CPU, one query at a time, indexing excluded. p95 and
  re-ranking-only latencies are in `NOTES.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py   # US-locale, Task 1 (small_version) parquet into data/
python scripts/build_index.py     # embed the catalog once (about 2 h on a laptop CPU)
```

## Commands

- Tests: `pytest -q`
- Evaluate: `python scripts/evaluate.py --pipelines all` (or e.g. `--pipelines bm25,hybrid`)
