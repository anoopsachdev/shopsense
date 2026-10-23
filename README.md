# ShopSense

LLM-assisted product search on Amazon's [ESCI shopping-queries dataset](https://github.com/amazon-science/esci-data),
scored against human relevance labels on quality, latency and cost per query.

Status: data loading, metrics and a BM25 baseline. Embeddings, re-ranking, LLM stages and the API are not built yet.

## Results

500 held-out test queries (US, Task 1 small version, seed 42).

**Full catalog** (the main setting): each query searches all 48,862 catalog products, which are the products
labelled for its own query plus 2,499 others. This is the setting with room to improve.

| Pipeline | NDCG@10 | Recall@50 |
|---|---|---|
| BM25 | 0.527 | 0.603 |

**Re-ranking only** (the original ESCI Task 1 setup): each query ranks only its own ~20 labelled candidates.
Most candidates are Exact matches, so BM25 already scores high and there is little headroom.

| Pipeline | NDCG@10 |
|---|---|
| BM25 | 0.829 |

### How scores are computed

- **NDCG gains:** Exact 3, Substitute 2, Complement 1, Irrelevant 0, with unlabelled products counting as 0. The esci-data README
  gives no Task 1 gains, so this is the fallback from CLAUDE.md. Note that it gives substitutes two-thirds credit, which
  pushes scores up. The esci-data repo's own evaluation script (`ranking/prepare_trec_eval_files.py`) uses
  E 1.0, S 0.1, C 0.01, I 0. The numbers here are therefore **not** comparable to the README's published 0.83 baseline.
  All pipelines use the same gains (`ESCI_GAINS` in `src/shopsense/metrics.py`).
- **Recall@50:** the share of a query's Exact, Substitute and Complement products found in the top 50.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py   # US-locale, Task 1 (small_version) parquet into data/
```

## Commands

- Tests: `pytest -q`
- Evaluate: `python scripts/evaluate.py`
