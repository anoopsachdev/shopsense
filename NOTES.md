# Notes: steps 1–7 (data, metrics, BM25 baseline)

## What was built

| Step | Files | Commit |
|---|---|---|
| 1 Scaffold | `pyproject.toml`, `.gitignore`, `README.md`, `src/shopsense/` | `b0aa97e` |
| 2 Data download | `scripts/download_data.py`, `tests/test_download_data.py` | `b2dcec8`, `25bca8d` |
| 3 Data loading | `src/shopsense/data.py`, `tests/test_data.py` | `610a0bc` |
| 4 Metrics | `src/shopsense/metrics.py`, `tests/test_metrics.py` | `889b84f` |
| 5 Retrieval | `src/shopsense/retrieval/base.py`, `bm25.py`, `tests/test_bm25.py` | `b47abc6` |
| 6 Evaluation | `scripts/evaluate.py`, `tests/test_evaluate.py` | `5b02c71` |

`pytest -q`: 38 passed.

Data in `data/` (gitignored):
- `examples.parquet`: 601,354 rows (US, `small_version == 1`, both splits). This matches the README's US Task 1 count.
- `eval_query_ids.json`: seed 42, 500 eval IDs and 2,000 distractor IDs, with no overlap.
- `products.parquet`: all 482,105 US products in the Task 1 rows, 233 MB.
- `raw/`: the two original ESCI parquet files (1.2 GB), sha256-checked. You can delete them; the script downloads them again only if they're missing.

The **search catalog is fixed at 48,862 products**: the products labelled for the 2,500 sampled queries. `data.build_catalog` builds it from `eval_query_ids.json`, and every pipeline searches it, so results stay comparable with the BM25 row. The other ~433k products are stored but never indexed or embedded.

## BM25 results (500 eval queries, seed 42)

```
catalog: 48,862 products from 2,500 queries
BM25 index built in 4.4s
BM25 full catalog:    NDCG@10 = 0.5270   Recall@50 = 0.6032   latency = 64.8 ms/query
BM25 re-ranking only: NDCG@10 = 0.8293
```

All 500 eval queries have at least one E/S/C product. The re-ranking-only number is not comparable to the README's 0.83 baseline, because that used different gains (see below) and a different query set.

## Assumptions and decisions

1. **Gains: E=3, S=2, C=1, I=0.** The esci-data README gives no Task 1 gains, so I used the CLAUDE.md fallback. The repo's own eval code uses different gains: E=1.0, S=0.1, C=0.01, I=0 (`ranking/prepare_trec_eval_files.py` and `ranking/train.py`). To switch, change `ESCI_GAINS` in `metrics.py`, or pass `gains=`.
2. **NDCG** uses linear gains with a `1/log2(rank+1)` discount. The ideal ranking comes from all of the query's labelled products. Unlabelled products count as gain 0. A query with no positive-gain product scores 0, which never happens in the current eval set.
3. **Recall@k** is binary: relevant means gain > 0 (E, S or C), and recall is the share of those found in the top k. "Graded" applies to NDCG only.
4. **Distractor queries come from the train split**, 2,000 of them, so the test queries outside the eval set stay untouched. Total catalog queries: 2,500.
5. **Sampling** uses `random.Random(seed).sample` over sorted query IDs: the eval set first, then the distractors from the same RNG. `eval_query_ids.json` stores both lists plus the seed.
6. **products.parquet holds every US Task 1 product (482,105).** The first version stored only the 48,862 catalog products because the disk was full. Once you freed space, I re-downloaded everything (`25bca8d`). The query sample and the BM25 numbers came out identical.
7. **The data source is pinned.** The LFS object hashes and sizes from the repo's pointer files are hardcoded in `download_data.py`. The script gets a download URL from GitHub's LFS batch API, downloads the raw files into `data/raw/`, and filters them locally. The download resumes after drops and asks for a fresh signed URL on each attempt.
8. **BM25:** `BM25Okapi` with default parameters (k1=1.5, b=0.75). Tokenizer: lowercase, split on non-alphanumerics, no stemming or stopwords. Text is title + brand + color + bullet points; the description is left out. Ties keep catalog order. Zero-score products are still returned, so every candidate gets ranked in the re-ranking setting.
9. **Re-ranking setting** reuses the full-catalog BM25 index, so IDF comes from all 48,862 products, and it scores only the query's labelled candidates.
10. **Python 3.12 instead of 3.11.** 3.11 isn't installed. `pyproject.toml` says `>=3.11`. `.venv` is now an isolated venv built with `pip install -e ".[dev]"`, with no Anaconda packages visible. The first version reused Anaconda's site-packages, and its old torch, torchvision and torchaudio broke sentence-transformers.
11. **Evaluation helpers live in `scripts/evaluate.py`** to keep to the CLAUDE.md layout. Tests load the scripts through `importlib`. `evaluate.py` has a `--data-dir` flag but no `--pipelines` flag yet, because there is only one pipeline.
12. I created no `rerank/`, `llm/`, `pipeline.py`, `api.py` or `web/`: those are outside steps 1–7.

## What failed or needs your attention

- **The disk was almost full on day one** (~1.1 GB free, ~160 MB at the worst point while swap grew). That's why the first commit stored only catalog products. Fixed once you freed space; ~13 GB is free now.
- **media.githubusercontent.com and, intermittently, raw.githubusercontent.com reset connections from this machine.** The workaround is GitHub's LFS batch API, which returns a `github-cloud.githubusercontent.com` URL that works. I added retries.
- **pyarrow 16's `pc.is_in` crashed (abort)** on the remote file, so product filtering uses a Python set instead.
- **The editable install isn't importable here.** macOS keeps putting the `hidden` flag on files under `.venv`, and Python 3.12 skips hidden `.pth` files. Pytest works through `pythonpath = ["src"]` in `pyproject.toml`. To run scripts, use:
  ```bash
  PYTHONPATH=src .venv/bin/python scripts/evaluate.py
  ```
- **The commit `b2dcec8` (download script) imports `shopsense.data`, which only lands in the next commit.** The data sampling had to happen before the product download. Every commit from `610a0bc` on runs.
- **The network is slow and flaky here.** The first full re-download, which streamed range requests, timed out after 100 minutes at 1.22M of 1.81M rows. The resumable version finished in about 2 hours with 4 read timeouts, all recovered.
