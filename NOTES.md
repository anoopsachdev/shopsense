# Notes: data, metrics, BM25, dense, hybrid and cross-encoder

## What was built

| Step | Files | Commit |
|---|---|---|
| 1 Scaffold | `pyproject.toml`, `.gitignore`, `README.md`, `src/shopsense/` | `b0aa97e` |
| 2 Data download | `scripts/download_data.py`, `tests/test_download_data.py` | `b2dcec8`, `25bca8d` |
| 3 Data loading | `src/shopsense/data.py`, `tests/test_data.py` | `610a0bc` |
| 4 Metrics | `src/shopsense/metrics.py`, `tests/test_metrics.py` | `889b84f` |
| 5 Retrieval | `src/shopsense/retrieval/base.py`, `bm25.py`, `tests/test_bm25.py` | `b47abc6` |
| 6 Evaluation | `scripts/evaluate.py`, `tests/test_evaluate.py` | `5b02c71` |
| 8 Dense | `src/shopsense/retrieval/dense.py`, `scripts/build_index.py`, tests | `1e8825b` |
| 9 Hybrid (RRF) | `src/shopsense/retrieval/hybrid.py`, `tests/test_hybrid.py` | `1ef723c` |
| 10 Cross-encoder | `src/shopsense/rerank/base.py`, `cross_encoder.py`, tests | `c91ba1a` |
| 11 Pipeline config | `src/shopsense/pipeline.py`, `tests/test_pipeline.py` | `e53d3ae` |
| 12 Evaluation table | `scripts/evaluate.py` (all pipelines, p50/p95), tests | `8bbc47f` |

`pytest -q`: 73 passed.

Data in `data/` (gitignored):
- `examples.parquet`: 601,354 rows (US, `small_version == 1`, both splits). This matches the README's US Task 1 count.
- `eval_query_ids.json`: seed 42, 500 eval IDs and 2,000 distractor IDs, with no overlap.
- `products.parquet`: all 482,105 US products in the Task 1 rows, 233 MB.
- `dense_bge-small-en-v1.5.npz`: 48,862 × 384 float32 product vectors plus their product IDs, 77 MB.
- `raw/`: the two original ESCI parquet files (1.2 GB), sha256-checked. You can delete them; the script downloads them again only if they're missing.

The **search catalog is fixed at 48,862 products**: the products labelled for the 2,500 sampled queries. `data.build_catalog` builds it from `eval_query_ids.json`, and every pipeline searches it, so results stay comparable with the BM25 row. The other ~433k products are stored but never indexed or embedded.

## Results (500 eval queries, seed 42, graded gains E3/S2/C1/I0, before the official-gains switch)

`python scripts/evaluate.py --pipelines all`, on a laptop CPU:

| Pipeline | Full NDCG@10 | Full Recall@50 | Full p50 ms | Full p95 ms | Rerank NDCG@10 | Rerank p50 ms | Rerank p95 ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.5270 | 0.6032 | 49.6 | 89.6 | 0.8293 | 1.2 | 1.3 |
| dense | 0.5627 | 0.6628 | 17.7 | 19.8 | 0.8517 | 2.9 | 3.9 |
| hybrid | 0.5828 | 0.6871 | 65.3 | 104.6 | 0.8451 | 4.4 | 5.2 |
| hybrid+cross_encoder | 0.6298 | 0.6871 | 1193.9 | 1428.0 | 0.8503 | 400.0 | 1161.9 |

- **Full catalog** is where the gains are. Dense beats BM25 by +0.036 NDCG@10. Hybrid adds another +0.020 and has the best Recall@50 (0.687). The cross-encoder takes NDCG@10 to 0.630, +0.103 over BM25, at about 1.2 s per query.
- **Hybrid and hybrid+cross_encoder have the same Recall@50 by construction.** The cross-encoder only re-orders the top 50 hybrid hits, so the set it returns is unchanged.
- **Re-ranking only** barely moves: 0.829 to 0.852. Most candidates are Exact matches. The cross-encoder (0.850) doesn't beat plain dense (0.852) here.
- **BM25 reproduces exactly** after the full products re-download and the venv rebuild. All 500 eval queries have at least one E/S/C product.
- **Latency** is search time per query, run sequentially, with indexing excluded. Dense p50 (18 ms) includes embedding the query. The cross-encoder dominates its pipeline's latency: 50 pairs of up to 512 tokens on CPU.
- **Embedding the catalog took 7,804 s (about 2 h 10 min)** on this laptop's CPU, at roughly 6 products/s and 2.7 GB peak RAM (`scripts/build_index.py`). It only needs to run again if the catalog, model or text fields change.

None of these numbers are comparable to the esci-data README's 0.83 baseline, because that used different gains and a different query set.

## Assumptions and decisions

1. **Gains: the headline is now the official ESCI gains (E=1.0, S=0.1, C=0.01, I=0).** These come from the repo's ranking scripts (`ranking/prepare_trec_eval_files.py`, `ranking/train.py`); the esci-data README itself gives none. Making them the headline puts the numbers on the same gain scale as published ESCI results (the query set and catalog still differ), and it stops substitutes from inflating scores. The graded gains (E=3, S=2, C=1, I=0, the original CLAUDE.md fallback and the headline until 2026-10-25) are still reported next to them. `ESCI_OFFICIAL_GAINS` is the default in `metrics.py`. `evaluate.py` prints both tables by default, official first; `--gains official|graded` prints one.
2. **NDCG** uses linear gains with a `1/log2(rank+1)` discount. The ideal ranking comes from all of the query's labelled products. Unlabelled products count as gain 0. A query with no positive-gain product scores 0, which never happens in the current eval set.
3. **Recall@k** is binary: relevant means gain > 0 (E, S or C), and recall is the share of those found in the top k. "Graded" applies to NDCG only.
4. **Distractor queries come from the train split**, 2,000 of them, so the test queries outside the eval set stay untouched. Total catalog queries: 2,500.
5. **Sampling** uses `random.Random(seed).sample` over sorted query IDs: the eval set first, then the distractors from the same RNG. `eval_query_ids.json` stores both lists plus the seed.
6. **products.parquet holds every US Task 1 product (482,105).** The first version stored only the 48,862 catalog products because the disk was full. Once you freed space, I re-downloaded everything (`25bca8d`). The query sample and the BM25 numbers came out identical.
7. **The data source is pinned.** The LFS object hashes and sizes from the repo's pointer files are hardcoded in `download_data.py`. The script gets a download URL from GitHub's LFS batch API, downloads the raw files into `data/raw/`, and filters them locally. The download resumes after drops and asks for a fresh signed URL on each attempt.
8. **BM25:** `BM25Okapi` with default parameters (k1=1.5, b=0.75). Tokenizer: lowercase, split on non-alphanumerics, no stemming or stopwords. Text is title + brand + color + bullet points; the description is left out. Ties keep catalog order. Zero-score products are still returned, so every candidate gets ranked in the re-ranking setting.
9. **Re-ranking setting** reuses the full-catalog BM25 index, so IDF comes from all 48,862 products, and it scores only the query's labelled candidates.
10. **Python 3.12 instead of 3.11.** 3.11 isn't installed. `pyproject.toml` says `>=3.11`. `.venv` is now an isolated venv built with `pip install -e ".[dev]"`, with no Anaconda packages visible. The first version reused Anaconda's site-packages, and its old torch, torchvision and torchaudio broke sentence-transformers.
11. **Evaluation helpers live in `scripts/evaluate.py`** to keep to the CLAUDE.md layout. Tests load the scripts through `importlib`. `--pipelines all` or a comma list chooses pipelines by name from `pipeline.PIPELINES`.
12. **Dense:** `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU), using `passage_embed` for products and `query_embed` for queries. I added no BGE query instruction prefix. It uses the same text fields as BM25, and the model truncates at 512 tokens. Vectors are L2-normalized, and search is brute-force cosine with numpy. 48,862 × 384 is small, so I didn't add FAISS.
13. **Saved vectors are tied to the catalog.** `DenseRetriever.index` loads `data/dense_bge-small-en-v1.5.npz` and raises if its product IDs differ from the catalog's, which means rerun `build_index.py`. If the file is missing, it embeds and saves. The vectors path is fixed at `data/...`; `evaluate.py --data-dir` doesn't move it.
14. **Hybrid:** RRF with k=60, scoring each product as the sum of `1/(60 + rank)`. Each retriever contributes its top 100, or k if larger. Ties keep first-appearance order. In the re-ranking setting, both retrievers score only the candidates.
15. **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` through sentence-transformers on CPU. I chose L-6, the cheapest MS MARCO MiniLM. The ESCI baseline used L-12, which would be about 2× slower. It reads the same text fields as the retrievers and re-orders the top 50 hybrid hits; any deeper hits keep hybrid order. In the re-ranking setting it re-orders all of a query's ~20 candidates.
16. **Pipeline configs** look like `{"retriever": ..., "reranker": ..., "rerank_depth": ...}`, with unknown keys or names raising `ValueError`. Hybrid is always BM25 + dense. Swapping a stage is a config change.
17. **Untested on purpose:** `FastEmbedEncoder` and `load_cross_encoder` aren't unit-tested, because they download models. Every class that uses them takes an injectable encoder or scorer, and tests pass fakes. The real models were exercised by `build_index.py` and the evaluation run.
18. **Model caches:** fastembed keeps bge-small under the system temp dir (`$TMPDIR/fastembed_cache`), which macOS may clear; it re-downloads in under a minute. The cross-encoder is in `~/.cache/huggingface`.
19. Nothing touches the LLM, GCP or the API.

## What failed or needs your attention

- **The disk was almost full on day one** (~1.1 GB free, ~160 MB at the worst point while swap grew). That's why the first commit stored only catalog products. Fixed once you freed space; ~13 GB is free now.
- **media.githubusercontent.com and, intermittently, raw.githubusercontent.com reset connections from this machine.** The workaround is GitHub's LFS batch API, which returns a `github-cloud.githubusercontent.com` URL that works. I added retries.
- **pyarrow 16's `pc.is_in` crashed (abort)** on the remote file, so product filtering uses a Python set instead.
- **The editable install isn't importable here.** macOS keeps putting the `hidden` flag on files under `.venv`, and Python 3.12 skips hidden `.pth` files. Pytest works through `pythonpath = ["src"]` in `pyproject.toml`. To run scripts, use:
  ```bash
  PYTHONPATH=src .venv/bin/python scripts/evaluate.py
  ```
- **The commit `b2dcec8` (download script) imports `shopsense.data`, which only lands in the next commit.** The data sampling had to happen before the product download. Every commit from `610a0bc` on runs.
- **Dependencies:** sentence-transformers 6 needs torch ≥ 2.6, but Anaconda has 2.3.1, and its torchvision and torchaudio clashed with a newer torch. So `.venv` was rebuilt as an isolated venv: torch 2.14.1, numpy 2.5.4, pyarrow 26, fastembed 0.9, sentence-transformers 6.2. That uses about 1.4 GB.
- **The network is slow and flaky here.** The first full re-download, which streamed range requests, timed out after 100 minutes at 1.22M of 1.81M rows. The resumable version finished in about 2 hours with 4 read timeouts, all recovered.
