# Notes: data, retrieval, reranking and the LLM layer

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
| 13 Official gains headline | `metrics.py`, `evaluate.py --gains`, README | `b547f72` |
| 14 Cross-encoder speed | `max_length=256`, `hybrid+cross_encoder_fast` (depth 20) | `ed8fb8d` |
| 15 Dense build guard | `DenseRetriever(allow_build=False)` | `6e4a960` |
| 16 CLAUDE.md in repo | Python 3.12, `PYTHONPATH=src` commands | `41c1737` |
| 17 LLM preflight | `.env.example`, `.gitignore` | `d7ad76e` |
| 18 LLM client | `src/shopsense/llm/client.py` | `768ac84` |
| 19 Query intent | `src/shopsense/llm/query_intent.py` | `c416c24` |
| 20 LLM reranker | `src/shopsense/rerank/llm.py` | `707698d` |
| 21 LLM pipelines | `hybrid+llm`, `intent+hybrid+llm` in `pipeline.py` | `5683869` |
| 22 LLM evaluation | `evaluate.py`: 200-query subset, cost, slices, regressions | `d93debc` |

`pytest -q`: 153 passed.

Data in `data/` (gitignored):
- `examples.parquet`: 601,354 rows (US, `small_version == 1`, both splits). This matches the README's US Task 1 count.
- `eval_query_ids.json`: seed 42, 500 eval IDs and 2,000 distractor IDs, with no overlap.
- `products.parquet`: all 482,105 US products in the Task 1 rows, 233 MB.
- `dense_bge-small-en-v1.5.npz`: 48,862 × 384 float32 product vectors plus their product IDs, 77 MB.
- `raw/`: the two original ESCI parquet files (1.2 GB), sha256-checked. You can delete them; the script downloads them again only if they're missing.

The **search catalog is fixed at 48,862 products**: the products labelled for the 2,500 sampled queries. `data.build_catalog` builds it from `eval_query_ids.json`, and every pipeline searches it, so results stay comparable with the BM25 row. The other ~433k products are stored but never indexed or embedded.

## Results (2026-10-25 run, `python scripts/evaluate.py --pipelines all`)

The headline uses **official ESCI gains** (E 1.0, S 0.1, C 0.01, I 0); graded-gain tables follow. Non-LLM pipelines ran one query at a time on all 500 eval queries, before any LLM work started. LLM pipelines ran on the fixed 200-query subset, 8 queries at a time. Cross-encoders use `max_length=256`.

### All 500 eval queries, official gains (headline)

| Pipeline | Full NDCG@10 | Full Recall@50 | Full p50 ms | Full p95 ms | Rerank NDCG@10 | Rerank p50 ms | Rerank p95 ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.4677 | 0.6032 | 63.5 | 118.0 | 0.7061 | 1.9 | 2.1 |
| dense | 0.5064 | 0.6628 | 43.0 | 61.7 | 0.7339 | 8.7 | 14.4 |
| hybrid | 0.5195 | 0.6871 | 102.5 | 201.7 | 0.7288 | 14.0 | 47.5 |
| hybrid+cross_encoder | 0.5738 | 0.6871 | 1156.1 | 2375.1 | 0.7442 | 426.7 | 996.2 |
| hybrid+cross_encoder_fast | 0.5643 | 0.6871 | 632.0 | 1207.5 | 0.7447 | 387.5 | 682.7 |

**The fast cross-encoder** reranks the top 20 instead of 50. It's about 1.8× faster on the full catalog (p50 632 vs 1,156 ms; p95 1,208 vs 2,375 ms), costs 0.0095 NDCG@10, and is tied in the re-ranking setting. In the previous run, with `max_length=512` and depth 50, the cross-encoder scored 0.6298 graded at 1,194 ms p50; at 256 it scores 0.6300 graded at 1,156 ms. So truncating to 256 lost nothing.

### LLM subset (200 queries), official gains (headline)

| Pipeline | Full NDCG@10 | Full Recall@50 | Full p50 ms | Full p95 ms | Full $/1K q | Full fallback | Rerank NDCG@10 | Rerank p50 ms | Rerank p95 ms | Rerank $/1K q | Rerank fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.4703 | 0.5941 | 62.8 | 105.1 | 0.00 | 0.0% | 0.7111 | 1.9 | 2.1 | 0.00 | 0.0% |
| dense | 0.4755 | 0.6489 | 43.1 | 63.1 | 0.00 | 0.0% | 0.7212 | 8.9 | 15.6 | 0.00 | 0.0% |
| hybrid | 0.5041 | 0.6803 | 105.2 | 197.3 | 0.00 | 0.0% | 0.7298 | 14.0 | 47.0 | 0.00 | 0.0% |
| hybrid+cross_encoder | 0.5772 | 0.6803 | 1170.2 | 2163.0 | 0.00 | 0.0% | 0.7493 | 419.4 | 955.1 | 0.00 | 0.0% |
| hybrid+cross_encoder_fast | 0.5651 | 0.6803 | 624.7 | 1207.5 | 0.00 | 0.0% | 0.7499 | 384.4 | 669.7 | 0.00 | 0.0% |
| hybrid+llm | 0.6004 | 0.6803 | 53315.0 | 117075.2 | 1.14 | 27.5% | 0.8006 | 63457.2 | 115861.2 | 0.76 | 45.0% |
| intent+hybrid+llm | **0.6074** | 0.6626 | 3119.9 | 116815.5 | 1.42 | 23.5% | **0.8008** | 2489.8 | 115146.4 | 1.12 | 35.0% |

- **The fallback columns understate the LLM.** Fallback queries were ranked by the cross-encoder, so these rows mix LLM and cross-encoder rankings. Almost all fallbacks came from **HTTP 500s from `gemini-3.8-flash`**, not timeouts: the log has 377 "ServerError, retry" lines and 1 ClientError. A call that fails twice falls back.
- **`intent+hybrid+llm`'s p50 latency (3.1 s) is not comparable.** Whenever its intent changed nothing, its rerank prompt matched `hybrid+llm`'s exactly, so it was answered from the cache. Its p95 (117 s) is the real uncached latency: two 60 s attempts.
- **Spend:** 938 LLM calls, of which 388 were cache hits. List-price cost was $0.89; actually billed was $0.57. Total logged spend is **$0.59 of the $3 cap**.

### Graded gains (E 3, S 2, C 1, I 0)

| Pipeline (500 queries) | Full NDCG@10 | Full Recall@50 | Rerank NDCG@10 |
|---|---|---|---|
| bm25 | 0.5270 | 0.6032 | 0.8293 |
| dense | 0.5627 | 0.6628 | 0.8517 |
| hybrid | 0.5828 | 0.6871 | 0.8451 |
| hybrid+cross_encoder | 0.6300 | 0.6871 | 0.8509 |
| hybrid+cross_encoder_fast | 0.6214 | 0.6871 | 0.8508 |

| Pipeline (200-query LLM subset) | Full NDCG@10 | Rerank NDCG@10 |
|---|---|---|
| hybrid+cross_encoder_fast | 0.6159 | 0.8521 |
| hybrid+llm | 0.6494 | 0.8884 |
| intent+hybrid+llm | 0.6530 | 0.8873 |

Latency, cost and fallback rate are the same as in the official tables; only NDCG changes with the gains. BM25 graded numbers match the earlier runs exactly.

### NDCG@10 by query slice (LLM subset, full catalog, official gains)

| Pipeline | all (n=200) | short (n=41) | long (n=26) | brand (n=32) | negation (n=12) |
|---|---|---|---|---|---|
| bm25 | 0.4703 | 0.3814 | 0.4756 | 0.6200 | 0.4870 |
| dense | 0.4755 | 0.3655 | 0.4742 | 0.6288 | 0.4090 |
| hybrid | 0.5041 | 0.4140 | 0.5204 | 0.6446 | 0.4819 |
| hybrid+cross_encoder | 0.5772 | 0.4537 | 0.5783 | 0.7471 | 0.5031 |
| hybrid+cross_encoder_fast | 0.5651 | 0.4372 | 0.5678 | 0.7517 | 0.4721 |
| hybrid+llm | 0.6004 | 0.4934 | 0.5326 | 0.8022 | 0.4352 |
| intent+hybrid+llm | 0.6074 | 0.5136 | 0.5622 | 0.8139 | 0.3721 |

### Biggest regressions against hybrid+cross_encoder_fast (full catalog, official gains)

The 10 worst of the 20 printed for each pipeline (the full lists are in the run log):

| Query | fast CE | hybrid+llm | intent+hybrid+llm |
|---|---|---|---|
| cemetery without crosses | 0.884 | (not among its 20 worst) | 0.030 (fell back) |
| loud bluetooth headset | 0.608 | (not among its 20 worst) | 0.052 |
| 1 ear headset without microphone | 0.553 | 0.000 | 0.000 |
| food freeze dryer machine not dehydrator | 0.512 | 0.108 | 0.191 (fell back) |
| facet machine lapidary | 0.963 | 0.580 | 0.580 |
| car ramps and anti skid | 0.694 | 0.356 | 0.356 |
| adidas sweats slim | 0.764 | 0.445 | 0.445 |
| cowboy boot spurs | 0.290 | 0.000 | 0.000 |
| liquor opener | 0.649 | 0.359 | 0.359 |
| 48 round dining table | 0.744 | 0.522 | 0.522 |

### Three findings

1. **The LLM reranker ranks best, but it can't serve live traffic as configured.** On the full catalog, `hybrid+llm` beats the fast cross-encoder by +0.035 NDCG@10 (0.600 vs 0.565) and the depth-50 one by +0.023. In the re-ranking setting it gains +0.05 (0.801 vs 0.750), even though 27–45% of its queries actually fell back to the cross-encoder. It costs about $1.14 per 1,000 queries. But the model took about 50 s per call under load (p95 117 s, two 60 s attempts) and returned HTTP 500s often. As it stands it's suitable for offline judging or caching popular queries, not interactive search.
2. **The gains come from short and brand queries.** On brand queries the LLM scores 0.80–0.81 against 0.75 for the fast cross-encoder; on short queries, 0.49–0.51 against 0.44. On long queries (6+ words) plain `hybrid+llm` is *worse* than the fast cross-encoder (0.533 vs 0.568). Our 300-character product snippets probably cut off the details those queries ask about.
3. **Taking "without X" literally scores worse against ESCI labels.** On the 12 negation queries the LLM drops to 0.435, and intent exclusions drop it to 0.372, against 0.472 for the fast cross-encoder. The worst losses are queries like "cemetery without crosses" (0.88 → 0.03) and "1 ear headset without microphone" (0.55 → 0). Annotators often labelled products as Exact even when they include the excluded thing, so removing them is penalized. Overall the intent stage nets only +0.007 NDCG@10, and it costs Recall@50 (0.663 vs 0.680) because of the products it drops. The slice is small (n=12), so treat this as a direction, not a measurement.

## Rerank answer format: before and after (2026-10-25)

The rerank model only ever returned product numbers, but as JSON mode's pretty-printed `{"ranking": [...]}`, one number per line: about 102 output tokens for 20 products. The reranker now asks for the numbers alone, comma-separated (`3,1,2,...`), with no JSON schema. `parse_ranking` accepts only digits and commas, so anything else falls back to the cross-encoder.

I re-timed the 20 first queries of the LLM subset. Both formats got the same top-20 hybrid hits and the production settings (60 s timeout, 1 retry), one call at a time with the cache bypassed (`generate(..., refresh=True)`).

| Format | Answered | Output tokens (median / mean) | Median latency | p90 latency | List cost per call | NDCG@10 |
|---|---|---|---|---|---|---|
| Before: `{"ranking": [...]}` JSON | 20/20 | 102 / 99.2 | 15.5 s | 59.8 s | $0.00164 | 0.5807 |
| After: comma-separated IDs | 19/20 | 50 / 64.4 | 17.5 s | 63.9 s | $0.00148 | 0.5852 |

- NDCG@10 (official gains, full catalog) is over the 19 queries both formats answered. The one miss was a 429 RESOURCE_EXHAUSTED, not a parsing failure.
- The "after" mean includes one 323-token call. The model occasionally reports thinking tokens even with `thinking_budget=0`.
- **Output tokens halved, cost per call fell about 10%** (input, about 1,500 tokens, dominates), and **quality held**. **Latency didn't change.** It's set by the server: calls take either about 2 s or close to the timeout, whichever format is used.
- The re-timing cost $0.064. Total logged LLM spend is now **$0.67 of $3**. I didn't rerun the full evaluation, so the tables above are from the JSON format.

## LLM preflight (2026-10-25)

| Check | Result |
|---|---|
| `.env` with the given values | ✅ created (gitignored) |
| `.env.example` with placeholders | ✅ committed. Fixed: the `.env.*` rule was ignoring it, so I added `!.env.example`. |
| `.env` ignored and not staged | ✅ (`git check-ignore` matches it; nothing staged) |
| gcloud installed | ✅ Google Cloud SDK 581.0.0 (`/opt/homebrew/bin/gcloud`) |
| Application Default Credentials | ✅ `~/.config/gcloud/application_default_credentials.json`, type authorized_user |
| Quota project shopsense-512615 | ✅ ADC `quota_project_id` and `gcloud config` project both match |
| `gemini-3.5-flash-lite` @ global | ✅ answered "ok" (7 in, 1 out) |
| `gemini-3.8-flash` @ global | ✅ answered "ok" once thinking was off. The first call, capped at 16 output tokens, spent them all on thinking and returned no text. One default-settings call returned a transient 500, which passed on retry. |

Neither model returned "not found", so `GOOGLE_CLOUD_LOCATION` stays `global`.

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
13. **Saved vectors are tied to the catalog and never rebuilt by accident.** `DenseRetriever.index` loads `data/dense_bge-small-en-v1.5.npz`. If the file is missing it raises `FileNotFoundError`; if its product IDs differ from the catalog's it raises `ValueError`. Both messages say to run `build_index.py`. Only `build_index.py` passes `allow_build=True`, which embeds and saves instead. The vectors path is fixed at `data/...`; `evaluate.py --data-dir` doesn't move it.
14. **Hybrid:** RRF with k=60, scoring each product as the sum of `1/(60 + rank)`. Each retriever contributes its top 100, or k if larger. Ties keep first-appearance order. In the re-ranking setting, both retrievers score only the candidates.
15. **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` through sentence-transformers on CPU. I chose L-6, the cheapest MS MARCO MiniLM. The ESCI baseline used L-12, which would be about 2× slower. It reads the same text fields as the retrievers. Pairs are truncated at `max_length=256` tokens; the model allows 512, and the results before 2026-10-25 used 512. `hybrid+cross_encoder` re-orders the top 50 hybrid hits, and `hybrid+cross_encoder_fast` re-orders the top 20; any deeper hits keep hybrid order. In the re-ranking setting both re-order all of a query's ~20 candidates.
16. **Pipeline configs** look like `{"retriever": ..., "intent": ..., "reranker": ..., "rerank_depth": ...}`, with unknown keys or names raising `ValueError`. Hybrid is always BM25 + dense. Swapping a stage is a config change.
17. **Untested on purpose:** `FastEmbedEncoder` and `load_cross_encoder` aren't unit-tested, because they download models. Every class that uses them takes an injectable encoder or scorer, and tests pass fakes. The real models were exercised by `build_index.py` and the evaluation run.
18. **Model caches:** fastembed keeps bge-small under the system temp dir (`$TMPDIR/fastembed_cache`), which macOS may clear; it re-downloads in under a minute. The cross-encoder is in `~/.cache/huggingface`.
19. **CLAUDE.md now lives in the repo root**, copied from the brief in `~/Downloads`, which I left unchanged. Besides Python 3.12 and the `PYTHONPATH=src` commands, I changed its gains rule to match the official-gains headline, because it would otherwise contradict step 13.

### LLM layer

20. **Settings** come from `.env`; `.env.example` holds placeholders. Project, location (`global`), model names and USD prices per 1M tokens all come from there, and nothing is hardcoded. Auth is ADC. `llm/client.py` has its own 10-line `.env` reader rather than adding python-dotenv.
21. **One wrapper for every call** (`LLMClient.generate`):
    - Responses are cached in `data/llm_cache.sqlite`, keyed by sha256 of (model, system prompt, schema, prompt).
    - Every call is logged: to the `shopsense.llm` logger, and to a `calls` table with tokens, latency, list-price cost and billed cost. Cache hits are billed at $0.
    - Retries back off exponentially on any exception.
    - The client is thread-safe.
22. **Thinking is off** (`thinking_budget=0`) and temperature is 0. In the preflight, `gemini-3.8-flash` spent 98 thinking tokens, billed as output, just to say "ok". `thinking_level="MINIMAL"` is rejected with a 400 by that model. Even at temperature 0, it doesn't return the same ranking twice for the same prompt. The cache makes reruns repeatable.
23. **The $3 cap** is the sum of `billed_usd` in the call log, across all runs. Once it's reached, `generate` raises `BudgetExceeded`, a separate error that fallbacks don't catch, before making any new call. `evaluate.py` then stops the LLM pipelines and prints what finished. With 8 queries in flight it can overshoot by those 8 calls (about $0.02).
24. **Query intent** uses `gemini-3.5-flash-lite` with a JSON response schema: `{product_type, brand, color, attributes, exclusions}`. It has a 15 s timeout and 2 retries. Answers are lowercased, deduped and validated. Any failure gives a neutral intent: the query as `product_type` and nothing else. The prompt says brand means the product's own brand. That fixed a real mistake: on the first try, "iphone 14 case" got brand "apple".
25. **The intent stage** (in `intent+hybrid+llm`) runs on the hybrid hits before reranking:
    - It **drops** hits whose *title* mentions an exclusion as whole words. I match the title only, because bullet points often name things in passing ("not compatible with…").
    - It **boosts** by a stable sort on the number of brand and color matches (0–2). Brand matches the brand field or title; color matches the color field or title.
    - The retriever still gets the raw query.
26. **The LLM reranker** (`rerank/llm.py`, `gemini-3.8-flash`) is listwise:
    - One call per query covers the top 20 hybrid hits. Each product is numbered and cut to 300 characters, title first. The system prompt spells out the ESCI E/S/C/I definitions and says to respect exclusions.
    - The answer is the numbers only, comma-separated, with no JSON schema (since 2026-10-25; see "Rerank answer format"). The evaluation tables used the earlier `{"ranking": [...]}` JSON format.
    - Numbers the model leaves out keep hybrid order. Out-of-range or repeated numbers are ignored.
    - Timeout is 60 s with 1 retry. Sequential calls took 20–40 s, so 20 s would have timed out most of them.
    - On timeout, error or an unusable answer, the cross-encoder (max_length 256) orders the same 20 hits, which is exactly `hybrid+cross_encoder_fast`. The query is recorded in `fallback_queries`.
27. **The LLM subset** is 200 eval queries, `data.llm_eval_ids`: drawn from the 500 eval IDs with the sample's seed, so it's fixed for a given `eval_query_ids.json`. It isn't written to the JSON file, because it's derived from it. Non-LLM pipelines run on all 500 queries, and their LLM-subset rows are the same runs restricted to the 200.
28. **Concurrency:** LLM pipelines run 8 queries at a time (`--llm-workers`), so the 200 × 2 settings × 2 pipelines finish in about 1.5 h, not about 5 h. Their latency is per query, measured under that load. Non-LLM pipelines run one query at a time, before any LLM work starts.
29. **Cost per 1K queries** is at list price, counting cache hits as if paid. That's what serving would cost. `intent+hybrid+llm` reuses `hybrid+llm`'s cached rerank answer whenever its intent changes nothing, so its real bill is lower than its listed cost.
30. **Slices** (on the 200 subset):
    - short: ≤ 2 words.
    - long: ≥ 6 words.
    - brand: the query names, as whole words, the brand of one of its Exact products, ignoring generic brands.
    - negation: no / not / without / non / except / excluding / free, or an X-free / non-X word.
31. **Regressions** compare per-query NDCG@10 (official gains, full catalog) against `hybrid+cross_encoder_fast`.
32. **Untested on purpose:** the real Vertex calls. `genai_backend` is tested against a faked `genai.Client` that checks the request config. Everything else uses fake clients. The real API was exercised by the preflight, two spot checks of intent and rerank, an 8-query smoke test and the evaluation run.
33. Nothing touches GCP beyond Vertex AI calls. Nothing is deployed and no cloud resources were created.

### API

34. **`src/shopsense/api.py`** (FastAPI, Pydantic models) serves `GET /search?q=&pipeline=&k=`, `GET /pipelines` and `GET /health`. `/search` returns the results (rank, ID, title, brand, color, score), the parsed intent, which is set only for `intent+hybrid+llm`, the latency, and a `fell_back` flag for this request. Defaults are `pipeline=hybrid+cross_encoder_fast` and `k=10`; `k` runs 1–50 and `q` 1–200 characters, with anything else returning 422. An unknown pipeline returns 400, an unavailable one 503, and a reached LLM budget 503.
35. **Loading once:** at startup the API reads `data/catalog.parquet`, written by `build_index.py --catalog-only` from the same sample. It builds every pipeline from **shared stages** (`build_shared_pipelines`): one BM25 index, one set of vectors, one cross-encoder that is also the LLM fallback, and one LLM client per role. Indexing the same catalog object twice is a no-op. One warm-up search per non-LLM pipeline loads the models before the first request. The dense stage runs with `allow_build=False`, so missing vectors stop startup with the `build_index.py` message (tested). The catalog is never embedded.
36. **LLM availability:** LLM pipelines are listed as unavailable, with the reason, when `.env` or the environment lacks the Vertex settings, or when building their clients fails. The other pipelines still serve.
37. **Rate limit:** an in-memory sliding window allows `SHOPSENSE_LLM_RATE_PER_MIN` (default 10) LLM searches per minute per client IP. The IP is the first `X-Forwarded-For` entry, which Cloud Run sets, otherwise the socket peer. Over the limit it returns 429 with `Retry-After: 60`. The window is per process, so two Cloud Run instances each allow 10. `X-Forwarded-For` can be spoofed, so this is a guard against accidents, not a security boundary. Non-LLM pipelines aren't limited.
38. **Per-request intent and fallback** come from `Pipeline.search_details`, which returns `SearchDetails(hits, intent, fell_back)`, and `LLMReranker.rerank_with_status`. No shared "last call" state, so concurrent requests can't mix them up.
39. **Run locally:** `PYTHONPATH=src uvicorn shopsense.api:app --port 8080`, with all 7 pipelines available. The resident set was **~950 MB right after startup** and settled around 640 MB after a few searches. The default pipeline answered in about 1.1 s, BM25 in 58 ms, and `intent+hybrid+llm` in 8.4 s. Missing or bad parameters returned 422, and an unknown pipeline 400.
40. **Web page:** `web/index.html` is plain HTML, CSS and JS with no build step, served at `/` (`SHOPSENSE_WEB_DIR` can move it). It has a search box and a pipeline dropdown that defaults to `hybrid+cross_encoder_fast`; unavailable pipelines are disabled, with the reason as a tooltip. It shows the results, latency, a badge when the LLM fell back, and the parsed intent as a small table. The query and pipeline live in the URL, so a search can be shared. All text goes in through `textContent`, never as HTML. It supports light and dark mode and has no horizontal scroll at 375 px; I checked both in the in-app browser.

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
- **The rerank model is slow:** 20–40 s per 20-product call when sequential, and about 54 s with 8 in flight (the 8-query smoke test). The answer is only about 60 tokens, so most of the time is queueing or decoding on the server, not output length. That's why the timeout is 60 s and the evaluation runs queries concurrently.
- **The network is slow and flaky here.** The first full re-download, which streamed range requests, timed out after 100 minutes at 1.22M of 1.81M rows. The resumable version finished in about 2 hours with 4 read timeouts, all recovered.

## Ready for deploy

Nothing is deployed, pushed or created in the cloud. Here's where the container stands.

**Not built yet: Docker isn't installed on this Mac** (`docker: command not found`). Installing Docker Desktop or colima needs admin rights, and the disk has about 9 GB free, so I didn't. The `Dockerfile` and `.dockerignore` are committed but **unbuilt**. As the closest check, I ran the image's runtime setup natively:
- a folder holding only what the image copies in (`data/catalog.parquet`, the vectors, `web/`), with no `.env`;
- `HF_HUB_OFFLINE=1` and the baked-model cache path;
- the Dockerfile's own command, `sh -c 'exec uvicorn shopsense.api:app --host 0.0.0.0 --port ${PORT}'`, with `PORT=8090`.

Results: `/health` returned ok with 48,862 products. `/search?q=stainless+steel+water+bottle` returned water bottles in 432 ms. Both LLM pipelines were reported unavailable with "LLM settings missing", and `/` served the page.

| | Value | How it was measured |
|---|---|---|
| Image size | **about 2 GB uncompressed (estimate, not measured)** | Python packages are 1.49 GB in the macOS venv (torch 583 MB, pyarrow 127 MB, transformers 114 MB, scipy 99 MB, pandas 80 MB, onnxruntime 80 MB). The Linux CPU torch wheel is somewhat larger. Add `python:3.12-slim` (~130 MB), the baked models (~155 MB: bge-small ONNX 65 MB, MiniLM cross-encoder ~90 MB) and the data (catalog 36 MB, vectors 74 MB). |
| Memory at startup | **~950 MB resident** | 941 MB without LLM pipelines (container-style run) and 952 MB with all 7 (local run), after indexing and warm-up. It settled around 640 MB after a few searches. |
| Startup work | BM25 index 6–10 s, plus loading two models and the warm-up searches | Fits a startup probe on `/health` with a 120 s or longer budget. |

**What you need to do before deploying:**
1. **Build and run the image once yourself.** Install Docker Desktop or colima, then run:
   ```bash
   docker build --platform linux/amd64 -t shopsense .
   docker run --rm -p 8080:8080 shopsense
   ```
   Then curl `/health` and `/search?q=red+shoes`, and check `docker images` for the real size. Use `--platform linux/amd64` on Apple Silicon, since Cloud Run runs amd64. The build needs 3–4 GB of free disk.
2. **Size Cloud Run at 2 GiB of memory and 2 vCPU.** 1 GiB is too tight for ~950 MB resident plus request spikes. The cross-encoder is CPU-bound, so one vCPU roughly doubles its 0.6–1.2 s latency. Consider `--cpu-boost` and min-instances 1 to avoid slow cold starts, which cost money while idle.
3. **Ship the data separately.** `data/catalog.parquet` and `data/dense_bge-small-en-v1.5.npz` are gitignored, so a CI build (GitHub Actions) won't have them. Upload both to the Cloud Storage bucket from CLAUDE.md and have the workflow download them before `docker build`. Rebuild them with `build_index.py` only if the sample changes. `--catalog-only` writes the catalog without re-embedding.
4. **For LLM pipelines on Cloud Run:**
   - Set the `.env` keys as service environment variables: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, both `GEMINI_*_MODEL` and the four `*_PRICE_*`. The image never contains `.env`.
   - Give the service account `roles/aiplatform.user`. ADC then works with no key.
   - Without these, the API serves the 5 non-LLM pipelines and lists the other 2 as unavailable.
5. **The $3 cap doesn't carry over to Cloud Run.** It's the call log in `data/llm_cache.sqlite` inside the container, so each instance starts at $0 and the log is lost on restart. The per-IP rate limit is also per instance. Before exposing LLM pipelines publicly, set a **Cloud Billing budget with alerts**, or a Vertex AI quota. Better still, keep LLM pipelines off the public service. They took ~50 s per call with frequent HTTP 500s in the evaluation, which is too slow for interactive use anyway.
6. **Decide on access.** That's unauthenticated or IAM-only invocations, and whether you need Cloud Armor in front if it's public.
7. **Create the cloud plumbing** (not done, as you asked): an Artifact Registry repository, Workload Identity Federation for GitHub Actions, and the Cloud Run service.
8. **Merge or rebase:** none of this has been pushed. The repo has no remote configured.
