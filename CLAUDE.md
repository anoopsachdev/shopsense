# ShopSense

LLM-assisted product search on Amazon's ESCI shopping-queries dataset, scored against human relevance labels on quality, latency and cost per query. Portfolio project for a software engineering internship; the code will be read and changed live in interviews, so clarity beats cleverness.

## Goal
Better top-10 relevance than BM25 within a fixed latency and cost budget. Every pipeline stage is swappable and scored by one evaluation harness that uses the same pipeline objects the API serves.

## Data
- ESCI Task 1 (small version), US locale only. Raw parquet lives in `data/` (gitignored) and in a Cloud Storage bucket.
- Eval set: 500 held-out test queries (200 of them for LLM pipelines). Catalog: all products labelled for ~2,500 queries. Fixed seed; query IDs saved to `data/eval_query_ids.json`.
- Gains for NDCG: headline is the official ESCI Task 1 gains from the repo's ranking scripts (Exact 1.0, Substitute 0.1, Complement 0.01, Irrelevant 0); also report graded gains (Exact 3, Substitute 2, Complement 1, Irrelevant 0). Every pipeline uses the same gains. Unlabelled products count as irrelevant.

## Layout
```
src/shopsense/
  data.py            load ESCI, sample queries, build catalog (DuckDB)
  retrieval/         base.py (Retriever interface), bm25.py, dense.py, hybrid.py (RRF)
  rerank/            base.py (Reranker interface), cross_encoder.py, llm.py
  llm/               client.py (one Gemini wrapper), query_intent.py, tagging.py
  pipeline.py        composes stages from config
  metrics.py         NDCG@k, Recall@k, macro-F1
  api.py             FastAPI app
scripts/             build_index.py, evaluate.py
web/                 single-page UI
tests/
```

## Stack
Python 3.12, DuckDB, rank-bm25, fastembed (ONNX, bge-small), numpy/FAISS, sentence-transformers cross-encoder (ms-marco MiniLM), Gemini on Vertex AI via `google-genai` (`vertexai=True`, auth by Application Default Credentials, no API key), SQLite cache, FastAPI + Pydantic, pytest, Docker, GitHub Actions with Workload Identity Federation, Artifact Registry, Cloud Run, Cloud Logging, BigQuery for eval run history.

## Rules
- Every pipeline stage implements its base interface; swapping a stage is a config change.
- Type hints and docstrings on public functions. Small functions, plain names.
- Every new function gets a pytest test. Metrics are tested against hand-computed examples.
- All LLM calls go through `llm/client.py`: cached by prompt hash, timeout, token and cost logged. Reranker falls back to the cross-encoder on timeout or error.
- Never commit secrets, data files or `.env`. Never hardcode project IDs.
- Run `pytest` before saying a task is done. Don't add features or dependencies I didn't ask for; propose them instead.
- Default to the cheapest model that works: Gemini Flash-Lite for query intent, Flash for re-ranking.

## Commands
Scripts need `PYTHONPATH=src`: macOS keeps hiding the editable install's `.pth` file in `.venv`, and Python 3.12 skips hidden `.pth` files. Pytest doesn't need it (`pythonpath = ["src"]` in pyproject.toml).
- Tests: `.venv/bin/pytest -q`
- Download data: `PYTHONPATH=src .venv/bin/python scripts/download_data.py`
- Build index: `PYTHONPATH=src .venv/bin/python scripts/build_index.py` (embeds the catalog, ~2 h on CPU)
- Evaluate: `PYTHONPATH=src .venv/bin/python scripts/evaluate.py --pipelines all`
- API: `PYTHONPATH=src .venv/bin/uvicorn shopsense.api:app --reload`
