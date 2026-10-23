# ShopSense

LLM-assisted product search on Amazon's [ESCI shopping-queries dataset](https://github.com/amazon-science/esci-data),
scored against human relevance labels on quality, latency and cost per query.

Status: data loading, metrics and a BM25 baseline. Embeddings, re-ranking, LLM stages and the API are not built yet.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py   # US-locale, Task 1 (small_version) parquet into data/
```

## Commands

- Tests: `pytest -q`
- Evaluate: `python scripts/evaluate.py`
