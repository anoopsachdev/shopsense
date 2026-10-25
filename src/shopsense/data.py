"""Load ESCI into DuckDB, sample evaluation queries and build the product catalog.

Expected files in the data directory (written by scripts/download_data.py):
    examples.parquet   US-locale, Task 1 (small_version) query-product judgements
    products.parquet   US-locale product text for the catalog products
    eval_query_ids.json  the sampled eval and distractor query IDs

The catalog is every product labelled for the eval queries plus a set of distractor
queries, so each eval query has to find its products among ~2,500 queries' worth of
products rather than only its own ~20 candidates.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

DATA_DIR = Path("data")
EVAL_QUERY_IDS_FILE = "eval_query_ids.json"
CATALOG_FILE = "catalog.parquet"
DEFAULT_SEED = 42
NUM_EVAL_QUERIES = 500
NUM_DISTRACTOR_QUERIES = 2_000
NUM_LLM_EVAL_QUERIES = 200


@dataclass(frozen=True)
class QuerySample:
    """The query IDs that define one evaluation setup."""

    seed: int
    eval_ids: list[int]
    distractor_ids: list[int]

    @property
    def catalog_query_ids(self) -> list[int]:
        """All queries whose labelled products make up the catalog."""
        return self.eval_ids + self.distractor_ids


def connect(data_dir: Path = DATA_DIR) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB with `examples` (and `products`, if present) as views."""
    con = duckdb.connect()
    for name in ("examples", "products"):
        path = data_dir / f"{name}.parquet"
        if path.exists():
            # Views can't take prepared parameters, so the path is quoted inline.
            quoted = str(path).replace("'", "''")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{quoted}')")
    return con


def query_ids_in_split(con: duckdb.DuckDBPyConnection, split: str) -> list[int]:
    """Sorted distinct query IDs in a split ("train" or "test")."""
    rows = con.execute(
        "SELECT DISTINCT query_id FROM examples WHERE split = ? ORDER BY query_id",
        [split],
    ).fetchall()
    return [row[0] for row in rows]


def sample_queries(
    con: duckdb.DuckDBPyConnection,
    num_eval: int = NUM_EVAL_QUERIES,
    num_distractors: int = NUM_DISTRACTOR_QUERIES,
    seed: int = DEFAULT_SEED,
) -> QuerySample:
    """Sample eval queries from the test split and distractor queries from the train split.

    Sampling is over sorted IDs with a seeded RNG, so the same data and seed always give
    the same sample.
    """
    rng = random.Random(seed)
    eval_ids = sorted(rng.sample(query_ids_in_split(con, "test"), num_eval))
    distractor_ids = sorted(rng.sample(query_ids_in_split(con, "train"), num_distractors))
    return QuerySample(seed=seed, eval_ids=eval_ids, distractor_ids=distractor_ids)


def llm_eval_ids(sample: QuerySample, n: int = NUM_LLM_EVAL_QUERIES) -> list[int]:
    """The fixed subset of eval queries that LLM pipelines run on (cost is per query).

    Drawn from the eval IDs with the sample's own seed, so it never changes for a given
    eval_query_ids.json. Returns all eval IDs if there are n or fewer.
    """
    if len(sample.eval_ids) <= n:
        return list(sample.eval_ids)
    return sorted(random.Random(sample.seed).sample(sample.eval_ids, n))


def save_query_sample(sample: QuerySample, path: Path) -> None:
    """Write a query sample to JSON."""
    payload = {
        "seed": sample.seed,
        "eval_ids": sample.eval_ids,
        "distractor_ids": sample.distractor_ids,
    }
    path.write_text(json.dumps(payload, indent=1))


def load_query_sample(path: Path = DATA_DIR / EVAL_QUERY_IDS_FILE) -> QuerySample:
    """Read a query sample written by `save_query_sample`."""
    payload = json.loads(path.read_text())
    return QuerySample(
        seed=payload["seed"],
        eval_ids=payload["eval_ids"],
        distractor_ids=payload["distractor_ids"],
    )


def labelled_product_ids(con: duckdb.DuckDBPyConnection, query_ids: list[int]) -> list[str]:
    """Sorted distinct product IDs labelled for any of `query_ids`."""
    rows = con.execute(
        "SELECT DISTINCT product_id FROM examples WHERE list_contains(?, query_id) "
        "ORDER BY product_id",
        [query_ids],
    ).fetchall()
    return [row[0] for row in rows]


def build_catalog(con: duckdb.DuckDBPyConnection, query_ids: list[int]) -> pd.DataFrame:
    """Product rows for every product labelled for `query_ids`, sorted by product_id.

    Products that are labelled but missing from the products table are left out.
    """
    return con.execute(
        """
        SELECT p.*
        FROM products p
        JOIN (
            SELECT DISTINCT product_id FROM examples WHERE list_contains(?, query_id)
        ) labelled USING (product_id)
        ORDER BY p.product_id
        """,
        [query_ids],
    ).df()


def load_queries(con: duckdb.DuckDBPyConnection, query_ids: list[int]) -> dict[int, str]:
    """Map query_id to query text."""
    rows = con.execute(
        "SELECT DISTINCT query_id, query FROM examples WHERE list_contains(?, query_id)",
        [query_ids],
    ).fetchall()
    return dict(rows)


def load_labels(
    con: duckdb.DuckDBPyConnection, query_ids: list[int]
) -> dict[int, dict[str, str]]:
    """Map query_id to {product_id: ESCI label} for the labelled products of each query."""
    rows = con.execute(
        "SELECT query_id, product_id, esci_label FROM examples "
        "WHERE list_contains(?, query_id)",
        [query_ids],
    ).fetchall()
    labels: dict[int, dict[str, str]] = {qid: {} for qid in query_ids}
    for query_id, product_id, label in rows:
        labels[query_id][product_id] = label
    return labels
