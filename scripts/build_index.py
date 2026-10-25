"""Embed the search catalog once with bge-small and save the vectors to data/.

Evaluation loads these vectors instead of re-embedding. Rerun this after changing the
query sample (eval_query_ids.json), the embedding model or the product text fields.

It also writes data/catalog.parquet, the 48,862 catalog products on their own, which
is all the API (and its Docker image) needs besides the vectors.
`--catalog-only` writes just that file, without embedding.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from shopsense import data
from shopsense.retrieval.dense import DEFAULT_VECTORS_PATH, DenseRetriever, Encoder


def export_catalog(data_dir: Path) -> pd.DataFrame:
    """Build the catalog from the query sample and save it to data_dir/catalog.parquet."""
    con = data.connect(data_dir)
    sample = data.load_query_sample(data_dir / data.EVAL_QUERY_IDS_FILE)
    catalog = data.build_catalog(con, sample.catalog_query_ids)
    catalog.to_parquet(data_dir / data.CATALOG_FILE, index=False)
    return catalog


def build_dense_index(
    data_dir: Path, vectors_path: Path, encoder: Encoder | None = None
) -> float:
    """Write catalog.parquet, then embed the catalog and save the vectors.

    Replaces any saved vectors. Returns the seconds embedding took.
    """
    catalog = export_catalog(data_dir)
    vectors_path.unlink(missing_ok=True)
    print(f"embedding {len(catalog):,} products")
    start = time.perf_counter()
    # The one place allowed to embed: everything else only loads the saved vectors.
    dense = DenseRetriever(encoder=encoder, vectors_path=vectors_path, allow_build=True)
    dense.index(catalog)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data.DATA_DIR)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS_PATH)
    parser.add_argument(
        "--catalog-only", action="store_true", help="write catalog.parquet, don't embed"
    )
    args = parser.parse_args()
    if args.catalog_only:
        catalog = export_catalog(args.data_dir)
        print(f"wrote {len(catalog):,} products to {args.data_dir / data.CATALOG_FILE}")
        return
    seconds = build_dense_index(args.data_dir, args.vectors)
    print(f"embedding took {seconds:.0f}s; saved to {args.vectors}")


if __name__ == "__main__":
    main()
