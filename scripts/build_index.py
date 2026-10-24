"""Embed the search catalog once with bge-small and save the vectors to data/.

Evaluation loads these vectors instead of re-embedding. Rerun this after changing the
query sample (eval_query_ids.json), the embedding model or the product text fields.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from shopsense import data
from shopsense.retrieval.dense import DEFAULT_VECTORS_PATH, DenseRetriever, Encoder


def build_dense_index(
    data_dir: Path, vectors_path: Path, encoder: Encoder | None = None
) -> float:
    """Embed the catalog and save the vectors, replacing any saved ones.

    Returns the seconds embedding took.
    """
    con = data.connect(data_dir)
    sample = data.load_query_sample(data_dir / data.EVAL_QUERY_IDS_FILE)
    catalog = data.build_catalog(con, sample.catalog_query_ids)
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
    args = parser.parse_args()
    seconds = build_dense_index(args.data_dir, args.vectors)
    print(f"embedding took {seconds:.0f}s; saved to {args.vectors}")


if __name__ == "__main__":
    main()
