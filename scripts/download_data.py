"""Download the ESCI shopping-queries data and keep only US-locale, Task 1 rows.

The parquet files in amazon-science/esci-data are stored in Git LFS. This script asks
GitHub's LFS batch API for a direct download URL, then reads the files over HTTP range
requests, so the full 1.1 GB products file never has to be written to disk.

Output (in data/):
    examples.parquet     US rows with small_version == 1 (both splits)
    eval_query_ids.json  the sampled eval and distractor queries (see shopsense.data)
    products.parquet     US products labelled for those queries (the catalog)

Only catalog products are kept, not every US product: the full US Task 1 product set is
several hundred MB, and the catalog is all the evaluation needs.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from shopsense import data

LFS_BATCH_URL = "https://github.com/amazon-science/esci-data.git/info/lfs/objects/batch"

# (sha256 oid, size in bytes) from the Git LFS pointer files in
# amazon-science/esci-data/shopping_queries_dataset. Pinning them fixes the data version.
EXAMPLES_FILE = (
    "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    51_286_808,
)
PRODUCTS_FILE = (
    "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
    1_108_857_465,
)


def fetch(request: str | urllib.request.Request, attempts: int = 5) -> bytes:
    """GET or POST `request` and return the body, retrying on connection errors."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                return resp.read()
        except (urllib.error.URLError, ConnectionError) as err:
            if attempt == attempts:
                raise
            print(f"request failed ({err}), retrying in {2 ** attempt}s")
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def lfs_download_url(oid: str, size: int) -> str:
    """Ask GitHub's LFS batch API for a short-lived direct download URL."""
    body = json.dumps(
        {
            "operation": "download",
            "transfer": ["basic"],
            "objects": [{"oid": oid, "size": size}],
        }
    ).encode()
    request = urllib.request.Request(
        LFS_BATCH_URL,
        data=body,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
    )
    payload = json.loads(fetch(request))
    return payload["objects"][0]["actions"]["download"]["href"]


def open_remote_parquet(lfs_object: tuple[str, int]) -> pq.ParquetFile:
    """Open an ESCI parquet file on GitHub LFS for streaming reads."""
    oid, size = lfs_object
    url = lfs_download_url(oid, size)
    handle = fsspec.filesystem("http").open(url, cache_type="none", size=size)
    return pq.ParquetFile(handle, buffer_size=8 * 1024 * 1024, pre_buffer=False)


def filter_examples(table: pa.Table) -> pa.Table:
    """Keep US-locale rows that belong to the Task 1 (small_version) subset."""
    mask = pc.and_(
        pc.equal(table["product_locale"], "us"),
        pc.equal(table["small_version"], 1),
    )
    return table.filter(mask)


def product_mask(batch: pa.RecordBatch, product_ids: set[str]) -> pa.Array:
    """Boolean mask of US-locale rows whose product_id is in `product_ids`.

    Built in Python rather than with pyarrow.compute.is_in, which crashed on the
    remote file with pyarrow 16.
    """
    ids = batch.column("product_id").to_pylist()
    locales = batch.column("product_locale").to_pylist()
    return pa.array(
        [pid in product_ids and locale == "us" for pid, locale in zip(ids, locales)]
    )


def read_filtered_products(remote: pq.ParquetFile, product_ids: set[str]) -> pa.Table:
    """Stream the remote products file in small batches, keeping only matching rows.

    The products file is a single ~1.1 GB row group, so it is read as a buffered stream
    of record batches rather than whole columns, which keeps peak memory small.
    """
    kept = []
    for i, batch in enumerate(remote.iter_batches(batch_size=20_000)):
        kept.append(batch.filter(product_mask(batch, product_ids)))
        if i % 10 == 0:
            print(f"products: scanned {(i + 1) * 20_000:,} rows")
    return pa.Table.from_batches(kept, schema=remote.schema_arrow)


def download(data_dir: Path, seed: int = data.DEFAULT_SEED) -> None:
    """Write examples, the query sample and the catalog products into `data_dir`."""
    data_dir.mkdir(parents=True, exist_ok=True)

    examples = filter_examples(open_remote_parquet(EXAMPLES_FILE).read())
    examples = examples.drop_columns([c for c in examples.column_names if c.startswith("__")])
    pq.write_table(examples, data_dir / "examples.parquet", compression="zstd")
    print(f"examples: {examples.num_rows:,} rows")

    products_path = data_dir / "products.parquet"
    products_path.unlink(missing_ok=True)  # a stale file would be picked up by connect()
    con = data.connect(data_dir)
    sample = data.sample_queries(con, seed=seed)
    data.save_query_sample(sample, data_dir / data.EVAL_QUERY_IDS_FILE)
    product_ids = set(data.labelled_product_ids(con, sample.catalog_query_ids))
    print(f"catalog: {len(sample.catalog_query_ids):,} queries, {len(product_ids):,} products")

    products = read_filtered_products(open_remote_parquet(PRODUCTS_FILE), product_ids)
    # Write then rename, so an interrupted run never leaves a truncated products file.
    tmp_path = products_path.with_suffix(".parquet.tmp")
    pq.write_table(products, tmp_path, compression="zstd")
    tmp_path.rename(products_path)
    print(f"products: {products.num_rows:,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data.DATA_DIR)
    parser.add_argument("--seed", type=int, default=data.DEFAULT_SEED)
    args = parser.parse_args()
    download(args.data_dir, args.seed)


if __name__ == "__main__":
    main()
