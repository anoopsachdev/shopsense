"""Download the ESCI shopping-queries data and keep only US-locale, Task 1 rows.

The parquet files in amazon-science/esci-data are stored in Git LFS. This script asks
GitHub's LFS batch API for a direct download URL and downloads the raw files into
data/raw/ (about 1.2 GB; resumable, checked against the LFS sha256). Raw files that are
already there are reused, so re-running with a new seed doesn't download again.

Output (in data/):
    raw/                 the original ESCI parquet files (can be deleted afterwards)
    examples.parquet     US rows with small_version == 1 (both splits)
    eval_query_ids.json  the sampled eval and distractor queries (see shopsense.data)
    products.parquet     US products that appear in those rows
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from shopsense import data

LFS_BATCH_URL = "https://github.com/amazon-science/esci-data.git/info/lfs/objects/batch"

CHUNK_BYTES = 1024 * 1024

# (file name, sha256 oid, size in bytes) from the Git LFS pointer files in
# amazon-science/esci-data/shopping_queries_dataset. Pinning them fixes the data version.
EXAMPLES_FILE = (
    "shopping_queries_dataset_examples.parquet",
    "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    51_286_808,
)
PRODUCTS_FILE = (
    "shopping_queries_dataset_products.parquet",
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


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download_lfs_object(
    lfs_object: tuple[str, str, int], raw_dir: Path, attempts: int = 20
) -> Path:
    """Download one LFS file into `raw_dir`, resuming after dropped connections.

    Each attempt asks for a fresh signed URL (they expire) and continues from the bytes
    already in the `.part` file. The finished file is checked against the LFS sha256.
    """
    name, oid, size = lfs_object
    path = raw_dir / name
    if path.exists() and path.stat().st_size == size:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    for attempt in range(1, attempts + 1):
        done = part.stat().st_size if part.exists() else 0
        if done == size:
            break
        try:
            request = urllib.request.Request(
                lfs_download_url(oid, size), headers={"Range": f"bytes={done}-"}
            )
            with urllib.request.urlopen(request, timeout=60) as resp, part.open("ab") as out:
                if resp.status != 206 and done > 0:
                    raise ConnectionError("server ignored the Range header")
                while chunk := resp.read(CHUNK_BYTES):
                    out.write(chunk)
                    done += len(chunk)
                    if done % (100 * CHUNK_BYTES) < CHUNK_BYTES:
                        print(f"{name}: {done / 1e6:,.0f} / {size / 1e6:,.0f} MB")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
            print(f"{name}: download interrupted ({err}), attempt {attempt}/{attempts}")
            time.sleep(min(2**attempt, 60))
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError(f"{name}: download incomplete after {attempts} attempts")
    if sha256_of(part) != oid:
        part.unlink()
        raise RuntimeError(f"{name}: sha256 mismatch, deleted the partial file")
    part.rename(path)
    return path


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
    products file with pyarrow 16.
    """
    ids = batch.column("product_id").to_pylist()
    locales = batch.column("product_locale").to_pylist()
    return pa.array(
        [pid in product_ids and locale == "us" for pid, locale in zip(ids, locales)]
    )


def write_filtered_products(
    source: pq.ParquetFile, product_ids: set[str], path: Path
) -> int:
    """Stream the products file into `path`, keeping only matching rows.

    The products file is a single ~1.1 GB row group (about 2 GB decompressed), so it is
    read as record batches and each filtered batch is written straight out, which keeps
    peak memory small. Writes to a temporary file and renames it at the end, so an
    interrupted run never leaves a truncated file. Returns the number of rows written.
    """
    tmp_path = path.with_suffix(".parquet.tmp")
    kept = 0
    with pq.ParquetWriter(tmp_path, source.schema_arrow, compression="zstd") as writer:
        for i, batch in enumerate(source.iter_batches(batch_size=20_000)):
            filtered = batch.filter(product_mask(batch, product_ids))
            writer.write_batch(filtered)
            kept += filtered.num_rows
            if i % 10 == 0:
                print(f"products: scanned {(i + 1) * 20_000:,} rows, kept {kept:,}")
    tmp_path.rename(path)
    return kept


def download(data_dir: Path, seed: int = data.DEFAULT_SEED) -> None:
    """Write examples, the query sample and the US Task 1 products into `data_dir`."""
    raw_dir = data_dir / "raw"
    examples_raw = download_lfs_object(EXAMPLES_FILE, raw_dir)
    products_raw = download_lfs_object(PRODUCTS_FILE, raw_dir)

    examples = filter_examples(pq.read_table(examples_raw))
    examples = examples.drop_columns([c for c in examples.column_names if c.startswith("__")])
    pq.write_table(examples, data_dir / "examples.parquet", compression="zstd")
    print(f"examples: {examples.num_rows:,} rows")

    con = data.connect(data_dir)
    sample = data.sample_queries(con, seed=seed)
    data.save_query_sample(sample, data_dir / data.EVAL_QUERY_IDS_FILE)
    print(f"query sample: {len(sample.eval_ids)} eval, {len(sample.distractor_ids)} distractors")

    product_ids = set(examples["product_id"].to_pylist())
    source = pq.ParquetFile(products_raw, buffer_size=8 * CHUNK_BYTES)
    kept = write_filtered_products(source, product_ids, data_dir / "products.parquet")
    print(f"products: {kept:,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data.DATA_DIR)
    parser.add_argument("--seed", type=int, default=data.DEFAULT_SEED)
    args = parser.parse_args()
    download(args.data_dir, args.seed)


if __name__ == "__main__":
    main()
