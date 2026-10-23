import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
download_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_data)

PRODUCTS = pa.table(
    {
        "product_id": ["p1", "p2", "p1", "p3"],
        "product_title": ["us one", "us two", "jp one", "us three"],
        "product_locale": ["us", "us", "jp", "us"],
    }
)


def test_filter_examples_keeps_us_small_version_rows():
    table = pa.table(
        {
            "example_id": [1, 2, 3, 4],
            "product_locale": ["us", "us", "es", "us"],
            "small_version": [1, 0, 1, 1],
        }
    )
    assert download_data.filter_examples(table)["example_id"].to_pylist() == [1, 4]


def test_product_mask_requires_us_locale_and_wanted_id():
    batch = PRODUCTS.to_batches()[0]
    mask = download_data.product_mask(batch, {"p1", "p3"})
    assert mask.to_pylist() == [True, False, False, True]


def test_read_filtered_products_streams_and_filters(tmp_path):
    path = tmp_path / "products.parquet"
    pq.write_table(PRODUCTS, path)
    result = download_data.read_filtered_products(pq.ParquetFile(path), {"p1", "p2"})
    assert result["product_title"].to_pylist() == ["us one", "us two"]
    assert result.schema == PRODUCTS.schema
