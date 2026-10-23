import hashlib
import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


def test_write_filtered_products_streams_and_filters(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(PRODUCTS, source)
    out = tmp_path / "products.parquet"
    kept = download_data.write_filtered_products(pq.ParquetFile(source), {"p1", "p2"}, out)
    result = pq.read_table(out)
    assert kept == 2
    assert result["product_title"].to_pylist() == ["us one", "us two"]
    assert result.schema.remove_metadata() == PRODUCTS.schema.remove_metadata()
    assert not (tmp_path / "products.parquet.tmp").exists()


class FakeResponse:
    """Stands in for an HTTP response that serves `body` from byte `start`."""

    def __init__(self, body: bytes, start: int):
        self.status = 206 if start else 200
        self._data = body[start:]

    def read(self, n: int) -> bytes:
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_lfs(monkeypatch, body: bytes):
    """Serve `body` for any LFS download, honouring the Range header."""
    monkeypatch.setattr(download_data, "lfs_download_url", lambda oid, size: "https://lfs.test")

    def urlopen(request, timeout):
        start = int(request.headers["Range"].removeprefix("bytes=").rstrip("-"))
        return FakeResponse(body, start)

    monkeypatch.setattr(download_data.urllib.request, "urlopen", urlopen)


def lfs_object_for(body: bytes) -> tuple[str, str, int]:
    return ("file.parquet", hashlib.sha256(body).hexdigest(), len(body))


def test_sha256_of(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert download_data.sha256_of(path) == hashlib.sha256(b"abc").hexdigest()


def test_download_lfs_object_resumes_from_part_file(monkeypatch, tmp_path):
    body = b"0123456789"
    fake_lfs(monkeypatch, body)
    (tmp_path / "file.part").write_bytes(body[:4])
    path = download_data.download_lfs_object(lfs_object_for(body), tmp_path)
    assert path == tmp_path / "file.parquet"
    assert path.read_bytes() == body
    assert not (tmp_path / "file.part").exists()


def test_download_lfs_object_reuses_existing_file(monkeypatch, tmp_path):
    body = b"0123456789"
    (tmp_path / "file.parquet").write_bytes(body)
    monkeypatch.setattr(download_data, "lfs_download_url", None)  # would fail if called
    assert download_data.download_lfs_object(lfs_object_for(body), tmp_path).read_bytes() == body


def test_download_lfs_object_rejects_wrong_sha256(monkeypatch, tmp_path):
    fake_lfs(monkeypatch, b"corrupted!")
    name, _, size = lfs_object_for(b"0123456789")
    with pytest.raises(RuntimeError, match="sha256"):
        download_data.download_lfs_object((name, "0" * 64, size), tmp_path)
    assert not (tmp_path / "file.part").exists()
