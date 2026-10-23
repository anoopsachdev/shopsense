import importlib.util
from pathlib import Path

import pandas as pd

from shopsense import data
from shopsense.retrieval.dense import load_vectors
from tests.test_dense import FakeEncoder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_index.py"
spec = importlib.util.spec_from_file_location("build_index", SCRIPT)
build_index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_index)


def test_build_dense_index_embeds_only_the_catalog(tmp_path):
    examples = pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "query": ["a", "b", "c"],
            "product_id": ["p1", "p2", "p3"],
            "esci_label": ["E", "E", "E"],
            "split": ["test", "train", "train"],
        }
    )
    examples.to_parquet(tmp_path / "examples.parquet")
    products = pd.DataFrame(
        {"product_id": ["p1", "p2", "p3"], "product_title": ["shoes", "lamp", "coffee"]}
    )
    products.to_parquet(tmp_path / "products.parquet")
    data.save_query_sample(
        data.QuerySample(seed=0, eval_ids=[1], distractor_ids=[2]),
        tmp_path / data.EVAL_QUERY_IDS_FILE,
    )
    vectors_path = tmp_path / "vectors.npz"
    vectors_path.write_bytes(b"stale")  # replaced, not reused

    seconds = build_index.build_dense_index(tmp_path, vectors_path, encoder=FakeEncoder())

    ids, vectors = load_vectors(vectors_path)
    assert ids == ["p1", "p2"]
    assert vectors.shape == (2, 4)
    assert seconds >= 0
