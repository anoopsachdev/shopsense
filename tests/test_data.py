import pandas as pd
import pytest

from shopsense import data

# query_id -> (split, [(product_id, label), ...])
QUERIES = {
    1: ("test", [("p1", "E"), ("p2", "I")]),
    2: ("test", [("p2", "S"), ("p3", "E")]),
    3: ("test", [("p4", "C")]),
    10: ("train", [("p5", "E"), ("p1", "S")]),
    11: ("train", [("p6", "E")]),
    12: ("train", [("p7", "I")]),
}


@pytest.fixture
def data_dir(tmp_path):
    examples = pd.DataFrame(
        [
            {
                "query_id": qid,
                "query": f"query {qid}",
                "product_id": pid,
                "product_locale": "us",
                "esci_label": label,
                "small_version": 1,
                "split": split,
            }
            for qid, (split, products) in QUERIES.items()
            for pid, label in products
        ]
    )
    examples.to_parquet(tmp_path / "examples.parquet")
    # p7 is labelled but missing from the products table.
    products = pd.DataFrame(
        {
            "product_id": [f"p{i}" for i in range(1, 7)],
            "product_title": [f"title {i}" for i in range(1, 7)],
            "product_locale": "us",
        }
    )
    products.to_parquet(tmp_path / "products.parquet")
    return tmp_path


@pytest.fixture
def con(data_dir):
    return data.connect(data_dir)


def test_connect_creates_views(con):
    assert con.execute("SELECT count(*) FROM examples").fetchone()[0] == 9
    assert con.execute("SELECT count(*) FROM products").fetchone()[0] == 6


def test_connect_without_products_file(data_dir):
    (data_dir / "products.parquet").unlink()
    con = data.connect(data_dir)
    assert con.execute("SELECT count(*) FROM examples").fetchone()[0] == 9


def test_query_ids_in_split(con):
    assert data.query_ids_in_split(con, "test") == [1, 2, 3]
    assert data.query_ids_in_split(con, "train") == [10, 11, 12]


def test_sample_queries_draws_eval_from_test_and_distractors_from_train(con):
    sample = data.sample_queries(con, num_eval=2, num_distractors=2, seed=0)
    assert len(sample.eval_ids) == 2 and set(sample.eval_ids) <= {1, 2, 3}
    assert len(sample.distractor_ids) == 2 and set(sample.distractor_ids) <= {10, 11, 12}
    assert sample.catalog_query_ids == sample.eval_ids + sample.distractor_ids


def test_sample_queries_is_deterministic_for_a_seed(con):
    first = data.sample_queries(con, num_eval=2, num_distractors=2, seed=7)
    second = data.sample_queries(con, num_eval=2, num_distractors=2, seed=7)
    assert first == second


def test_query_sample_round_trips_through_json(tmp_path):
    sample = data.QuerySample(seed=3, eval_ids=[1, 2], distractor_ids=[10])
    path = tmp_path / "ids.json"
    data.save_query_sample(sample, path)
    assert data.load_query_sample(path) == sample


def test_labelled_product_ids(con):
    assert data.labelled_product_ids(con, [1, 10]) == ["p1", "p2", "p5"]


def test_build_catalog_has_each_labelled_product_once(con):
    catalog = data.build_catalog(con, [1, 2, 10])
    assert catalog["product_id"].tolist() == ["p1", "p2", "p3", "p5"]
    assert catalog["product_title"].tolist() == ["title 1", "title 2", "title 3", "title 5"]


def test_build_catalog_skips_products_missing_from_products_table(con):
    assert data.build_catalog(con, [12]).empty


def test_load_queries(con):
    assert data.load_queries(con, [1, 11]) == {1: "query 1", 11: "query 11"}


def test_load_labels(con):
    assert data.load_labels(con, [1, 3]) == {1: {"p1": "E", "p2": "I"}, 3: {"p4": "C"}}


def test_llm_eval_ids_is_a_fixed_subset_of_eval_ids():
    sample = data.QuerySample(seed=5, eval_ids=list(range(100, 150)), distractor_ids=[1])
    subset = data.llm_eval_ids(sample, n=10)
    assert len(subset) == 10 and set(subset) <= set(sample.eval_ids)
    assert subset == sorted(subset) == data.llm_eval_ids(sample, n=10)
    other_seed = data.QuerySample(seed=6, eval_ids=sample.eval_ids, distractor_ids=[1])
    assert data.llm_eval_ids(other_seed, n=10) != subset


def test_llm_eval_ids_returns_everything_when_small():
    sample = data.QuerySample(seed=5, eval_ids=[3, 1], distractor_ids=[])
    assert data.llm_eval_ids(sample, n=10) == [3, 1]
