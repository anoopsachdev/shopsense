import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from shopsense import api
from shopsense.llm.client import BudgetExceeded, LLMError, LLMResponse
from shopsense.llm.query_intent import IntentBooster, QueryIntentExtractor
from shopsense.pipeline import PIPELINES, Pipeline
from shopsense.rerank.base import Reranker
from shopsense.rerank.llm import LLMReranker
from shopsense.retrieval.base import Hit
from shopsense.retrieval.bm25 import BM25Retriever

CATALOG = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3"],
        "product_title": ["Red running shoes", "Blue running shoes", "Coffee mug"],
        "product_brand": ["Acme", None, "Mugs Inc"],
        "product_color": ["red", "blue", None],
    }
)


class FakeLLM:
    """Answers intent and rerank prompts; can be switched to fail."""

    def __init__(self):
        self.error = None

    def generate(self, prompt, system=None, response_schema=None):
        if self.error:
            raise self.error
        if response_schema is None:  # the reranker asks for plain numbers
            text = "2,1,3"
        else:
            text = json.dumps({"product_type": "shoes", "brand": None, "color": "blue",
                               "attributes": [], "exclusions": ["mug"]})
        return LLMResponse(text, 10, 5, 0.0001, 0.0001, 0.01, False)


class KeepOrder(Reranker):
    def index(self, catalog):
        pass

    def rerank(self, query, hits):
        return list(hits)


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(llm):
    bm25 = BM25Retriever()
    llm_pipeline = Pipeline(
        bm25,
        LLMReranker(llm, fallback=KeepOrder()),
        rerank_depth=20,
        intent=IntentBooster(QueryIntentExtractor(llm)),
    )
    pipelines = {"bm25": Pipeline(bm25), api.DEFAULT_PIPELINE: Pipeline(bm25),
                 "intent+hybrid+llm": llm_pipeline}
    for pipeline in pipelines.values():
        pipeline.index(CATALOG)

    def load():
        return api.AppState(
            products=api._product_lookup(CATALOG),
            pipelines=pipelines,
            unavailable={"hybrid+llm": "LLM settings missing: GOOGLE_CLOUD_PROJECT"},
            limiter=api.RateLimiter(limit=2),
        )

    with TestClient(api.create_app(load)) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "catalog_products": 3,
        "pipelines": ["bm25", api.DEFAULT_PIPELINE, "intent+hybrid+llm"],
    }


def test_pipelines_lists_every_config_with_availability(client):
    rows = {row["name"]: row for row in client.get("/pipelines").json()}
    assert list(rows) == list(PIPELINES)
    assert rows["bm25"] == {
        "name": "bm25", "config": {"retriever": "bm25"}, "uses_llm": False,
        "available": True, "reason": None,
    }
    assert rows["hybrid+llm"]["available"] is False
    assert "GOOGLE_CLOUD_PROJECT" in rows["hybrid+llm"]["reason"]
    assert rows["intent+hybrid+llm"]["uses_llm"] is True
    assert rows["dense"]["available"] is False and rows["dense"]["reason"] is None


def test_search_defaults_to_the_fast_cross_encoder_pipeline(client):
    body = client.get("/search", params={"q": "red shoes"}).json()
    assert body["pipeline"] == "hybrid+cross_encoder_fast"
    assert body["k"] == 10
    assert body["intent"] is None and body["fell_back"] is False
    assert body["latency_ms"] >= 0
    first = body["results"][0]
    assert first == {
        "rank": 1, "product_id": "p1", "title": "Red running shoes", "brand": "Acme",
        "color": "red", "score": pytest.approx(first["score"]),
    }
    assert [r["rank"] for r in body["results"]] == list(range(1, len(body["results"]) + 1))


def test_search_respects_k(client):
    assert len(client.get("/search", params={"q": "shoes", "k": 1}).json()["results"]) == 1


@pytest.mark.parametrize(
    "params", [{}, {"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 51}, {"q": "x" * 201}]
)
def test_search_validates_parameters(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_unknown_pipeline_is_400(client):
    response = client.get("/search", params={"q": "x", "pipeline": "bm26"})
    assert response.status_code == 400 and "unknown pipeline" in response.json()["detail"]


def test_unavailable_pipeline_is_503_with_reason(client):
    response = client.get("/search", params={"q": "x", "pipeline": "hybrid+llm"})
    assert response.status_code == 503 and "GOOGLE_CLOUD_PROJECT" in response.json()["detail"]


def test_llm_search_returns_intent_and_llm_order(client):
    params = {"q": "blue shoes no mug", "pipeline": "intent+hybrid+llm"}
    body = client.get("/search", params=params).json()
    assert body["intent"] == {"product_type": "shoes", "brand": None, "color": "blue",
                              "attributes": [], "exclusions": ["mug"]}
    # The mug is dropped by the exclusion, blue shoes are boosted, the fake LLM swaps 1 and 2.
    assert [r["product_id"] for r in body["results"]] == ["p1", "p2"]
    assert body["fell_back"] is False


def test_llm_search_reports_fallback(client, llm):
    llm.error = LLMError("timeout")
    body = client.get("/search", params={"q": "shoes", "pipeline": "intent+hybrid+llm"}).json()
    assert body["fell_back"] is True
    assert body["intent"]["product_type"] == "shoes"  # neutral intent after the failure


def test_llm_budget_reached_is_503(client, llm):
    llm.error = BudgetExceeded("$3")
    response = client.get("/search", params={"q": "shoes", "pipeline": "intent+hybrid+llm"})
    assert response.status_code == 503 and "budget" in response.json()["detail"]


def test_llm_pipelines_are_rate_limited_per_ip(client):
    def llm_search(ip):
        return client.get(
            "/search",
            params={"q": "shoes", "pipeline": "intent+hybrid+llm"},
            headers={"X-Forwarded-For": f"{ip}, 10.0.0.1"},
        )

    assert [llm_search("1.1.1.1").status_code for _ in range(3)] == [200, 200, 429]
    assert llm_search("1.1.1.1").headers["retry-after"] == "60"
    assert llm_search("2.2.2.2").status_code == 200
    # Non-LLM pipelines are not limited.
    assert all(client.get("/search", params={"q": "x"}).status_code == 200 for _ in range(5))


def test_rate_limiter_sliding_window():
    now = [0.0]
    limiter = api.RateLimiter(limit=2, window_s=60, clock=lambda: now[0])
    assert [limiter.allow("a"), limiter.allow("a"), limiter.allow("a")] == [True, True, False]
    assert limiter.allow("b")
    now[0] = 60.0
    assert limiter.allow("a")


def test_llm_unavailable_reason():
    full = {key: "x" for key in api.LLM_ENV_KEYS}
    assert api.llm_unavailable_reason(full) is None
    missing_project = {**full, "GOOGLE_CLOUD_PROJECT": ""}
    assert "GOOGLE_CLOUD_PROJECT" in api.llm_unavailable_reason(missing_project)


def test_build_available_pipelines_skips_llm_when_clients_fail():
    built = []

    def build(names):
        built.append(names)
        if any("llm" in name for name in names):
            raise RuntimeError("no credentials")
        return {name: object() for name in names}

    pipelines, unavailable = api.build_available_pipelines(None, build)
    assert set(pipelines) == {n for n in PIPELINES if "llm" not in n}
    assert set(unavailable) == {"hybrid+llm", "intent+hybrid+llm"}
    assert "no credentials" in unavailable["hybrid+llm"]
    assert len(built) == 2


def test_build_available_pipelines_with_llm_disabled():
    pipelines, unavailable = api.build_available_pipelines(
        "settings missing", lambda names: {name: object() for name in names}
    )
    assert "hybrid+llm" not in pipelines and unavailable["hybrid+llm"] == "settings missing"


def test_product_lookup_turns_missing_values_into_none():
    lookup = api._product_lookup(CATALOG)
    assert lookup["p2"] == {"title": "Blue running shoes", "brand": None, "color": "blue"}
    assert lookup["p3"]["color"] is None


def test_load_state_never_embeds(tmp_path, monkeypatch):
    CATALOG.to_parquet(tmp_path / "catalog.parquet")
    monkeypatch.setattr(api, "load_env", lambda: {})
    monkeypatch.chdir(tmp_path)  # no data/dense vectors here
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        api.load_state(tmp_path)


class Peer:
    host = "9.9.9.9"


def test_client_ip_prefers_forwarded_for():
    class FakeRequest:
        def __init__(self, headers):
            self.headers = headers
            self.client = Peer()

    assert api.client_ip(FakeRequest({"x-forwarded-for": "1.2.3.4, 5.6.7.8"})) == "1.2.3.4"
    assert api.client_ip(FakeRequest({})) == "9.9.9.9"
