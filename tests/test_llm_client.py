import logging
from types import SimpleNamespace

import pytest

from shopsense.llm import client as llm_client
from shopsense.llm.client import (
    BackendResponse,
    BudgetExceeded,
    LLMClient,
    LLMError,
    ModelConfig,
    cache_key,
    genai_backend,
    load_env,
    model_config,
)

MODEL = ModelConfig("fake-model", price_input_per_m=1.0, price_output_per_m=10.0)


class FakeBackend:
    """Answers "echo: <prompt>" with 100 input / 10 output tokens; can fail first N calls."""

    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []

    def __call__(self, model, prompt, system, schema, timeout_s):
        self.calls.append((model, prompt, system, schema, timeout_s))
        if self.failures:
            self.failures -= 1
            raise TimeoutError("slow")
        return BackendResponse(f"echo: {prompt}", input_tokens=100, output_tokens=10)


def make(tmp_path, backend, **kwargs):
    cache_path = tmp_path / "cache.sqlite"
    return LLMClient(MODEL, backend, cache_path=cache_path, sleep=lambda s: None, **kwargs)


def test_model_config_cost_per_million_tokens():
    # 100 * $1/M + 10 * $10/M = $0.0001 + $0.0001
    assert MODEL.cost(100, 10) == pytest.approx(0.0002)


def test_load_env_reads_file_and_environment_overrides(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("# comment\nA=1\nB = two\n\nC=x=y\n")
    monkeypatch.setenv("B", "override")
    assert load_env(path) == {"A": "1", "B": "override", "C": "x=y"}
    assert load_env(tmp_path / "missing") == {}


def test_model_config_from_env():
    env = {
        "GEMINI_RERANK_MODEL": "m",
        "RERANK_PRICE_INPUT_PER_M": "0.75",
        "RERANK_PRICE_OUTPUT_PER_M": "3.75",
    }
    assert model_config(env, "rerank") == ModelConfig("m", 0.75, 3.75)


def test_cache_key_depends_on_everything_that_shapes_the_answer():
    base = cache_key("m", "p", None, None)
    assert base == cache_key("m", "p", None, None)
    variants = [
        cache_key("m2", "p", None, None),
        cache_key("m", "p2", None, None),
        cache_key("m", "p", "sys", None),
        cache_key("m", "p", None, {"type": "object"}),
    ]
    assert len({base, *variants}) == 5


def test_generate_calls_backend_then_serves_from_cache(tmp_path):
    backend = FakeBackend()
    llm = make(tmp_path, backend, timeout_s=7)
    first = llm.generate("hi", system="be brief", response_schema={"type": "object"})
    second = llm.generate("hi", system="be brief", response_schema={"type": "object"})
    assert backend.calls == [("fake-model", "hi", "be brief", {"type": "object"}, 7)]
    assert first.text == second.text == "echo: hi"
    assert not first.cached and second.cached
    assert first.billed_usd == pytest.approx(0.0002) and second.billed_usd == 0.0
    assert second.cost_usd == pytest.approx(0.0002)  # list-price cost even when cached


def test_cache_persists_across_clients(tmp_path):
    backend = FakeBackend()
    make(tmp_path, backend).generate("hi")
    assert make(tmp_path, backend).generate("hi").cached
    assert len(backend.calls) == 1


def test_spent_usd_sums_only_billed_calls(tmp_path):
    llm = make(tmp_path, FakeBackend())
    llm.generate("a")
    llm.generate("a")
    llm.generate("b")
    assert llm.spent_usd() == pytest.approx(0.0004)


def test_retries_transient_errors(tmp_path):
    backend = FakeBackend(failures=2)
    assert make(tmp_path, backend, max_retries=2).generate("hi").text == "echo: hi"
    assert len(backend.calls) == 3


def test_raises_llm_error_after_retries(tmp_path):
    backend = FakeBackend(failures=5)
    with pytest.raises(LLMError, match="TimeoutError"):
        make(tmp_path, backend, max_retries=1).generate("hi")
    assert len(backend.calls) == 2


def test_budget_stops_new_calls_but_not_cache_hits(tmp_path):
    backend = FakeBackend()
    llm = make(tmp_path, backend, budget_usd=0.0003)
    llm.generate("a")
    llm.generate("b")  # spent 0.0002 < 0.0003, allowed; now 0.0004
    with pytest.raises(BudgetExceeded):
        llm.generate("c")
    assert llm.generate("a").cached
    assert len(backend.calls) == 2
    assert not issubclass(BudgetExceeded, LLMError)


def test_logs_tokens_latency_and_cost(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="shopsense.llm"):
        make(tmp_path, FakeBackend()).generate("hi")
    assert "fake-model cached=False in=100 out=10" in caplog.text
    assert "cost=$0.000200" in caplog.text


def test_genai_backend_uses_vertex_json_mode_and_no_thinking(monkeypatch):
    created, requests = [], []

    class FakeModels:
        def generate_content(self, model, contents, config):
            requests.append((model, contents, config))
            usage = SimpleNamespace(
                prompt_token_count=12, candidates_token_count=3, thoughts_token_count=2
            )
            return SimpleNamespace(text='{"ok": true}', usage_metadata=usage)

    class FakeGenaiClient:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.models = FakeModels()

    from google import genai

    monkeypatch.setattr(genai, "Client", FakeGenaiClient)
    backend = genai_backend({"GOOGLE_CLOUD_PROJECT": "proj", "GOOGLE_CLOUD_LOCATION": "global"})
    response = backend("m", "prompt", "sys", {"type": "object"}, 5.0)

    assert created == [{"vertexai": True, "project": "proj", "location": "global"}]
    model, contents, config = requests[0]
    assert (model, contents) == ("m", "prompt")
    assert config.system_instruction == "sys"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == {"type": "object"}
    assert config.thinking_config.thinking_budget == 0
    assert config.http_options.timeout == 5000
    # Thinking tokens are billed as output, so they are counted.
    assert response == BackendResponse('{"ok": true}', input_tokens=12, output_tokens=5)


def test_make_client_reads_roles_from_env(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_client, "genai_backend", lambda env: FakeBackend())
    env = {
        "GEMINI_INTENT_MODEL": "lite",
        "INTENT_PRICE_INPUT_PER_M": "0.3",
        "INTENT_PRICE_OUTPUT_PER_M": "2.5",
    }
    llm = llm_client.make_client("intent", env, cache_path=tmp_path / "c.sqlite")
    assert llm.model == ModelConfig("lite", 0.3, 2.5)


def test_client_can_be_shared_between_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    backend = FakeBackend()
    llm = make(tmp_path, backend)
    with ThreadPoolExecutor(8) as pool:
        texts = list(pool.map(lambda i: llm.generate(f"p{i % 20}").text, range(100)))
    assert texts == [f"echo: p{i % 20}" for i in range(100)]
    assert llm.spent_usd() == pytest.approx(0.0002 * len(backend.calls))


def test_usage_since_counts_calls_and_cache_hits(tmp_path):
    import time

    llm = make(tmp_path, FakeBackend())
    llm.generate("old")
    start = time.time()
    llm.generate("a")
    llm.generate("a")
    assert llm.usage_since(start) == {
        "calls": 2,
        "cached": 1,
        "cost_usd": pytest.approx(0.0004),
        "billed_usd": pytest.approx(0.0002),
    }
