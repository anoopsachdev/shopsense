import json

import pandas as pd
import pytest

from shopsense.llm.client import BudgetExceeded, LLMError, LLMResponse
from shopsense.rerank.base import Reranker
from shopsense.rerank.llm import (
    RANKING_SCHEMA,
    SYSTEM_PROMPT,
    LLMReranker,
    build_prompt,
    parse_ranking,
)
from shopsense.retrieval.base import Hit

CATALOG = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3"],
        "product_title": ["red shoes", "blue shoes", "coffee mug " + "x" * 400],
        "product_brand": ["Acme", None, None],
    }
)
HITS = [Hit("p1", 3.0), Hit("p2", 2.0), Hit("p3", 1.0)]


class FakeClient:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.requests = []

    def generate(self, prompt, system=None, response_schema=None):
        self.requests.append((prompt, system, response_schema))
        if self.error:
            raise self.error
        return LLMResponse(self.text, 500, 30, 0.002, 0.002, 1.0, False)


class ReverseFallback(Reranker):
    def __init__(self):
        self.indexed = False

    def index(self, catalog):
        self.indexed = True

    def rerank(self, query, hits):
        return [Hit(hit.product_id, -1.0) for hit in reversed(hits)]


def make(text=None, error=None):
    reranker = LLMReranker(FakeClient(text, error), fallback=ReverseFallback(), text_chars=40)
    reranker.index(CATALOG)
    return reranker


def test_build_prompt_numbers_products_from_one():
    assert build_prompt("shoes", ["a", "b"]) == "Query: shoes\n\nProducts:\n[1] a\n[2] b"


def test_parse_ranking_converts_to_zero_based():
    assert parse_ranking('{"ranking": [3, 1, 2]}', 3) == [2, 0, 1]


def test_parse_ranking_appends_left_out_products_in_input_order():
    assert parse_ranking('{"ranking": [3, 3, 9, 0, true]}', 4) == [2, 0, 1, 3]


@pytest.mark.parametrize("text", ["nope", "[1, 2]", '{"order": [1]}', '{"ranking": [7, 8]}'])
def test_parse_ranking_rejects_unusable_answers(text):
    with pytest.raises(ValueError):
        parse_ranking(text, 3)


def test_rerank_follows_the_model_and_scores_n_to_1():
    reranker = make(text=json.dumps({"ranking": [2, 3, 1]}))
    expected = [Hit("p2", 3.0), Hit("p3", 2.0), Hit("p1", 1.0)]
    assert reranker.rerank("blue shoes", HITS) == expected
    assert reranker.stats == {"calls": 1, "fallbacks": 0, "cost_usd": pytest.approx(0.002)}
    assert reranker.fallback_queries == []


def test_rerank_sends_shortened_texts_with_schema():
    reranker = make(text='{"ranking": [1, 2, 3]}')
    reranker.rerank("q", HITS)
    prompt, system, schema = reranker.client.requests[0]
    assert (system, schema) == (SYSTEM_PROMPT, RANKING_SCHEMA)
    assert "[1] red shoes Acme" in prompt
    assert prompt.endswith("[3] coffee mug " + "x" * 29)  # cut to text_chars=40


@pytest.mark.parametrize("client_kwargs", [{"error": LLMError("timeout")}, {"text": "garbage"}])
def test_rerank_falls_back_on_error_or_bad_answer(client_kwargs):
    reranker = make(**client_kwargs)
    assert [hit.product_id for hit in reranker.rerank("q", HITS)] == ["p3", "p2", "p1"]
    assert reranker.stats["fallbacks"] == 1
    assert reranker.fallback_queries == ["q"]


def test_budget_exceeded_is_not_a_fallback():
    with pytest.raises(BudgetExceeded):
        make(error=BudgetExceeded("$3")).rerank("q", HITS)


def test_index_also_indexes_the_fallback():
    assert make().fallback.indexed


def test_rerank_of_no_hits_makes_no_call():
    reranker = make(text="{}")
    assert reranker.rerank("q", []) == []
    assert reranker.client.requests == [] and reranker.stats["calls"] == 0


def test_reset_stats():
    reranker = make(error=LLMError("x"))
    reranker.rerank("q", HITS)
    reranker.reset_stats()
    assert reranker.stats == {"calls": 0, "fallbacks": 0, "cost_usd": 0.0}
    assert reranker.fallback_queries == []


def test_bad_answer_still_counts_its_cost():
    reranker = make(text="garbage")
    reranker.rerank("q", HITS)
    assert reranker.stats == {"calls": 1, "fallbacks": 1, "cost_usd": pytest.approx(0.002)}


def test_stats_are_consistent_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    reranker = make(text='{"ranking": [1, 2, 3]}')
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: reranker.rerank(f"q{i}", HITS), range(200)))
    assert reranker.stats["calls"] == 200
    assert reranker.stats["cost_usd"] == pytest.approx(0.4)
