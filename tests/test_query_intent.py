import json

import pytest

from shopsense.llm.client import BudgetExceeded, LLMError, LLMResponse
from shopsense.llm.query_intent import (
    INTENT_SCHEMA,
    SYSTEM_PROMPT,
    QueryIntent,
    QueryIntentExtractor,
    neutral_intent,
    parse_intent,
)


def answer(product_type, brand=None, color=None, attributes=(), exclusions=()):
    return json.dumps(
        {
            "product_type": product_type,
            "brand": brand,
            "color": color,
            "attributes": list(attributes),
            "exclusions": list(exclusions),
        }
    )


# Five hand-written queries and the answers a well-behaved intent model would give.
CANNED = {
    "Query: nike running shoes men size 10": answer(
        "Running Shoes", brand="Nike", attributes=["men", "size 10"]
    ),
    "Query: red dress without sleeves": answer("dress", color="red", exclusions=["sleeves"]),
    "Query: sugar free gum": answer("gum", attributes=["sugar free"]),
    "Query: iphone 14 case not otterbox": answer(
        "phone case", attributes=["iphone 14"], exclusions=["otterbox"]
    ),
    "Query: blue": answer("", color="blue"),
}

EXPECTED = {
    "nike running shoes men size 10": QueryIntent(
        "running shoes", brand="nike", attributes=("men", "size 10")
    ),
    "red dress without sleeves": QueryIntent("dress", color="red", exclusions=("sleeves",)),
    "sugar free gum": QueryIntent("gum", attributes=("sugar free",)),
    "iphone 14 case not otterbox": QueryIntent(
        "phone case", attributes=("iphone 14",), exclusions=("otterbox",)
    ),
    # An empty product_type falls back to the query text.
    "blue": QueryIntent("blue", color="blue"),
}


class FakeClient:
    """Stands in for LLMClient: canned text by prompt, or an exception."""

    def __init__(self, answers=CANNED, error=None):
        self.answers = answers
        self.error = error
        self.requests = []

    def generate(self, prompt, system=None, response_schema=None):
        self.requests.append((prompt, system, response_schema))
        if self.error:
            raise self.error
        return LLMResponse(self.answers[prompt], 50, 20, 0.001, 0.001, 0.1, False)


@pytest.mark.parametrize("query", list(EXPECTED))
def test_extract_hand_written_queries(query):
    client = FakeClient()
    assert QueryIntentExtractor(client).extract(query) == EXPECTED[query]
    assert client.requests == [(f"Query: {query}", SYSTEM_PROMPT, INTENT_SCHEMA)]


def test_parse_intent_lowercases_and_dedupes():
    text = answer(" Shoes ", brand="  ", attributes=["Wide", "wide", ""])
    assert parse_intent(text, "q") == QueryIntent("shoes", brand=None, attributes=("wide",))


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[1, 2]",
        json.dumps({"product_type": "x"}),
        json.dumps(
            {"product_type": "x", "brand": None, "color": None, "attributes": "wide",
             "exclusions": []}
        ),
        answer("x", brand=7),
    ],
)
def test_parse_intent_rejects_answers_that_break_the_schema(text):
    with pytest.raises(ValueError):
        parse_intent(text, "q")


def test_bad_answer_falls_back_to_neutral_intent_and_counts_failure():
    extractor = QueryIntentExtractor(FakeClient(answers={"Query: Red Shoes": "oops"}))
    assert extractor.extract("Red Shoes") == neutral_intent("Red Shoes")
    assert neutral_intent("Red Shoes") == QueryIntent("red shoes")
    assert extractor.stats == {"calls": 1, "failures": 1, "cost_usd": pytest.approx(0.001)}


def test_llm_error_falls_back_to_neutral_intent():
    extractor = QueryIntentExtractor(FakeClient(error=LLMError("timeout")))
    assert extractor.extract("lamp") == QueryIntent("lamp")
    assert extractor.stats["failures"] == 1


def test_budget_exceeded_is_not_swallowed():
    with pytest.raises(BudgetExceeded):
        QueryIntentExtractor(FakeClient(error=BudgetExceeded("$3"))).extract("lamp")


def test_stats_accumulate_cost_and_reset():
    extractor = QueryIntentExtractor(FakeClient())
    extractor.extract("blue")
    extractor.extract("sugar free gum")
    assert extractor.stats == {"calls": 2, "failures": 0, "cost_usd": pytest.approx(0.002)}
    extractor.reset_stats()
    assert extractor.stats == {"calls": 0, "failures": 0, "cost_usd": 0.0}
