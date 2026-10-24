"""The one wrapper for every LLM call: Gemini on Vertex AI through google-genai.

Every call goes through `LLMClient.generate`, which
- caches responses in SQLite, keyed by a hash of (model, system, schema, prompt);
- applies a timeout and retries transient errors;
- logs tokens, latency and cost per call, to the `shopsense.llm` logger and to a
  `calls` table in the same SQLite file;
- refuses new (uncached) calls once the logged spend passes the budget.

Settings come from .env (see .env.example): GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION,
model names and USD prices per 1M tokens. Auth is Application Default Credentials.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("shopsense.llm")

DEFAULT_CACHE_PATH = Path("data") / "llm_cache.sqlite"
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BUDGET_USD = 3.0


class LLMError(RuntimeError):
    """A call failed after all retries (timeout, server error or bad response)."""


class BudgetExceeded(RuntimeError):
    """The logged spend has reached the budget.

    Deliberately not an LLMError, so callers that fall back on LLM errors still stop.
    """


@dataclass(frozen=True)
class ModelConfig:
    """A model name and its USD price per 1M input and output tokens."""

    name: str
    price_input_per_m: float
    price_output_per_m: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost of one call at list price."""
        return (
            input_tokens * self.price_input_per_m + output_tokens * self.price_output_per_m
        ) / 1_000_000


@dataclass(frozen=True)
class BackendResponse:
    """What a backend returns for one call."""

    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    """One generate() result.

    `cost_usd` is the list-price cost even when cached (what the call costs uncached);
    `billed_usd` is 0 for cache hits.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    billed_usd: float
    latency_s: float
    cached: bool


# (model, prompt, system, response_schema, timeout_s) -> BackendResponse
Backend = Callable[[str, str, str | None, Mapping[str, Any] | None, float], BackendResponse]


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    """KEY=VALUE pairs from a .env file (if present), overridden by the environment."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    values.update({key: os.environ[key] for key in values if key in os.environ})
    return values


def model_config(env: Mapping[str, str], role: str) -> ModelConfig:
    """The model for a role ("intent" or "rerank").

    Reads GEMINI_<ROLE>_MODEL, <ROLE>_PRICE_INPUT_PER_M and <ROLE>_PRICE_OUTPUT_PER_M.
    """
    prefix = role.upper()
    return ModelConfig(
        name=env[f"GEMINI_{prefix}_MODEL"],
        price_input_per_m=float(env[f"{prefix}_PRICE_INPUT_PER_M"]),
        price_output_per_m=float(env[f"{prefix}_PRICE_OUTPUT_PER_M"]),
    )


def genai_backend(env: Mapping[str, str]) -> Backend:
    """A backend calling Gemini on Vertex AI (google-genai, vertexai=True, ADC auth).

    Thinking is turned off (thinking_budget=0): in the preflight the rerank model spent
    ~100 thinking tokens, billed as output, to answer "ok". Temperature is 0.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=env["GOOGLE_CLOUD_PROJECT"],
        location=env["GOOGLE_CLOUD_LOCATION"],
    )

    def call(model, prompt, system, schema, timeout_s):
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json" if schema else None,
            response_json_schema=schema,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        usage = response.usage_metadata
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return BackendResponse(response.text or "", usage.prompt_token_count or 0, output_tokens)

    return call


def cache_key(model: str, prompt: str, system: str | None, schema: Mapping | None) -> str:
    """sha256 of everything that determines the response."""
    payload = json.dumps([model, system, schema, prompt], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMClient:
    """Cached, logged, budgeted generate() calls to one model.

    Safe to share between threads: the SQLite connection is guarded by a lock and model
    calls run outside it. With several threads, spend can pass the budget by at most
    the calls already in flight.
    """

    def __init__(
        self,
        model: ModelConfig,
        backend: Backend,
        cache_path: Path = DEFAULT_CACHE_PATH,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        budget_usd: float = DEFAULT_BUDGET_USD,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.backend = backend
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.budget_usd = budget_usd
        self._sleep = sleep
        self._lock = threading.Lock()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY, model TEXT, text TEXT,
                input_tokens INTEGER, output_tokens INTEGER, latency_s REAL
            );
            CREATE TABLE IF NOT EXISTS calls (
                ts REAL, model TEXT, key TEXT, cached INTEGER, input_tokens INTEGER,
                output_tokens INTEGER, latency_s REAL, cost_usd REAL, billed_usd REAL
            );
            """
        )

    def spent_usd(self) -> float:
        """Total billed spend logged in this cache file, across all models and runs."""
        with self._lock:
            row = self._db.execute("SELECT COALESCE(SUM(billed_usd), 0) FROM calls").fetchone()
        return row[0]

    def usage_since(self, since_ts: float) -> dict[str, float]:
        """Calls, cache hits, list-price cost and billed spend logged since `since_ts`
        (a time.time() value), across all models sharing this cache file."""
        with self._lock:
            calls, cached, cost, billed = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(cached), 0), COALESCE(SUM(cost_usd), 0), "
                "COALESCE(SUM(billed_usd), 0) FROM calls WHERE ts >= ?",
                [since_ts],
            ).fetchone()
        return {"calls": calls, "cached": cached, "cost_usd": cost, "billed_usd": billed}

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Return the model's text for `prompt`, from cache if seen before.

        Raises BudgetExceeded before a new call once the logged spend reaches the budget,
        and LLMError if the call still fails after `max_retries` retries.
        """
        key = cache_key(self.model.name, prompt, system, response_schema)
        with self._lock:
            row = self._db.execute(
                "SELECT text, input_tokens, output_tokens FROM responses WHERE key = ?", [key]
            ).fetchone()
        if row is not None:
            text, input_tokens, output_tokens = row
            return self._log(key, text, input_tokens, output_tokens, latency_s=0.0, cached=True)

        spent = self.spent_usd()
        if spent >= self.budget_usd:
            raise BudgetExceeded(
                f"logged spend ${spent:.4f} reached the ${self.budget_usd:.2f} budget"
            )
        start = time.perf_counter()
        response = self._call_with_retries(prompt, system, response_schema)
        latency_s = time.perf_counter() - start
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?)",
                [key, self.model.name, response.text, response.input_tokens,
                 response.output_tokens, latency_s],
            )
        return self._log(
            key, response.text, response.input_tokens, response.output_tokens, latency_s, False
        )

    def _call_with_retries(
        self, prompt: str, system: str | None, schema: Mapping[str, Any] | None
    ) -> BackendResponse:
        for attempt in range(self.max_retries + 1):
            try:
                return self.backend(self.model.name, prompt, system, schema, self.timeout_s)
            except Exception as err:  # timeouts, 429/5xx, network: all retried
                if attempt == self.max_retries:
                    raise LLMError(f"{self.model.name}: {type(err).__name__}: {err}") from err
                name = type(err).__name__
                logger.warning("%s: %s, retry %d", self.model.name, name, attempt + 1)
                self._sleep(2**attempt)
        raise AssertionError("unreachable")

    def _log(
        self, key: str, text: str, input_tokens: int, output_tokens: int,
        latency_s: float, cached: bool,
    ) -> LLMResponse:
        cost = self.model.cost(input_tokens, output_tokens)
        billed = 0.0 if cached else cost
        with self._lock:
            self._db.execute(
                "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [time.time(), self.model.name, key, int(cached), input_tokens, output_tokens,
                 latency_s, cost, billed],
            )
            self._db.commit()
        logger.info(
            "%s cached=%s in=%d out=%d latency=%.2fs cost=$%.6f",
            self.model.name, cached, input_tokens, output_tokens, latency_s, cost,
        )
        return LLMResponse(text, input_tokens, output_tokens, cost, billed, latency_s, cached)


def make_client(role: str, env: Mapping[str, str] | None = None, **kwargs: Any) -> LLMClient:
    """An LLMClient for "intent" or "rerank" using .env settings and the Vertex AI backend."""
    env = load_env() if env is None else env
    return LLMClient(model_config(env, role), genai_backend(env), **kwargs)
