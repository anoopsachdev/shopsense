"""FastAPI app: product search over the 48,862-product ESCI catalog.

    GET /search?q=red+shoes&pipeline=hybrid+cross_encoder_fast&k=10
    GET /pipelines
    GET /health

At startup it loads data/catalog.parquet and the saved dense vectors, builds every
pipeline with shared stages (one BM25 index, one cross-encoder, ...), and runs one
warm-up search per non-LLM pipeline so models are loaded before the first request.
It never embeds the catalog: missing vectors stop startup with a pointer to
scripts/build_index.py.

Paths are relative to the working directory: data/catalog.parquet and
data/dense_bge-small-en-v1.5.npz.

LLM pipelines need .env or the same environment variables (see .env.example) and
Application Default Credentials. Without them they are listed as unavailable. They
are rate-limited per client IP. Settings:
    SHOPSENSE_LLM_RATE_PER_MIN    LLM searches per IP per minute (default: 10)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from shopsense import data
from shopsense.llm.client import BudgetExceeded, load_env
from shopsense.llm.query_intent import QueryIntent
from shopsense.pipeline import PIPELINES, Pipeline, build_shared_pipelines

logger = logging.getLogger("shopsense.api")

DEFAULT_PIPELINE = "hybrid+cross_encoder_fast"
MAX_K = 50
LLM_ENV_KEYS = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GEMINI_RERANK_MODEL")


class Product(BaseModel):
    rank: int
    product_id: str
    title: str
    brand: str | None
    color: str | None
    score: float


class Intent(BaseModel):
    product_type: str
    brand: str | None
    color: str | None
    attributes: list[str]
    exclusions: list[str]


class SearchResponse(BaseModel):
    query: str
    pipeline: str
    k: int
    latency_ms: float
    fell_back: bool
    intent: Intent | None
    results: list[Product]


class PipelineInfo(BaseModel):
    name: str
    config: dict[str, Any]
    uses_llm: bool
    available: bool
    reason: str | None = None


class Health(BaseModel):
    status: str
    catalog_products: int
    pipelines: list[str]


class RateLimiter:
    """At most `limit` requests per `window_s` seconds per key (sliding window)."""

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._seen: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Record a request for `key` and say whether it is within the limit."""
        now = self._clock()
        with self._lock:
            seen = self._seen[key]
            while seen and seen[0] <= now - self.window_s:
                seen.popleft()
            if len(seen) >= self.limit:
                return False
            seen.append(now)
            return True


@dataclass
class AppState:
    """Everything the endpoints need, loaded once at startup."""

    products: dict[str, dict[str, Any]]
    pipelines: dict[str, Pipeline]
    unavailable: dict[str, str] = field(default_factory=dict)
    limiter: RateLimiter = field(default_factory=lambda: RateLimiter(10))


def llm_unavailable_reason(env: Mapping[str, str]) -> str | None:
    """Why LLM pipelines can't run with these settings, or None if they can."""
    missing = [key for key in LLM_ENV_KEYS if not env.get(key)]
    if missing:
        return f"LLM settings missing: {', '.join(missing)} (see .env.example)"
    return None


def load_state(data_dir: Path = data.DATA_DIR) -> AppState:
    """Load the catalog, build and index every available pipeline, and warm up models."""
    catalog = pd.read_parquet(data_dir / data.CATALOG_FILE)
    pipelines, unavailable = build_available_pipelines(llm_unavailable_reason(load_env()))
    for name, pipeline in pipelines.items():
        start = time.perf_counter()
        pipeline.index(catalog)
        if not pipeline.uses_llm:
            pipeline.search("warm up", k=1)  # loads the embedding and cross-encoder models
        logger.info("%s ready in %.1fs", name, time.perf_counter() - start)
    rate = int(os.environ.get("SHOPSENSE_LLM_RATE_PER_MIN", "10"))
    return AppState(
        products=_product_lookup(catalog),
        pipelines=pipelines,
        unavailable=unavailable,
        limiter=RateLimiter(rate),
    )


def build_available_pipelines(
    llm_reason: str | None,
    build: Callable[[list[str]], dict[str, Pipeline]] | None = None,
) -> tuple[dict[str, Pipeline], dict[str, str]]:
    """Every pipeline that can be built, and why the others can't.

    LLM pipelines are skipped when `llm_reason` says so, or when building their clients
    fails (e.g. no Application Default Credentials); the rest are built either way.
    """
    build = build or build_shared_pipelines
    llm_names = [name for name, config in PIPELINES.items() if _config_uses_llm(config)]
    plain_names = [name for name in PIPELINES if name not in llm_names]
    if llm_reason is None:
        try:
            return build(list(PIPELINES)), {}
        except Exception as err:  # client construction: credentials, bad settings
            llm_reason = f"could not create LLM clients: {type(err).__name__}: {err}"
            logger.warning(llm_reason)
    return build(plain_names), {name: llm_reason for name in llm_names}


def _config_uses_llm(config: Mapping[str, Any]) -> bool:
    return config.get("reranker") == "llm" or config.get("intent") is not None


def _product_lookup(catalog: pd.DataFrame) -> dict[str, dict[str, Any]]:
    columns = ["product_id", "product_title", "product_brand", "product_color"]
    rows = catalog.reindex(columns=columns).astype(object)
    rows = rows.where(rows.notna(), None)
    return {
        row["product_id"]: {
            "title": row["product_title"] or "",
            "brand": row["product_brand"],
            "color": row["product_color"],
        }
        for row in rows.to_dict("records")
    }


def client_ip(request: Request) -> str:
    """The caller's IP: the first X-Forwarded-For entry if present, else the peer.

    Cloud Run's proxy sets X-Forwarded-For; a client can also send it, so the limit is
    a guard against accidents and casual abuse, not a security boundary.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _intent_model(intent: QueryIntent | None) -> Intent | None:
    if intent is None:
        return None
    return Intent(
        product_type=intent.product_type,
        brand=intent.brand,
        color=intent.color,
        attributes=list(intent.attributes),
        exclusions=list(intent.exclusions),
    )


def create_app(load: Callable[[], AppState] = load_state) -> FastAPI:
    """The app. `load` builds the state at startup (tests pass a fake)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.shop = load()
        yield

    app = FastAPI(title="ShopSense", lifespan=lifespan)

    def state(request: Request) -> AppState:
        return request.app.state.shop

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        shop = state(request)
        return Health(
            status="ok", catalog_products=len(shop.products), pipelines=list(shop.pipelines)
        )

    @app.get("/pipelines", response_model=list[PipelineInfo])
    def pipelines(request: Request) -> list[PipelineInfo]:
        shop = state(request)
        return [
            PipelineInfo(
                name=name,
                config=config,
                uses_llm=_config_uses_llm(config),
                available=name in shop.pipelines,
                reason=shop.unavailable.get(name),
            )
            for name, config in PIPELINES.items()
        ]

    @app.get("/search", response_model=SearchResponse)
    def search(  # sync: FastAPI runs it in a worker thread, so LLM calls don't block
        request: Request,
        q: str = Query(..., min_length=1, max_length=200, description="search query"),
        pipeline: str = Query(DEFAULT_PIPELINE, description="a name from /pipelines"),
        k: int = Query(10, ge=1, le=MAX_K, description="number of results"),
    ) -> SearchResponse:
        shop = state(request)
        if pipeline in shop.unavailable:
            raise HTTPException(503, f"{pipeline} is unavailable: {shop.unavailable[pipeline]}")
        if pipeline not in shop.pipelines:
            raise HTTPException(400, f"unknown pipeline {pipeline!r}; see /pipelines")
        chosen = shop.pipelines[pipeline]
        if chosen.uses_llm and not shop.limiter.allow(client_ip(request)):
            raise HTTPException(
                429, "too many LLM searches; try again in a minute", {"Retry-After": "60"}
            )
        start = time.perf_counter()
        try:
            details = chosen.search_details(q, k)
        except BudgetExceeded as err:
            raise HTTPException(503, f"LLM budget reached: {err}") from err
        latency_ms = 1000 * (time.perf_counter() - start)
        unknown = {"title": "", "brand": None, "color": None}
        results = [
            Product(
                rank=rank,
                product_id=hit.product_id,
                score=hit.score,
                **shop.products.get(hit.product_id, unknown),
            )
            for rank, hit in enumerate(details.hits, start=1)
        ]
        return SearchResponse(
            query=q,
            pipeline=pipeline,
            k=k,
            latency_ms=round(latency_ms, 1),
            fell_back=details.fell_back,
            intent=_intent_model(details.intent),
            results=results,
        )

    return app


app = create_app()
