"""
plrs.api.app
============
FastAPI application exposing PLRS as a REST API.

Authentication: API key via X-API-Key header.
Rate limiting:  Sliding window per key (per-minute + per-day).

Public endpoints (no auth required):
    GET  /          — root
    GET  /health    — liveness check
    GET  /docs      — OpenAPI docs

Protected endpoints (require X-API-Key):
    GET  /curriculum/{domain}
    POST /recommend
    POST /what-if
    GET  /usage     — caller's rate limit usage

Admin endpoints (require internal tier key):
    GET    /admin/keys          — list all keys
    POST   /admin/keys          — create a key
    DELETE /admin/keys/{key}    — revoke a key

Dev mode: set PLRS_DEV_MODE=1 to bypass auth (local development).
"""

from __future__ import annotations

import os
import time
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from plrs.api.auth import APIKey, KeyStore, get_key_store
from plrs.api.rate_limit import RateLimiter, get_limiter
from plrs.pipeline import PLRSPipeline

DEV_MODE = os.getenv("PLRS_DEV_MODE", "0") == "1"

app = FastAPI(
    title="PLRS API",
    description=(
        "Personalized Learning Recommendation System — "
        "constraint-aware recommendations powered by knowledge tracing.\n\n"
        "**Authentication:** `X-API-Key: plrs_<your_key>` header.\n\n"
        "**Rate limits:** 60 req/min, 1000 req/day (standard tier)."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
    expose_headers=[
        "X-RateLimit-Limit-Minute", "X-RateLimit-Limit-Day",
        "X-RateLimit-Remaining-Minute", "X-RateLimit-Remaining-Day",
        "Retry-After",
    ],
)

# ── Pipeline registry ─────────────────────────────────────────────────────────

_pipelines: dict[str, PLRSPipeline] = {}


def register_pipeline(domain: str, pipeline: PLRSPipeline) -> None:
    _pipelines[domain] = pipeline


def get_pipeline(domain: str) -> PLRSPipeline:
    if domain not in _pipelines:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain}' not found. Available: {list(_pipelines.keys())}",
        )
    return _pipelines[domain]


# ── Auth + rate limit dependency ──────────────────────────────────────────────

async def authenticate(
    response: Response,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    key_store: KeyStore = Depends(get_key_store),
    limiter: RateLimiter = Depends(get_limiter),
) -> APIKey:
    if DEV_MODE:
        return APIKey(key_id="plrs_dev", name="dev", tier="internal", created_at=time.time())

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    try:
        api_key = key_store.validate(x_api_key)
    except KeyError:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    result = limiter.check(
        key_id=x_api_key,
        requests_per_minute=api_key.requests_per_minute,
        requests_per_day=api_key.requests_per_day,
    )

    for header, value in result.headers.items():
        response.headers[header] = value

    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail=result.to_dict(),
            headers={"Retry-After": str(result.retry_after)},
        )

    return api_key


AuthDep = Annotated[APIKey, Depends(authenticate)]


async def require_internal(api_key: AuthDep) -> APIKey:
    if not DEV_MODE and api_key.tier != "internal":
        raise HTTPException(status_code=403, detail="Admin endpoints require internal tier key.")
    return api_key


InternalDep = Annotated[APIKey, Depends(require_internal)]


# ── Pydantic models ───────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    domain: str = Field(..., description="Curriculum domain key (e.g. 'math', 'cs')")
    mastery_scores: dict[str, float] = Field(
        ...,
        description="Topic ID → mastery probability [0.0, 1.0]",
        json_schema_extra={"example": {"algebra_basics": 0.85, "quadratic_equations": 0.42}},
    )
    top_n: int            = Field(5,    ge=1,   le=20)
    threshold: float      = Field(0.70, ge=0.0, le=1.0)
    soft_threshold: float = Field(0.50, ge=0.0, le=1.0)


class WhatIfRequest(BaseModel):
    domain:   str = Field(..., description="Curriculum domain key")
    topic_id: str = Field(..., description="Topic to simulate mastering")


class HealthResponse(BaseModel):
    status:         str
    version:        str
    loaded_domains: list[str]
    auth_enabled:   bool


class CreateKeyRequest(BaseModel):
    name:     str = Field(..., description="Human label for this key")
    tier:     str = Field("standard", description="free | standard | premium | internal")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Public endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {
        "name":   "PLRS API",
        "docs":   "/docs",
        "health": "/health",
        "github": "https://github.com/clementina-tom/plrs",
        "auth":   "disabled (dev mode)" if DEV_MODE else "required (X-API-Key header)",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check — no authentication required."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        loaded_domains=list(_pipelines.keys()),
        auth_enabled=not DEV_MODE,
    )


# ── Protected endpoints ───────────────────────────────────────────────────────

@app.get("/curriculum/{domain}", tags=["Curriculum"])
def get_curriculum(domain: str, api_key: AuthDep) -> dict[str, Any]:
    """Inspect a curriculum's nodes and edges."""
    pipeline = get_pipeline(domain)
    c = pipeline.curriculum
    return {
        "domain":    c.domain,
        "num_nodes": c.num_nodes,
        "num_edges": c.num_edges,
        "nodes": [
            {
                "id":            node,
                "label":         c.label(node),
                "level":         c.level(node),
                "prerequisites": c.prerequisites(node),
                "successors":    c.successors(node),
            }
            for node in c.nodes
        ],
    }


@app.post("/recommend", tags=["Recommendations"])
def recommend(req: RecommendRequest, api_key: AuthDep) -> dict[str, Any]:
    """Generate personalized topic recommendations."""
    pipeline = get_pipeline(req.domain)
    pipeline.threshold      = req.threshold
    pipeline.soft_threshold = req.soft_threshold
    pipeline.top_n          = req.top_n
    return pipeline.recommend_from_mastery(req.mastery_scores)


@app.post("/what-if", tags=["Recommendations"])
def what_if(req: WhatIfRequest, api_key: AuthDep) -> dict[str, Any]:
    """Simulate mastering a topic and see what it unlocks."""
    pipeline = get_pipeline(req.domain)
    if req.topic_id not in pipeline.curriculum.nodes:
        raise HTTPException(
            status_code=404,
            detail=f"Topic '{req.topic_id}' not found in domain '{req.domain}'.",
        )
    return pipeline.what_if(req.topic_id)


@app.get("/usage", tags=["Account"])
def usage(
    api_key: AuthDep,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    limiter: RateLimiter = Depends(get_limiter),
) -> dict[str, Any]:
    """Return current rate limit usage for the caller's key."""
    key_id = "plrs_dev" if DEV_MODE else (x_api_key or "")
    stats  = limiter.stats(key_id)
    return {
        "key_name":         api_key.name,
        "tier":             api_key.tier,
        "requests_minute":  stats["requests_minute"],
        "requests_day":     stats["requests_day"],
        "limit_minute":     api_key.requests_per_minute,
        "limit_day":        api_key.requests_per_day,
        "remaining_minute": max(0, api_key.requests_per_minute - stats["requests_minute"]),
        "remaining_day":    max(0, api_key.requests_per_day    - stats["requests_day"]),
    }


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.get("/admin/keys", tags=["Admin"])
def list_keys(api_key: InternalDep, key_store: KeyStore = Depends(get_key_store)) -> dict:
    """List all API keys (internal tier only)."""
    return {"keys": key_store.list_keys(), "total": len(key_store)}


@app.post("/admin/keys", tags=["Admin"], status_code=201)
def create_key(
    req: CreateKeyRequest,
    api_key: InternalDep,
    key_store: KeyStore = Depends(get_key_store),
) -> dict:
    """Create a new API key. Raw key shown once — store it safely."""
    try:
        raw_key = key_store.create_key(name=req.name, tier=req.tier, metadata=req.metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    created = key_store.validate(raw_key)
    return {
        "key":     raw_key,
        "name":    created.name,
        "tier":    created.tier,
        "limits":  created.limits,
        "warning": "Store this key safely — it will not be shown again.",
    }


@app.delete("/admin/keys/{raw_key}", tags=["Admin"])
def revoke_key(
    raw_key: str,
    api_key: InternalDep,
    key_store: KeyStore = Depends(get_key_store),
    limiter: RateLimiter = Depends(get_limiter),
) -> dict:
    """Revoke an API key (internal tier only)."""
    try:
        key_store.revoke(raw_key)
        limiter.reset(raw_key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Key not found.")
    return {"status": "revoked", "key_prefix": raw_key[:12] + "..."}
