"""
plrs.api.app
============
FastAPI application exposing PLRS as a REST API.

Run locally:
    uvicorn plrs.api.app:app --reload

Endpoints:
    POST /recommend      — get recommendations from mastery scores
    POST /what-if        — simulate mastering a topic
    GET  /curriculum     — inspect the loaded curriculum
    GET  /health         — liveness check
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from plrs.curriculum.loader import CurriculumGraph
from plrs.pipeline import PLRSPipeline

app = FastAPI(
    title="PLRS API",
    description=(
        "Personalized Learning Recommendation System — "
        "constraint-aware recommendations powered by knowledge tracing."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Pipeline registry — multiple curricula can be loaded               #
# ------------------------------------------------------------------ #

_pipelines: dict[str, PLRSPipeline] = {}


def register_pipeline(domain: str, pipeline: PLRSPipeline) -> None:
    """Register a PLRSPipeline instance under a domain key."""
    _pipelines[domain] = pipeline


def get_pipeline(domain: str) -> PLRSPipeline:
    if domain not in _pipelines:
        available = list(_pipelines.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain}' not found. Available: {available}",
        )
    return _pipelines[domain]


# ------------------------------------------------------------------ #
# Request / Response models                                           #
# ------------------------------------------------------------------ #

class RecommendRequest(BaseModel):
    domain: str = Field(..., description="Curriculum domain key (e.g. 'math', 'cs')")
    mastery_scores: dict[str, float] = Field(
        ...,
        description="Topic ID → mastery probability [0.0, 1.0]",
        json_schema_extra={"example": {"algebra_basics": 0.85, "quadratic_equations": 0.42}},
    )
    top_n: int = Field(5, ge=1, le=20, description="Number of top recommendations")
    threshold: float = Field(0.70, ge=0.0, le=1.0)
    soft_threshold: float = Field(0.50, ge=0.0, le=1.0)


class WhatIfRequest(BaseModel):
    domain: str = Field(..., description="Curriculum domain key")
    topic_id: str = Field(..., description="Topic to simulate mastering")


class HealthResponse(BaseModel):
    status: str
    version: str
    loaded_domains: list[str]


# ------------------------------------------------------------------ #
# Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        loaded_domains=list(_pipelines.keys()),
    )


@app.get("/curriculum/{domain}", tags=["Curriculum"])
def get_curriculum(domain: str) -> dict[str, Any]:
    """Inspect a loaded curriculum's nodes and edges."""
    pipeline = get_pipeline(domain)
    c = pipeline.curriculum
    return {
        "domain": c.domain,
        "num_nodes": c.num_nodes,
        "num_edges": c.num_edges,
        "nodes": [
            {
                "id": node,
                "label": c.label(node),
                "level": c.level(node),
                "prerequisites": c.prerequisites(node),
                "successors": c.successors(node),
            }
            for node in c.nodes
        ],
    }


@app.post("/recommend", tags=["Recommendations"])
def recommend(req: RecommendRequest) -> dict[str, Any]:
    """
    Generate personalized topic recommendations.

    Send a student's mastery scores per topic; receive ranked recommendations
    classified as approved, challenging, or vetoed by the DAG constraint layer.
    """
    pipeline = get_pipeline(req.domain)

    # Apply per-request threshold overrides
    pipeline.threshold = req.threshold
    pipeline.soft_threshold = req.soft_threshold
    pipeline.top_n = req.top_n
    pipeline.constraint_layer = pipeline.constraint_layer  # already bound to curriculum

    results = pipeline.recommend_from_mastery(req.mastery_scores)
    return results


@app.post("/what-if", tags=["Recommendations"])
def what_if(req: WhatIfRequest) -> dict[str, Any]:
    """
    Simulate mastering a topic and see what it unlocks.

    Returns direct and transitive unlocks, plus blocking prerequisites.
    """
    pipeline = get_pipeline(req.domain)
    if req.topic_id not in pipeline.curriculum.nodes:
        raise HTTPException(
            status_code=404,
            detail=f"Topic '{req.topic_id}' not found in curriculum '{req.domain}'.",
        )
    return pipeline.what_if(req.topic_id)


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {
        "name": "PLRS API",
        "docs": "/docs",
        "health": "/health",
        "github": "https://github.com/clementina-tom/plrs",
    }
