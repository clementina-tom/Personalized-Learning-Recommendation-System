"""
scripts/serve.py
================
Startup script — loads curricula and launches the full PLRS API.

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8080
    python scripts/serve.py --model checkpoints/sakt_decay_best.pt
    python scripts/serve.py --llm ollama --dev

Environment variables:
    PLRS_MODEL_PATH   — path to trained SAKTWithDecay .pt file
    PLRS_HOST         — bind host (default: 127.0.0.1)
    PLRS_PORT         — bind port (default: 8000)
    PLRS_DEV_MODE     — set to 1 to bypass API key auth
    PLRS_LLM          — default LLM provider (claude/openai/ollama/huggingface)
    PLRS_DB_URL       — PostgreSQL URL (omit for in-memory KeyStore)
    PLRS_REDIS_URL    — Redis URL (omit for in-memory rate limiter)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAPS = DATA / "knowledge_maps"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PLRS API server")
    p.add_argument("--host",           default=os.getenv("PLRS_HOST", "127.0.0.1"))
    p.add_argument("--port",           default=int(os.getenv("PLRS_PORT", 8000)), type=int)
    p.add_argument("--reload",         action="store_true", help="Hot reload (dev only)")
    p.add_argument("--dev",            action="store_true", help="Enable DEV_MODE (no auth)")
    p.add_argument("--model",          default=os.getenv("PLRS_MODEL_PATH"), help="SAKTWithDecay .pt path")
    p.add_argument("--llm",            default=os.getenv("PLRS_LLM"), help="Default LLM provider")
    p.add_argument("--threshold",      type=float, default=0.70)
    p.add_argument("--soft-threshold", type=float, default=0.50)
    p.add_argument("--workers",        type=int,   default=1)
    return p.parse_args()


def bootstrap(args: argparse.Namespace) -> None:
    """Load all modules and register pipelines, engines, and learners."""

    # ── DEV MODE ─────────────────────────────────────────────────────────────
    if args.dev:
        os.environ["PLRS_DEV_MODE"] = "1"
        print("  ⚠️  DEV MODE — auth disabled")

    # ── Imports (after env vars set) ──────────────────────────────────────────
    from plrs.api.app import register_pipeline
    from plrs.api.llm_router import llm_router, register_default_llm_provider
    from plrs.assess.diagnostic import DiagnosticEngine, register_engine
    from plrs.assess.router import assess_router
    from plrs.curriculum.loader import load_dag
    from plrs.model.online import OnlineLearner, register_learner
    from plrs.pipeline import PLRSPipeline

    # ── Main app ──────────────────────────────────────────────────────────────
    from plrs.api.app import app
    app.include_router(llm_router)
    app.include_router(assess_router)

    # ── Online learning router ────────────────────────────────────────────────
    _register_online_router(app)

    # ── LLM provider ─────────────────────────────────────────────────────────
    if args.llm:
        try:
            from plrs.llm.base import resolve_provider
            provider = resolve_provider(args.llm)
            register_default_llm_provider(provider)
            print(f"  ✅ LLM provider: {provider.provider_name}")
        except (ImportError, ValueError) as e:
            print(f"  ⚠️  LLM provider '{args.llm}' unavailable: {e}")

    # ── Curricula ─────────────────────────────────────────────────────────────
    curricula = {
        "math": MAPS / "math_dag.json",
        "cs":   MAPS / "cs_dag.json",
    }

    for domain, path in curricula.items():
        if not path.exists():
            print(f"  [WARN] Curriculum not found, skipping: {path}")
            continue

        curriculum = load_dag(path)

        # Pipeline
        pipeline = PLRSPipeline(
            curriculum=curriculum,
            model_path=args.model,
            threshold=args.threshold,
            soft_threshold=args.soft_threshold,
        )
        register_pipeline(domain, pipeline)

        # Assessment engine (cold start)
        engine = DiagnosticEngine(curriculum)
        register_engine(domain, engine)

        # Online learner
        learner = OnlineLearner(pipeline, mode="bkt")
        register_learner(domain, learner)

        model_status = "SAKTWithDecay loaded" if args.model else "mastery-dict mode"
        print(
            f"  ✅ [{domain}] {curriculum.domain} — "
            f"{curriculum.num_nodes} nodes · {curriculum.num_edges} edges · {model_status}"
        )


def _register_online_router(app) -> None:
    """Register the /online endpoints inline (small enough not to need a separate file)."""
    from typing import Any
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field

    online_router = APIRouter(prefix="/online", tags=["Online Learning"])

    class UpdateRequest(BaseModel):
        domain:     str   = Field(..., description="Curriculum domain: 'math' or 'cs'")
        student_id: str   = Field(..., description="Unique student identifier")
        topic_id:   str   = Field(..., description="Curriculum topic ID")
        correct:    bool  = Field(..., description="Whether the student answered correctly")
        skill_id:   int | None = Field(None, description="SAKT skill ID (for sakt mode)")
        time_ms:    int | None = Field(None, description="Response time in milliseconds")

    class RecommendRequest(BaseModel):
        domain:     str = Field(..., description="Curriculum domain")
        student_id: str = Field(..., description="Student identifier")
        top_n:      int = Field(5, ge=1, le=20)

    class SessionRequest(BaseModel):
        domain:     str = Field(..., description="Curriculum domain")
        student_id: str = Field(..., description="Student identifier")

    @online_router.post("/update")
    def online_update(req: UpdateRequest) -> dict[str, Any]:
        """
        Update a student's mastery after a single interaction.

        Call this after every question/exercise the student completes.
        Returns the updated mastery estimate and delta immediately.
        """
        from plrs.model.online import get_learner
        learner = get_learner(req.domain)
        if learner is None:
            raise HTTPException(
                status_code=404,
                detail=f"No online learner for domain '{req.domain}'.",
            )
        result = learner.update(
            student_id=req.student_id,
            topic_id=req.topic_id,
            correct=req.correct,
            skill_id=req.skill_id,
        )
        return result.to_dict()

    @online_router.post("/recommend")
    def online_recommend(req: RecommendRequest) -> dict[str, Any]:
        """Get recommendations using the student's current online mastery state."""
        from plrs.model.online import get_learner
        learner = get_learner(req.domain)
        if learner is None:
            raise HTTPException(status_code=404, detail=f"No online learner for domain '{req.domain}'.")
        return learner.recommend(req.student_id, top_n=req.top_n)

    @online_router.post("/session")
    def online_session(req: SessionRequest) -> dict[str, Any]:
        """Get a student's current session summary and mastery vector."""
        from plrs.model.online import get_learner
        learner = get_learner(req.domain)
        if learner is None:
            raise HTTPException(status_code=404, detail=f"No online learner for domain '{req.domain}'.")
        return learner.session_summary(req.student_id)

    app.include_router(online_router)


def main() -> None:
    args = parse_args()

    print("\n🧠 PLRS — Personalized Learning Recommendation System")
    print("=" * 58)
    bootstrap(args)
    print(f"\n  API:    http://{args.host}:{args.port}")
    print(f"  Docs:   http://{args.host}:{args.port}/docs")
    print(f"  Health: http://{args.host}:{args.port}/health")
    print("=" * 58 + "\n")

    uvicorn.run(
        "plrs.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
