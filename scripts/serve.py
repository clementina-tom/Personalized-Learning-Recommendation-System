"""
scripts/serve.py
================
Startup script — loads Nigerian curriculum DAGs and launches the PLRS API.

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8080 --reload
    python scripts/serve.py --model path/to/sakt_model.pt

Environment variables:
    PLRS_MODEL_PATH   — path to trained SAKT .pt file
    PLRS_HOST         — bind host (default: 127.0.0.1)
    PLRS_PORT         — bind port (default: 8000)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

# ── Resolve paths ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAPS = DATA / "knowledge_maps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLRS API server")
    parser.add_argument("--host",  default=os.getenv("PLRS_HOST", "127.0.0.1"))
    parser.add_argument("--port",  default=int(os.getenv("PLRS_PORT", 8000)), type=int)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument("--model", default=os.getenv("PLRS_MODEL_PATH"), help="Path to SAKT .pt file")
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Mastery threshold (default 0.70)"
    )
    parser.add_argument(
        "--soft-threshold", type=float, default=0.50,
        help="Soft constraint threshold (default 0.50)"
    )
    return parser.parse_args()


def bootstrap(args: argparse.Namespace) -> None:
    """Load curricula and register pipelines before server starts."""
    from plrs.api.app import register_pipeline
    from plrs.curriculum.loader import load_dag
    from plrs.pipeline import PLRSPipeline

    curricula = {
        "math": MAPS / "math_dag.json",
        "cs":   MAPS / "cs_dag.json",
    }

    model_path = args.model

    for domain, path in curricula.items():
        if not path.exists():
            print(f"  [WARN] Curriculum file not found, skipping: {path}")
            continue

        curriculum = load_dag(path)
        pipeline = PLRSPipeline(
            curriculum=curriculum,
            model_path=model_path,
            threshold=args.threshold,
            soft_threshold=args.soft_threshold,
        )
        register_pipeline(domain, pipeline)

        print(
            f"  ✅ [{domain}] {curriculum.domain} — "
            f"{curriculum.num_nodes} nodes, {curriculum.num_edges} edges"
            + (f"  + SAKT model loaded" if model_path else "  (mastery-dict mode)")
        )


def main() -> None:
    args = parse_args()

    print("\n🧠 PLRS — Personalized Learning Recommendation System")
    print("=" * 55)
    print("Loading curricula...")
    bootstrap(args)
    print(f"\nStarting API at http://{args.host}:{args.port}")
    print(f"  Docs:   http://{args.host}:{args.port}/docs")
    print(f"  Health: http://{args.host}:{args.port}/health")
    print("=" * 55 + "\n")

    uvicorn.run(
        "plrs.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
