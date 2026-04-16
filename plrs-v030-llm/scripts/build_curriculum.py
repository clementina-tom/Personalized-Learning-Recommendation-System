"""
scripts/build_curriculum.py
============================
Build a PLRS curriculum DAG from topic names or a text file using an LLM.

Usage:
    # From a list of topics
    python scripts/build_curriculum.py \\
        --topics "Whole Numbers" "Fractions" "Algebra" "Quadratic Equations" \\
        --domain "Secondary School Mathematics" \\
        --provider claude \\
        --output data/knowledge_maps/my_math_dag.json

    # From a text file (syllabus, course outline)
    python scripts/build_curriculum.py \\
        --text-file syllabus.txt \\
        --domain "Physics" \\
        --provider ollama --model llama3.2 \\
        --output data/knowledge_maps/physics_dag.json

    # Free local model (no API key)
    python scripts/build_curriculum.py \\
        --topics "Mechanics" "Kinematics" "Dynamics" "Energy" \\
        --provider ollama --model llama3.2

    # With validation pass (extra LLM call to fix issues)
    python scripts/build_curriculum.py \\
        --topics "A" "B" "C" \\
        --provider claude --validate

Provider options:
    claude       — Anthropic API (ANTHROPIC_API_KEY required)
    openai       — OpenAI API (OPENAI_API_KEY required)
    ollama       — Local Ollama (free, no key, needs Ollama running)
    huggingface  — HF Inference API (HF_TOKEN required)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a PLRS curriculum DAG using an LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--topics", nargs="+",
        metavar="TOPIC",
        help="List of topic names (quoted if they contain spaces)",
    )
    input_group.add_argument(
        "--text-file",
        metavar="PATH",
        help="Path to a text file containing syllabus or course description",
    )

    # Domain
    p.add_argument("--domain", default="Unknown Domain",
                   help="Curriculum domain name (default: 'Unknown Domain')")

    # Provider
    p.add_argument(
        "--provider",
        default="claude",
        choices=["claude", "openai", "ollama", "huggingface"],
        help="LLM provider (default: claude)",
    )
    p.add_argument("--model", default=None,
                   help="Model override (e.g. 'llama3.2' for ollama, 'gpt-4o' for openai)")
    p.add_argument("--api-key", default=None,
                   help="API key (falls back to env var if not set)")

    # Output
    p.add_argument("--output", "-o", default=None,
                   help="Output JSON path. Defaults to stdout if not set.")
    p.add_argument("--validate", action="store_true",
                   help="Run a validation pass (extra LLM call to fix issues)")

    # Options
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)

    return p.parse_args()


def build_provider(args: argparse.Namespace):
    """Construct the LLM provider from CLI args."""
    provider_key = args.provider
    model = args.model
    api_key = args.api_key

    if provider_key == "claude":
        from plrs.llm.providers.claude import ClaudeProvider
        kwargs = {}
        if model:   kwargs["model"]   = model
        if api_key: kwargs["api_key"] = api_key
        return ClaudeProvider(**kwargs)

    if provider_key == "openai":
        from plrs.llm.providers.openai import OpenAIProvider
        kwargs = {}
        if model:   kwargs["model"]   = model
        if api_key: kwargs["api_key"] = api_key
        return OpenAIProvider(**kwargs)

    if provider_key == "ollama":
        from plrs.llm.providers.ollama import OllamaProvider
        return OllamaProvider(model=model or "llama3.2")

    if provider_key == "huggingface":
        from plrs.llm.providers.huggingface import HuggingFaceProvider
        kwargs = {}
        if model:   kwargs["model"]   = model
        if api_key: kwargs["api_key"] = api_key
        return HuggingFaceProvider(**kwargs)

    raise ValueError(f"Unknown provider: {provider_key}")


def main() -> None:
    args = parse_args()

    from plrs.curriculum.llm_builder import CurriculumBuilder

    print(f"\n🧠 PLRS Curriculum Builder")
    print(f"   Provider : {args.provider}" + (f" ({args.model})" if args.model else ""))
    print(f"   Domain   : {args.domain}")
    print(f"   Validate : {args.validate}")
    print()

    try:
        provider = build_provider(args)
    except ImportError as e:
        print(f"❌ {e}")
        sys.exit(1)

    builder = CurriculumBuilder(
        provider=provider,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        validate=args.validate,
    )

    try:
        if args.topics:
            print(f"Building from {len(args.topics)} topics...")
            result = builder.from_topics(topics=args.topics, domain=args.domain)
        else:
            text_path = Path(args.text_file)
            if not text_path.exists():
                print(f"❌ File not found: {text_path}")
                sys.exit(1)
            text = text_path.read_text(encoding="utf-8")
            print(f"Building from text file: {text_path} ({len(text)} chars)...")
            result = builder.from_text(text=text, domain=args.domain)

    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

    # Report
    print(f"✅ Built: {result.num_nodes} nodes, {result.num_edges} edges")
    if result.warnings:
        print(f"⚠️  Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"   - {w}")

    # Output
    dag_json = json.dumps(result.dag, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(dag_json)
        print(f"\n💾 Saved to: {out_path}")
        print(f"\nLoad with:")
        print(f"  from plrs.curriculum import load_dag")
        print(f"  curriculum = load_dag('{out_path}')\n")
    else:
        print("\n" + dag_json)


if __name__ == "__main__":
    main()
