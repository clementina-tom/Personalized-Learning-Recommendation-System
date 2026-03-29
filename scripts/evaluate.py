"""
scripts/evaluate.py
===================
Evaluate PLRS against baselines on a held-out test set.

Usage:
    python scripts/evaluate.py --data interactions.csv --domain math
    python scripts/evaluate.py --data interactions.csv --model checkpoints/sakt_best.pt --domain math
    python scripts/evaluate.py --data interactions.csv --output results/eval.json

Output:
    Prints a formatted report to stdout.
    Optionally saves full JSON results to --output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PLRS knowledge tracing and recommendations")
    p.add_argument("--data",        required=True,  help="Path to interactions CSV")
    p.add_argument("--domain",      default="math", choices=["math", "cs"],
                   help="Curriculum domain (default: math)")
    p.add_argument("--model",       default=None,   help="Path to trained SAKT .pt file (optional)")
    p.add_argument("--output",      default=None,   help="Path to save JSON report")
    p.add_argument("--test-split",  default=0.2,    type=float)
    p.add_argument("--student-col", default="student_id")
    p.add_argument("--skill-col",   default="skill_id")
    p.add_argument("--correct-col", default="correct")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--no-baselines", action="store_true")
    return p.parse_args()


def build_skill_to_topic(domain: str) -> dict[int, str]:
    """Build a skill_id → topic_id mapping using skill_encoder_v2.csv."""
    import pandas as pd

    encoder_path = ROOT / "data" / "skill_encoder_v2.csv"
    if not encoder_path.exists():
        print(f"  [WARN] {encoder_path} not found — recommendation metrics will be limited")
        return {}

    ACTIVITY_TO_TOPIC = {
        "math": {
            "oucontent": "algebraic_expressions", "forumng": "statistics_basic",
            "homepage": "whole_numbers", "subpage": "plane_shapes",
            "resource": "indices", "url": "number_bases",
            "ouwiki": "proportion_variation", "glossary": "algebraic_factorization",
            "quiz": "quadratic_equations",
        },
        "cs": {
            "oucontent": "programming_concepts", "forumng": "ethics_technology",
            "homepage": "computer_basics", "subpage": "html_basics",
            "resource": "networking_fundamentals", "url": "internet_basics",
            "ouwiki": "cloud_basics", "glossary": "intro_databases",
            "quiz": "python_basics",
        },
    }
    mapping = ACTIVITY_TO_TOPIC[domain]
    df = pd.read_csv(encoder_path)
    skill_to_topic = {}
    for _, row in df.iterrows():
        topic = mapping.get(row.get("activity_type", ""))
        if topic:
            skill_to_topic[int(row["skill_id"])] = topic
    return skill_to_topic


def main() -> None:
    args = parse_args()

    from plrs.curriculum.loader import load_dag
    from plrs.model.evaluator import PLRSEvaluator
    from plrs.model.trainer import load_sequences_from_csv
    from plrs.pipeline import PLRSPipeline

    print("\n🧠 PLRS Evaluation")
    print("=" * 50)

    # Load sequences
    timestamp_col = None if args.timestamp_col == "none" else args.timestamp_col
    all_sequences = load_sequences_from_csv(
        path=args.data,
        student_col=args.student_col,
        skill_col=args.skill_col,
        correct_col=args.correct_col,
        timestamp_col=timestamp_col,
    )

    import numpy as np
    np.random.seed(42)
    idx = np.random.permutation(len(all_sequences))
    n_test = max(1, int(len(all_sequences) * args.test_split))
    test_sequences  = [all_sequences[i] for i in idx[:n_test]]
    train_sequences = [all_sequences[i] for i in idx[n_test:]]
    print(f"Train: {len(train_sequences)} students | Test: {len(test_sequences)} students")

    # Load curriculum + pipeline
    dag_path = ROOT / "data" / "knowledge_maps" / f"{args.domain}_dag.json"
    curriculum = load_dag(dag_path)
    pipeline   = PLRSPipeline(curriculum, model_path=args.model)
    print(f"Curriculum: {curriculum.domain} ({curriculum.num_nodes} nodes)")
    print(f"Model: {'SAKT loaded' if args.model else 'no model (baselines only)'}")

    # Build skill→topic mapping
    skill_to_topic = build_skill_to_topic(args.domain)
    print(f"Skill→topic mapping: {len(skill_to_topic)} entries")

    # Evaluate
    print("\nRunning evaluation...")
    evaluator = PLRSEvaluator(pipeline)
    report = evaluator.evaluate(
        test_sequences=test_sequences,
        skill_to_topic=skill_to_topic,
        train_sequences=train_sequences,
        include_baselines=not args.no_baselines,
    )

    # Print report
    report.print()

    # Save JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
