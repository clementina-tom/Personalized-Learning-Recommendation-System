"""
scripts/train.py
================
Train a SAKT model on student interaction data.

Usage:
    python scripts/train.py --data path/to/interactions.csv
    python scripts/train.py --data interactions.csv --epochs 30 --embed-dim 128
    python scripts/train.py --data interactions.csv --device cuda --run-name sakt_v2

Expected CSV format:
    student_id, skill_id, correct, [timestamp]

Output:
    checkpoints/<run_name>_best.pt   ← best model by val AUC
    checkpoints/<run_name>_history.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAKT knowledge tracing model")

    # Data
    p.add_argument("--data",          required=True,  help="Path to interactions CSV")
    p.add_argument("--student-col",   default="student_id")
    p.add_argument("--skill-col",     default="skill_id")
    p.add_argument("--correct-col",   default="correct")
    p.add_argument("--timestamp-col", default="timestamp", help="Set to 'none' to skip sorting")
    p.add_argument("--min-seq-len",   default=5, type=int)

    # Model
    p.add_argument("--num-skills",  default=None, type=int,
                   help="Number of unique skills. Auto-detected from data if not set.")
    p.add_argument("--embed-dim",   default=64,   type=int)
    p.add_argument("--num-heads",   default=8,    type=int)
    p.add_argument("--dropout",     default=0.2,  type=float)
    p.add_argument("--max-seq-len", default=100,  type=int)

    # Training
    p.add_argument("--epochs",        default=50,   type=int)
    p.add_argument("--batch-size",    default=64,   type=int)
    p.add_argument("--lr",            default=1e-3, type=float)
    p.add_argument("--weight-decay",  default=1e-5, type=float)
    p.add_argument("--val-split",     default=0.1,  type=float)
    p.add_argument("--patience",      default=5,    type=int)

    # Output
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--run-name",   default="sakt_run")
    p.add_argument("--device",     default="auto", choices=["auto", "cpu", "cuda", "mps"])

    return p.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd
    from plrs.model.trainer import SAKTTrainer, TrainerConfig, load_sequences_from_csv

    print("\n🧠 PLRS — SAKT Training")
    print("=" * 50)

    # Load data
    timestamp_col = None if args.timestamp_col == "none" else args.timestamp_col
    sequences = load_sequences_from_csv(
        path=args.data,
        student_col=args.student_col,
        skill_col=args.skill_col,
        correct_col=args.correct_col,
        timestamp_col=timestamp_col,
        min_seq_len=args.min_seq_len,
    )

    if not sequences:
        print("❌ No sequences loaded. Check your CSV format and column names.")
        return

    # Auto-detect num_skills if not provided
    if args.num_skills is None:
        df = pd.read_csv(args.data)
        num_skills = int(df[args.skill_col].max()) + 1
        print(f"Auto-detected num_skills: {num_skills}")
    else:
        num_skills = args.num_skills

    config = TrainerConfig(
        num_skills=num_skills,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        patience=args.patience,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
    )

    print(f"\nConfig:")
    print(f"  num_skills  : {config.num_skills}")
    print(f"  embed_dim   : {config.embed_dim}")
    print(f"  epochs      : {config.epochs} (patience={config.patience})")
    print(f"  batch_size  : {config.batch_size}")
    print(f"  lr          : {config.lr}")
    print(f"  output_dir  : {config.output_dir}/{config.run_name}_best.pt")
    print()

    trainer = SAKTTrainer(config)
    history = trainer.fit(sequences)

    # Save history
    history_path = Path(args.output_dir) / f"{args.run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(
            [
                {
                    "epoch": m.epoch,
                    "train_loss": round(m.train_loss, 6),
                    "val_loss": round(m.val_loss, 6),
                    "val_auc": round(m.val_auc, 6),
                    "val_acc": round(m.val_acc, 6),
                    "elapsed_s": round(m.elapsed, 2),
                }
                for m in history
            ],
            f,
            indent=2,
        )

    best_auc = max(m.val_auc for m in history)
    print(f"\nHistory saved to: {history_path}")
    print(f"Best val AUC: {best_auc:.4f}")
    print("\nTo serve the trained model:")
    print(f"  python scripts/serve.py --model {args.output_dir}/{args.run_name}_best.pt")


if __name__ == "__main__":
    main()
