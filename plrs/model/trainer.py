"""
plrs.model.trainer
==================
Training loop for the SAKT knowledge tracing model.

Handles:
  - Dataset preparation from raw interaction logs
  - Train / validation split
  - Training with early stopping
  - Checkpoint saving (best val AUC)
  - Metrics: AUC, accuracy, loss

Expected input format (CSV or DataFrame):
    student_id | skill_id | correct | timestamp (optional)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ------------------------------------------------------------------ #
# Dataset                                                             #
# ------------------------------------------------------------------ #

class KTDataset(Dataset):
    """
    Knowledge Tracing dataset.

    Each sample is one student's full interaction sequence, windowed to
    max_seq_len. Long sequences are split into multiple windows.

    Parameters
    ----------
    sequences : list of (skill_seq, correct_seq)
        Each element is a tuple of parallel lists.
    max_seq_len : int
    n_skills : int
    """

    def __init__(
        self,
        sequences: list[tuple[list[int], list[int]]],
        max_seq_len: int = 100,
        n_skills: int = 5736,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.n_skills = n_skills
        self.samples: list[tuple[list[int], list[int]]] = []

        for skill_seq, correct_seq in sequences:
            # Window long sequences
            for start in range(0, max(1, len(skill_seq) - 1), max_seq_len // 2):
                end = start + max_seq_len + 1
                s = skill_seq[start:end]
                c = correct_seq[start:end]
                if len(s) >= 2:
                    self.samples.append((s, c))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        skill_seq, correct_seq = self.samples[idx]

        if len(skill_seq) > self.max_seq_len + 1:
            skill_seq = skill_seq[-self.max_seq_len - 1:]
            correct_seq = correct_seq[-self.max_seq_len - 1:]

        interactions = [s + c * self.n_skills + 1 for s, c in zip(skill_seq[:-1], correct_seq[:-1])]  # +1: reserve 0 for padding
        target_skills = skill_seq[1:]
        target_correct = correct_seq[1:]

        seq_len = len(interactions)
        pad_len = self.max_seq_len - seq_len

        interactions_padded = [0] * pad_len + interactions
        target_padded       = [0] * pad_len + target_skills
        correct_padded      = [0] * pad_len + target_correct
        mask                = [False] * pad_len + [True] * seq_len

        return {
            "interactions":    torch.LongTensor(interactions_padded),
            "target_skills":   torch.LongTensor(target_padded),
            "target_correct":  torch.FloatTensor(correct_padded),
            "mask":            torch.BoolTensor(mask),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ------------------------------------------------------------------ #
# Trainer config                                                      #
# ------------------------------------------------------------------ #

@dataclass
class TrainerConfig:
    # Model
    num_skills:   int   = 5736
    embed_dim:    int   = 64
    num_heads:    int   = 8
    dropout:      float = 0.2
    max_seq_len:  int   = 100

    # Training
    epochs:       int   = 50
    batch_size:   int   = 64
    lr:           float = 1e-3
    weight_decay: float = 1e-5
    val_split:    float = 0.1

    # Early stopping
    patience:     int   = 5
    min_delta:    float = 1e-4

    # Output
    output_dir:   str   = "checkpoints"
    run_name:     str   = "sakt_run"

    # Device
    device:       str   = "auto"   # "auto" | "cpu" | "cuda" | "mps"


# ------------------------------------------------------------------ #
# Trainer                                                             #
# ------------------------------------------------------------------ #

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float
    val_acc: float
    elapsed: float


class SAKTTrainer:
    """
    Trainer for the SAKT knowledge tracing model.

    Parameters
    ----------
    config : TrainerConfig
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # Public API                                                        #
    # ---------------------------------------------------------------- #

    def fit(
        self,
        sequences: list[tuple[list[int], list[int]]],
        val_sequences: list[tuple[list[int], list[int]]] | None = None,
    ) -> list[EpochMetrics]:
        """
        Train the SAKT model on interaction sequences.

        Parameters
        ----------
        sequences : list of (skill_seq, correct_seq)
            Training data. Each element is a student's full history.
        val_sequences : list of (skill_seq, correct_seq), optional
            If None, val_split fraction of sequences is held out.

        Returns
        -------
        list[EpochMetrics] — training history
        """
        from plrs.model.sakt import SAKTModel

        cfg = self.config

        # Split if no explicit val set
        if val_sequences is None:
            n_val = max(1, int(len(sequences) * cfg.val_split))
            idx = np.random.permutation(len(sequences))
            val_sequences  = [sequences[i] for i in idx[:n_val]]
            train_sequences = [sequences[i] for i in idx[n_val:]]
        else:
            train_sequences = sequences

        print(f"Training samples : {len(train_sequences)} students")
        print(f"Validation samples: {len(val_sequences)} students")
        print(f"Device: {self.device}")

        train_ds = KTDataset(train_sequences, cfg.max_seq_len, cfg.num_skills)
        val_ds   = KTDataset(val_sequences,   cfg.max_seq_len, cfg.num_skills)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        model = SAKTModel(
            num_skills=cfg.num_skills,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len,
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # Zero out NaN gradients that arise from softmax backward over fully-padded rows.
        # This is a known issue with nn.MultiheadAttention + bool key_padding_mask.
        # The hook is safe: it only zeroes truly NaN gradients, never valid ones.
        def _zero_nan_grad(grad: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(grad, nan=0.0)
        for p in model.parameters():
            p.register_hook(_zero_nan_grad)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=2, factor=0.5
        )
        criterion = nn.BCEWithLogitsLoss()

        history: list[EpochMetrics] = []
        best_auc = 0.0
        patience_counter = 0
        best_path = self.output_dir / f"{cfg.run_name}_best.pt"

        print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>9} {'Val AUC':>9} {'Val Acc':>9} {'Time':>7}")
        print("-" * 58)

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_auc, val_acc = self._val_epoch(model, val_loader, criterion)

            scheduler.step(val_auc)
            elapsed = time.time() - t0

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_auc=val_auc,
                val_acc=val_acc,
                elapsed=elapsed,
            )
            history.append(metrics)

            print(
                f"{epoch:>6} {train_loss:>11.4f} {val_loss:>9.4f} "
                f"{val_auc:>9.4f} {val_acc:>9.4f} {elapsed:>6.1f}s"
            )

            # Save best
            if val_auc > best_auc + cfg.min_delta:
                best_auc = val_auc
                patience_counter = 0
                model.save(best_path, config=self._model_config())
                print(f"         ✅ New best AUC: {best_auc:.4f} → saved to {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={cfg.patience})")
                    break

        print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
        print(f"Best model: {best_path}")
        return history

    # ---------------------------------------------------------------- #
    # Internal                                                          #
    # ---------------------------------------------------------------- #

    def _train_epoch(self, model, loader, optimizer, criterion) -> float:
        model.train()
        total_loss = 0.0

        for batch in loader:
            interactions   = batch["interactions"].to(self.device)
            target_skills  = batch["target_skills"].to(self.device)
            target_correct = batch["target_correct"].to(self.device)
            mask           = batch["mask"].to(self.device)

            optimizer.zero_grad()
            logits = model(interactions, target_skills, mask)

            # Only compute loss on real (non-padded) positions
            real_logits  = logits[mask]
            real_targets = target_correct[mask]

            loss = criterion(real_logits, real_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _val_epoch(self, model, loader, criterion) -> tuple[float, float, float]:
        model.eval()
        total_loss = 0.0
        all_probs: list[float] = []
        all_labels: list[float] = []

        for batch in loader:
            interactions   = batch["interactions"].to(self.device)
            target_skills  = batch["target_skills"].to(self.device)
            target_correct = batch["target_correct"].to(self.device)
            mask           = batch["mask"].to(self.device)

            logits = model(interactions, target_skills, mask)
            real_logits  = logits[mask]
            real_targets = target_correct[mask]

            loss = criterion(real_logits, real_targets)
            total_loss += loss.item()

            probs = torch.sigmoid(real_logits).cpu().numpy()
            labels = real_targets.cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

        avg_loss = total_loss / max(len(loader), 1)
        all_probs_arr  = np.array(all_probs)
        all_labels_arr = np.array(all_labels)

        # Guard against NaN (can occur with very small val sets)
        all_probs_arr  = np.nan_to_num(all_probs_arr,  nan=0.5)
        all_labels_arr = np.nan_to_num(all_labels_arr, nan=0.0)

        if HAS_SKLEARN and len(np.unique(all_labels_arr)) > 1:
            auc = float(roc_auc_score(all_labels_arr, all_probs_arr))
        else:
            auc = 0.5  # fallback (single class or no sklearn)

        acc = float(((all_probs_arr >= 0.5) == all_labels_arr).mean())
        return avg_loss, auc, acc

    def _model_config(self) -> dict:
        cfg = self.config
        return {
            "num_skills":  cfg.num_skills,
            "embed_dim":   cfg.embed_dim,
            "num_heads":   cfg.num_heads,
            "dropout":     cfg.dropout,
            "max_seq_len": cfg.max_seq_len,
        }

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)


# ------------------------------------------------------------------ #
# Utilities                                                           #
# ------------------------------------------------------------------ #

def load_sequences_from_csv(
    path: str | Path,
    student_col: str = "student_id",
    skill_col: str = "skill_id",
    correct_col: str = "correct",
    timestamp_col: str | None = "timestamp",
    min_seq_len: int = 5,
) -> list[tuple[list[int], list[int]]]:
    """
    Load student interaction sequences from a CSV file.

    Parameters
    ----------
    path : str or Path
        CSV with columns: student_id, skill_id, correct, [timestamp]
    student_col, skill_col, correct_col : str
        Column names.
    timestamp_col : str or None
        If provided, sort interactions by this column within each student.
    min_seq_len : int
        Drop students with fewer than this many interactions.

    Returns
    -------
    list of (skill_seq, correct_seq) tuples
    """
    import pandas as pd

    df = pd.read_csv(path)

    required = [student_col, skill_col, correct_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {df.columns.tolist()}")

    if timestamp_col and timestamp_col in df.columns:
        df = df.sort_values([student_col, timestamp_col])

    sequences = []
    for _, group in df.groupby(student_col):
        skills   = group[skill_col].astype(int).tolist()
        corrects = group[correct_col].astype(int).tolist()
        if len(skills) >= min_seq_len:
            sequences.append((skills, corrects))

    print(f"Loaded {len(sequences)} student sequences from {path}")
    return sequences
