"""
plrs.model.sakt
===============
Self-Attentive Knowledge Tracing (SAKT) model.

Architecture: transformer-style attention over student interaction sequences.
Each interaction is encoded as (skill_id + correctness * n_skills).

Reference: Pandey & Karypis, 2019 — "A Self-Attentive model for Knowledge Tracing"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class SAKTModel(nn.Module):
    """
    SAKT: Self-Attentive Knowledge Tracing.

    Parameters
    ----------
    num_skills : int
        Total number of unique skills in the dataset.
    embed_dim : int
        Embedding dimension for interactions and positions.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    max_seq_len : int
        Maximum interaction sequence length.
    """

    def __init__(
        self,
        num_skills: int,
        embed_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_seq_len: int = 100,
    ) -> None:
        super().__init__()
        self.num_skills = num_skills
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Interaction embedding: (skill, correct) → dense vector
        self.interaction_embed = nn.Embedding(2 * num_skills + 2, embed_dim, padding_idx=0)  # +2: shift+1 means max index = 2*n+1
        # Positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Skill query embedding for target prediction
        self.skill_embed = nn.Embedding(num_skills + 1, embed_dim, padding_idx=0)

        self.output_layer = nn.Linear(embed_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        interactions: torch.Tensor,   # (batch, seq_len)
        target_skills: torch.Tensor,  # (batch, seq_len)
        mask: torch.Tensor,           # (batch, seq_len) bool — True = real token
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns
        -------
        torch.Tensor of shape (batch, seq_len) — logits per position.
        """
        batch_size, seq_len = interactions.shape
        positions = torch.arange(seq_len, device=interactions.device).unsqueeze(0)

        x = self.interaction_embed(interactions) + self.pos_embed(positions)
        x = self.dropout(x)

        # Causal mask — bool upper-triangular (MHA handles conversion internally)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        # Key padding mask: True = ignore (PyTorch MHA convention)
        key_padding_mask = ~mask  # (batch, seq_len) bool

        x_attn, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
        # Replace any NaN in attention output (from fully-masked rows) with 0
        x_attn = torch.nan_to_num(x_attn, nan=0.0)
        x = self.layer_norm1(x + x_attn)
        x = self.layer_norm2(x + self.ffn(x))

        # Concatenate with target skill embedding for final prediction
        skill_x = self.skill_embed(target_skills)
        out = self.output_layer(torch.cat([x, skill_x], dim=-1)).squeeze(-1)

        return out  # (batch, seq_len) logits

    # ------------------------------------------------------------------ #
    # Inference helpers                                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_mastery(
        self,
        skill_seq: list[int],
        correct_seq: list[int],
        device: torch.device | str = "cpu",
    ) -> dict[int, float]:
        """
        Run inference on a student's interaction history.

        Parameters
        ----------
        skill_seq : list[int]
            Sequence of skill IDs the student interacted with.
        correct_seq : list[int]
            Corresponding correctness (1 = correct, 0 = incorrect).
        device : str or torch.device

        Returns
        -------
        dict[int, float]
            Mapping from skill_id → predicted mastery probability.
        """
        if len(skill_seq) < 2:
            return {}

        if len(skill_seq) > self.max_seq_len:
            skill_seq = skill_seq[-self.max_seq_len:]
            correct_seq = correct_seq[-self.max_seq_len:]

        interactions = [s + c * self.num_skills + 1 for s, c in zip(skill_seq[:-1], correct_seq[:-1])]  # +1: reserve 0 for padding
        target_skills = skill_seq[1:]

        seq_len = len(interactions)
        pad_len = self.max_seq_len - seq_len

        interactions_padded = [0] * pad_len + interactions
        target_padded = [0] * pad_len + target_skills
        mask = [False] * pad_len + [True] * seq_len

        interactions_t = torch.LongTensor([interactions_padded]).to(device)
        target_t = torch.LongTensor([target_padded]).to(device)
        mask_t = torch.BoolTensor([mask]).to(device)

        self.eval()
        self.to(device)

        logits = self(interactions_t, target_t, mask_t)
        probs = torch.sigmoid(logits).squeeze(0)

        real_probs = probs[torch.BoolTensor(mask)].cpu().numpy()
        mastery = {
            int(skill_id): float(prob)
            for skill_id, prob in zip(target_skills, real_probs)
        }
        return mastery

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path, config: dict[str, Any] | None = None) -> None:
        """Save model weights and config to a .pt file."""
        payload = {
            "state_dict": self.state_dict(),
            "config": config or {
                "num_skills": self.num_skills,
                "embed_dim": self.embed_dim,
                "max_seq_len": self.max_seq_len,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "SAKTModel":
        """Load a saved SAKT model."""
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        model = cls(
            num_skills=config["num_skills"],
            embed_dim=config.get("embed_dim", 64),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.2),
            max_seq_len=config.get("max_seq_len", 100),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        return model
