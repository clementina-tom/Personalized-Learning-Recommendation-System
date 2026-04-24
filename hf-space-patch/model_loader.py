"""
HF Space model loader — updated for SAKTWithDecay (v0.2.0 weights).

Drop this file into your HF Space as `model_loader.py` and call
`load_model_from_hub()` in app.py instead of the old loading logic.

The v0.2.0 weights (sakt_decay_best.pt) are saved with our new format:
    {
        "state_dict": {...},
        "model_type": "SAKTWithDecay",
        "config": {"num_skills": 20, "embed_dim": 64, ...}
    }

Falls back gracefully to mastery-dict mode if weights can't be loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

HF_REPO = "Clementio/PLRS"


def load_model_from_hub(device: str = "cpu"):
    """
    Load SAKTWithDecay weights from HuggingFace Hub.

    Tries files in priority order:
      1. sakt_decay_best.pt  (v0.2.0 — real OULAD training, AUC 0.8613)
      2. sakt_model.pt       (v0.1.0 — synthetic baseline, AUC 0.7692)

    Returns (model, model_type_str) or (None, "unavailable").
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None, "huggingface_hub not installed"

    # Try SAKTWithDecay first (v0.2.0)
    for filename, model_type in [
        ("sakt_decay_best.pt", "SAKTWithDecay"),
        ("sakt_model.pt",      "SAKTModel"),
    ]:
        try:
            path = hf_hub_download(repo_id=HF_REPO, filename=filename)
            model = _load_weights(path, model_type, device)
            if model is not None:
                return model, model_type
        except Exception:
            continue

    return None, "unavailable"


def _load_weights(path: str, preferred_type: str, device: str):
    """Load model weights from a .pt file, handling both old and new formats."""
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return None

    # ── New format (v0.2.0): {"state_dict": ..., "model_type": ..., "config": ...}
    if isinstance(payload, dict) and "state_dict" in payload:
        cfg        = payload.get("config", {})
        model_type = payload.get("model_type", preferred_type)

        if model_type == "SAKTWithDecay":
            from plrs.model.sakt_decay import SAKTWithDecay
            model = SAKTWithDecay(
                num_skills=cfg.get("num_skills", 20),
                embed_dim=cfg.get("embed_dim", 64),
                num_heads=cfg.get("num_heads", 8),
                dropout=cfg.get("dropout", 0.2),
                max_seq_len=cfg.get("max_seq_len", 100),
                decay_init=cfg.get("decay_init", 1.0),
            )
        else:
            from plrs.model.sakt import SAKTModel
            model = SAKTModel(
                num_skills=cfg.get("num_skills", 5736),
                embed_dim=cfg.get("embed_dim", 64),
                num_heads=cfg.get("num_heads", 8),
                dropout=cfg.get("dropout", 0.2),
                max_seq_len=cfg.get("max_seq_len", 100),
            )

        try:
            model.load_state_dict(payload["state_dict"], strict=False)
            model.eval()
            model.to(device)
            return model
        except Exception:
            return None

    # ── Old format (v0.1.0 FYP): raw state_dict + separate config.json
    try:
        config_path = Path(path).parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
        else:
            config = {"num_skills": 5736, "embed_dim": 64}

        from plrs.model.sakt import SAKTModel
        model = SAKTModel(
            num_skills=config.get("num_skills", 5736),
            embed_dim=config.get("embed_dim", 64),
        )
        model.load_state_dict(payload, strict=False)
        model.eval()
        return model
    except Exception:
        return None
