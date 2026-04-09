"""
plrs.api.auth
=============
API key management for the PLRS REST API.

Design: in-memory store backed by an optional JSON file.
This is the open-core implementation — swap _store backend
for a database (Postgres, Redis) in the hosted version.

Key format: plrs_<32 random hex chars>
Example:    plrs_a3f8c2e1d4b7a9f0e5c3b8d2a1f6e4c9

Usage:
    from plrs.api.auth import KeyStore, require_api_key

    store = KeyStore()
    key   = store.create_key(name="my-app", tier="standard")
    store.validate(key)   # → APIKey or raises
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ── Key tiers ─────────────────────────────────────────────────────────────────

TIERS: dict[str, dict[str, int]] = {
    "free": {
        "requests_per_minute": 10,
        "requests_per_day":    100,
    },
    "standard": {
        "requests_per_minute": 60,
        "requests_per_day":    1_000,
    },
    "premium": {
        "requests_per_minute": 300,
        "requests_per_day":    10_000,
    },
    "internal": {
        "requests_per_minute": 10_000,
        "requests_per_day":    1_000_000,
    },
}

DEFAULT_TIER = "standard"


# ── Key dataclass ─────────────────────────────────────────────────────────────

@dataclass
class APIKey:
    key_id:    str            # plrs_<32 hex>
    name:      str            # human label e.g. "my-school-app"
    tier:      str            # "free" | "standard" | "premium" | "internal"
    created_at: float         # unix timestamp
    is_active: bool = True
    metadata:  dict[str, Any] = field(default_factory=dict)

    @property
    def limits(self) -> dict[str, int]:
        return TIERS.get(self.tier, TIERS[DEFAULT_TIER])

    @property
    def requests_per_minute(self) -> int:
        return self.limits["requests_per_minute"]

    @property
    def requests_per_day(self) -> int:
        return self.limits["requests_per_day"]

    def to_dict(self) -> dict:
        return {
            "key_id":     self.key_id,
            "name":       self.name,
            "tier":       self.tier,
            "created_at": self.created_at,
            "is_active":  self.is_active,
            "metadata":   self.metadata,
            "limits":     self.limits,
        }


# ── Key store ─────────────────────────────────────────────────────────────────

class KeyStore:
    """
    In-memory API key store with optional JSON persistence.

    Parameters
    ----------
    persist_path : str or Path, optional
        Path to a JSON file for persistence across restarts.
        If None, keys are lost on restart (fine for dev/testing).
    """

    KEY_PREFIX = "plrs_"
    KEY_BYTES  = 16   # 32 hex chars

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._keys:  dict[str, APIKey] = {}   # key_id → APIKey
        self._hashes: dict[str, str]   = {}   # sha256(key_id) → key_id (for fast lookup)
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    # CRUD                                                                #
    # ------------------------------------------------------------------ #

    def create_key(
        self,
        name: str,
        tier: str = DEFAULT_TIER,
        metadata: dict | None = None,
    ) -> str:
        """
        Create a new API key.

        Parameters
        ----------
        name : str
            Human-readable label for this key.
        tier : str
            Rate limit tier.
        metadata : dict, optional
            Arbitrary metadata (e.g. {"org": "Greenfield Academy"}).

        Returns
        -------
        str — the raw API key (shown once; store it safely).
        """
        if tier not in TIERS:
            raise ValueError(f"Unknown tier '{tier}'. Valid: {list(TIERS)}")

        raw_key = self.KEY_PREFIX + secrets.token_hex(self.KEY_BYTES)
        key_hash = self._hash(raw_key)

        api_key = APIKey(
            key_id=raw_key,
            name=name,
            tier=tier,
            created_at=time.time(),
            metadata=metadata or {},
        )

        self._keys[raw_key]   = api_key
        self._hashes[key_hash] = raw_key

        if self._persist_path:
            self._save()

        return raw_key

    def validate(self, raw_key: str) -> APIKey:
        """
        Validate an API key.

        Returns the APIKey if valid and active.
        Raises KeyError if not found, ValueError if inactive.
        """
        api_key = self._keys.get(raw_key)
        if api_key is None:
            raise KeyError(f"API key not found.")
        if not api_key.is_active:
            raise ValueError(f"API key '{api_key.name}' is inactive.")
        return api_key

    def revoke(self, raw_key: str) -> None:
        """Deactivate an API key."""
        api_key = self._keys.get(raw_key)
        if api_key is None:
            raise KeyError("API key not found.")
        api_key.is_active = False
        if self._persist_path:
            self._save()

    def delete(self, raw_key: str) -> None:
        """Permanently delete an API key."""
        if raw_key not in self._keys:
            raise KeyError("API key not found.")
        key_hash = self._hash(raw_key)
        del self._keys[raw_key]
        self._hashes.pop(key_hash, None)
        if self._persist_path:
            self._save()

    def list_keys(self) -> list[dict]:
        """Return all keys (without exposing the raw key value)."""
        return [
            {
                "name":       k.name,
                "tier":       k.tier,
                "is_active":  k.is_active,
                "created_at": k.created_at,
                "limits":     k.limits,
                # Never return raw key_id in list — only shown at creation
                "key_prefix": k.key_id[:12] + "...",
            }
            for k in self._keys.values()
        ]

    def __len__(self) -> int:
        return len(self._keys)

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        """Persist keys to JSON file."""
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {key_id: asdict(api_key) for key_id, api_key in self._keys.items()}
        with open(self._persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load keys from JSON file."""
        with open(self._persist_path) as f:
            data = json.load(f)
        for key_id, d in data.items():
            self._keys[key_id] = APIKey(**d)
            self._hashes[self._hash(key_id)] = key_id

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()


# ── Global store instance ─────────────────────────────────────────────────────

_key_store: KeyStore = KeyStore(
    persist_path=os.getenv("PLRS_KEYS_PATH", None)
)


def get_key_store() -> KeyStore:
    """Return the global key store."""
    return _key_store


def init_key_store(persist_path: str | Path | None = None) -> KeyStore:
    """(Re)initialise the global key store."""
    global _key_store
    _key_store = KeyStore(persist_path=persist_path)
    return _key_store
