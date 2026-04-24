"""
plrs.api.db
===========
PostgreSQL-backed persistence layer for PLRS API keys and usage logs.

This module provides:
  - SQLAlchemy models: APIKeyRecord, UsageLog
  - PostgresKeyStore: drop-in replacement for in-memory KeyStore
  - get_db_key_store(): factory that returns the right store based on env

The in-memory KeyStore (plrs.api.auth) still works with no configuration.
Set PLRS_DB_URL to switch to Postgres:

    PLRS_DB_URL=postgresql://plrs:secret@localhost:5432/plrs

Schema:
    api_keys   — one row per key (id, name, tier, key_hash, is_active, created_at, metadata)
    usage_logs — one row per request (key_id, endpoint, timestamp, response_ms)

Usage:
    from plrs.api.db import get_db_key_store

    store = get_db_key_store()   # returns PostgresKeyStore or in-memory KeyStore
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

# ── Tier definitions (duplicated from auth.py for self-contained module) ────────
TIERS: dict[str, dict[str, int]] = {
    "free":     {"requests_per_minute": 10,    "requests_per_day": 100},
    "standard": {"requests_per_minute": 60,    "requests_per_day": 1_000},
    "premium":  {"requests_per_minute": 300,   "requests_per_day": 10_000},
    "internal": {"requests_per_minute": 10_000,"requests_per_day": 1_000_000},
}
DEFAULT_TIER = "standard"

# ── SQLAlchemy setup ──────────────────────────────────────────────────────────

try:
    from sqlalchemy import (
        Boolean, Column, Float, Integer, String, Text,
        create_engine, text,
    )
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


# ── Models ────────────────────────────────────────────────────────────────────

if HAS_SQLALCHEMY:
    class Base(DeclarativeBase):
        pass

    class APIKeyRecord(Base):
        """Persistent API key record."""
        __tablename__ = "api_keys"

        id         = Column(Integer, primary_key=True, autoincrement=True)
        key_id     = Column(String(64), unique=True, nullable=False, index=True)
        key_hash   = Column(String(64), unique=True, nullable=False, index=True)
        name       = Column(String(255), nullable=False)
        tier       = Column(String(32), nullable=False, default="standard")
        is_active  = Column(Boolean, nullable=False, default=True)
        created_at = Column(Float, nullable=False)
        meta_json  = Column(Text, nullable=False, default="{}")

        def to_api_key(self) -> "PostgresAPIKey":
                return PostgresAPIKey(
                key_id=self.key_id,
                name=self.name,
                tier=self.tier,
                created_at=self.created_at,
                is_active=self.is_active,
                metadata=json.loads(self.meta_json or "{}"),
            )

    class UsageLog(Base):
        """Per-request usage log for billing and analytics."""
        __tablename__ = "usage_logs"

        id          = Column(Integer, primary_key=True, autoincrement=True)
        key_id      = Column(String(64), nullable=False, index=True)
        endpoint    = Column(String(255), nullable=False)
        timestamp   = Column(Float, nullable=False)
        response_ms = Column(Integer, nullable=True)
        status_code = Column(Integer, nullable=True)


# ── PostgresAPIKey ─────────────────────────────────────────────────────────────

@dataclass
class PostgresAPIKey:
    """API key with Postgres-backed identity."""
    key_id:     str
    name:       str
    tier:       str
    created_at: float
    is_active:  bool = True
    metadata:   dict[str, Any] = field(default_factory=dict)

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


# ── PostgresKeyStore ──────────────────────────────────────────────────────────

class PostgresKeyStore:
    """
    PostgreSQL-backed API key store.

    Drop-in replacement for the in-memory KeyStore.
    Same interface — swap by setting PLRS_DB_URL.

    Parameters
    ----------
    db_url : str
        SQLAlchemy database URL.
        e.g. "postgresql://plrs:secret@localhost:5432/plrs"
    echo : bool
        Log SQL statements (useful for debugging).
    """

    KEY_PREFIX = "plrs_"
    KEY_BYTES  = 16

    def __init__(self, db_url: str, echo: bool = False) -> None:
        if not HAS_SQLALCHEMY:
            raise ImportError(
                "sqlalchemy not installed. "
                "Run: pip install sqlalchemy psycopg2-binary"
            )
        self._engine       = create_engine(db_url, echo=echo, pool_pre_ping=True)
        self._SessionLocal = sessionmaker(bind=self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)

    # ------------------------------------------------------------------ #
    # CRUD                                                                #
    # ------------------------------------------------------------------ #

    def create_key(
        self,
        name: str,
        tier: str = "standard",
        metadata: dict | None = None,
    ) -> str:
        if tier not in TIERS:
            raise ValueError(f"Unknown tier '{tier}'. Valid: {list(TIERS)}")

        raw_key  = self.KEY_PREFIX + secrets.token_hex(self.KEY_BYTES)
        key_hash = self._hash(raw_key)

        with self._SessionLocal() as session:
            record = APIKeyRecord(
                key_id=raw_key,
                key_hash=key_hash,
                name=name,
                tier=tier,
                is_active=True,
                created_at=time.time(),
                meta_json=json.dumps(metadata or {}),
            )
            session.add(record)
            session.commit()

        return raw_key

    def validate(self, raw_key: str) -> PostgresAPIKey:
        with self._SessionLocal() as session:
            record = session.query(APIKeyRecord).filter_by(key_id=raw_key).first()
            if record is None:
                raise KeyError("API key not found.")
            if not record.is_active:
                raise ValueError(f"API key '{record.name}' is inactive.")
            return record.to_api_key()

    def revoke(self, raw_key: str) -> None:
        with self._SessionLocal() as session:
            record = session.query(APIKeyRecord).filter_by(key_id=raw_key).first()
            if record is None:
                raise KeyError("API key not found.")
            record.is_active = False
            session.commit()

    def delete(self, raw_key: str) -> None:
        with self._SessionLocal() as session:
            record = session.query(APIKeyRecord).filter_by(key_id=raw_key).first()
            if record is None:
                raise KeyError("API key not found.")
            session.delete(record)
            session.commit()

    def list_keys(self) -> list[dict]:
        with self._SessionLocal() as session:
            records = session.query(APIKeyRecord).all()
            return [
                {
                    "name":       r.name,
                    "tier":       r.tier,
                    "is_active":  r.is_active,
                    "created_at": r.created_at,
                    "limits":     r.to_api_key().limits,
                    "key_prefix": r.key_id[:12] + "...",
                }
                for r in records
            ]

    def log_request(
        self,
        key_id: str,
        endpoint: str,
        response_ms: int | None = None,
        status_code: int | None = None,
    ) -> None:
        """Log an API request for usage analytics."""
        with self._SessionLocal() as session:
            session.add(UsageLog(
                key_id=key_id,
                endpoint=endpoint,
                timestamp=time.time(),
                response_ms=response_ms,
                status_code=status_code,
            ))
            session.commit()

    def usage_stats(self, key_id: str, last_days: int = 30) -> dict:
        """Return usage stats for a key."""
        cutoff = time.time() - (last_days * 86400)
        with self._SessionLocal() as session:
            logs = session.query(UsageLog).filter(
                UsageLog.key_id == key_id,
                UsageLog.timestamp >= cutoff,
            ).all()
            return {
                "total_requests": len(logs),
                "period_days":    last_days,
                "endpoints":      {},
            }

    def __len__(self) -> int:
        with self._SessionLocal() as session:
            return session.query(APIKeyRecord).count()

    def health_check(self) -> bool:
        """Return True if DB is reachable."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()


# ── Factory ───────────────────────────────────────────────────────────────────

def get_db_key_store():
    """
    Return the appropriate key store based on environment.

    PLRS_DB_URL set → PostgresKeyStore
    Not set         → in-memory KeyStore (falls back to SQLite if auth unavailable)
    """
    db_url = os.getenv("PLRS_DB_URL")
    if db_url:
        return PostgresKeyStore(db_url)
    try:
        from plrs.api.auth import get_key_store
        return get_key_store()
    except ImportError:
        return PostgresKeyStore("sqlite:///:memory:")
