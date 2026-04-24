"""
tests/test_docker_infra.py
===========================
Tests for PostgreSQL-backed KeyStore and Redis-backed RateLimiter.

Uses SQLite in-memory for DB tests (no real Postgres needed).
Uses fakeredis for Redis tests (no real Redis needed).
"""

from __future__ import annotations

import time
import pytest


# ── PostgresKeyStore tests (SQLite in-memory) ─────────────────────────────────

class TestPostgresKeyStore:
    """Tests using SQLite in-memory — same SQLAlchemy interface as Postgres."""

    @pytest.fixture
    def store(self):
        from plrs.api.db import PostgresKeyStore
        return PostgresKeyStore("sqlite:///:memory:")

    def test_create_key_returns_plrs_prefix(self, store):
        key = store.create_key(name="test")
        assert key.startswith("plrs_")

    def test_validate_valid_key(self, store):
        key = store.create_key(name="test", tier="standard")
        api_key = store.validate(key)
        assert api_key.name == "test"
        assert api_key.tier == "standard"
        assert api_key.is_active is True

    def test_validate_invalid_key_raises(self, store):
        with pytest.raises(KeyError):
            store.validate("plrs_doesnotexist1234567890abcdef")

    def test_validate_revoked_key_raises(self, store):
        key = store.create_key(name="test")
        store.revoke(key)
        with pytest.raises(ValueError, match="inactive"):
            store.validate(key)

    def test_revoke_deactivates(self, store):
        key = store.create_key(name="test")
        store.revoke(key)
        keys = store.list_keys()
        assert any(not k["is_active"] for k in keys)

    def test_delete_removes_key(self, store):
        key = store.create_key(name="test")
        store.delete(key)
        with pytest.raises(KeyError):
            store.validate(key)

    def test_list_keys_masks_raw_key(self, store):
        store.create_key(name="masked-test")
        keys = store.list_keys()
        assert any("key_prefix" in k for k in keys)
        for k in keys:
            assert k["key_prefix"].endswith("...")

    def test_multiple_keys(self, store):
        store.create_key(name="key1", tier="free")
        store.create_key(name="key2", tier="standard")
        store.create_key(name="key3", tier="premium")
        assert len(store) == 3

    def test_tier_limits_on_key(self, store):
        free_key    = store.create_key(name="free", tier="free")
        premium_key = store.create_key(name="premium", tier="premium")
        assert store.validate(free_key).requests_per_minute < \
               store.validate(premium_key).requests_per_minute

    def test_invalid_tier_raises(self, store):
        with pytest.raises(ValueError, match="Unknown tier"):
            store.create_key(name="bad", tier="ultra_cosmic")

    def test_metadata_persisted(self, store):
        key = store.create_key(name="test", metadata={"org": "Greenfield Academy"})
        api_key = store.validate(key)
        assert api_key.metadata.get("org") == "Greenfield Academy"

    def test_health_check_passes(self, store):
        assert store.health_check() is True

    def test_create_key_multiple_and_validate_all(self, store):
        keys = [store.create_key(name=f"key-{i}") for i in range(5)]
        for key in keys:
            api_key = store.validate(key)
            assert api_key.is_active is True

    def test_len_matches_created_keys(self, store):
        assert len(store) == 0
        store.create_key(name="a")
        store.create_key(name="b")
        assert len(store) == 2

    def test_log_request_does_not_raise(self, store):
        key = store.create_key(name="test")
        store.log_request(key_id=key, endpoint="/recommend", response_ms=42, status_code=200)


# ── PostgresAPIKey dataclass tests ────────────────────────────────────────────

class TestPostgresAPIKey:
    def test_limits_from_tier(self):
        from plrs.api.db import PostgresAPIKey
        key = PostgresAPIKey(key_id="plrs_test", name="test", tier="standard", created_at=time.time())
        assert key.requests_per_minute == 60
        assert key.requests_per_day == 1000

    def test_free_tier_limits(self):
        from plrs.api.db import PostgresAPIKey
        key = PostgresAPIKey(key_id="plrs_test", name="test", tier="free", created_at=time.time())
        assert key.requests_per_minute == 10
        assert key.requests_per_day == 100

    def test_to_dict(self):
        from plrs.api.db import PostgresAPIKey
        key = PostgresAPIKey(key_id="plrs_test", name="test", tier="standard", created_at=time.time())
        d = key.to_dict()
        assert "limits" in d
        assert d["name"] == "test"


# ── RedisRateLimiter tests (fakeredis) ────────────────────────────────────────

class TestRedisRateLimiter:
    """Tests using fakeredis — same interface as real Redis."""

    @pytest.fixture
    def limiter(self):
        try:
            import fakeredis
        except ImportError:
            pytest.skip("fakeredis not installed — run: pip install fakeredis")

        import fakeredis
        from plrs.api.redis_rate_limit import RedisRateLimiter

        limiter = RedisRateLimiter.__new__(RedisRateLimiter)
        limiter._redis  = fakeredis.FakeRedis(decode_responses=True)
        limiter._prefix = "plrs:rl:"
        return limiter

    def test_first_request_allowed(self, limiter):
        r = limiter.check("key1", requests_per_minute=10, requests_per_day=100)
        assert r.allowed is True

    def test_within_limits_allowed(self, limiter):
        for _ in range(5):
            r = limiter.check("key2", requests_per_minute=10, requests_per_day=100)
            assert r.allowed is True

    def test_exceeds_minute_limit_blocked(self, limiter):
        for _ in range(3):
            limiter.check("key3", requests_per_minute=3, requests_per_day=1000)
        result = limiter.check("key3", requests_per_minute=3, requests_per_day=1000)
        assert result.allowed is False
        assert result.limit_type == "minute"

    def test_exceeds_day_limit_blocked(self, limiter):
        for _ in range(5):
            limiter.check("key4", requests_per_minute=1000, requests_per_day=5)
        result = limiter.check("key4", requests_per_minute=1000, requests_per_day=5)
        assert result.allowed is False
        assert result.limit_type == "day"

    def test_retry_after_positive(self, limiter):
        for _ in range(2):
            limiter.check("key5", requests_per_minute=2, requests_per_day=1000)
        result = limiter.check("key5", requests_per_minute=2, requests_per_day=1000)
        assert result.retry_after > 0

    def test_headers_present(self, limiter):
        result = limiter.check("key6", requests_per_minute=60, requests_per_day=1000)
        headers = result.headers
        assert "X-RateLimit-Limit-Minute" in headers
        assert "X-RateLimit-Remaining-Minute" in headers

    def test_result_to_dict_on_block(self, limiter):
        for _ in range(2):
            limiter.check("key7", requests_per_minute=2, requests_per_day=1000)
        result = limiter.check("key7", requests_per_minute=2, requests_per_day=1000)
        d = result.to_dict()
        assert "error" in d
        assert "retry_after_s" in d

    def test_reset_clears_state(self, limiter):
        for _ in range(3):
            limiter.check("key8", requests_per_minute=3, requests_per_day=1000)
        r = limiter.check("key8", requests_per_minute=3, requests_per_day=1000)
        assert r.allowed is False
        limiter.reset("key8")
        r2 = limiter.check("key8", requests_per_minute=3, requests_per_day=1000)
        assert r2.allowed is True

    def test_stats_returns_counts(self, limiter):
        limiter.check("key9", requests_per_minute=60, requests_per_day=1000)
        limiter.check("key9", requests_per_minute=60, requests_per_day=1000)
        stats = limiter.stats("key9")
        assert stats["requests_minute"] == 2
        assert stats["requests_day"] == 2

    def test_peek_no_record(self, limiter):
        limiter.check("key10", requests_per_minute=2, requests_per_day=100, record=False)
        limiter.check("key10", requests_per_minute=2, requests_per_day=100, record=False)
        r = limiter.check("key10", requests_per_minute=2, requests_per_day=100, record=True)
        assert r.requests_minute == 1

    def test_different_keys_independent(self, limiter):
        for _ in range(3):
            limiter.check("keyA", requests_per_minute=3, requests_per_day=100)
        r_blocked = limiter.check("keyA", requests_per_minute=3, requests_per_day=100)
        r_allowed = limiter.check("keyB", requests_per_minute=3, requests_per_day=100)
        assert r_blocked.allowed is False
        assert r_allowed.allowed is True


# ── Factory function tests ────────────────────────────────────────────────────

class TestFactories:
    def test_get_db_key_store_no_env_returns_postgres_sqlite(self, monkeypatch):
        """Without PLRS_DB_URL, falls back — we test with SQLite."""
        monkeypatch.setenv("PLRS_DB_URL", "sqlite:///:memory:")
        from plrs.api.db import get_db_key_store, PostgresKeyStore
        store = get_db_key_store()
        assert isinstance(store, PostgresKeyStore)

    def test_get_db_key_store_with_sqlite_url(self, monkeypatch):
        monkeypatch.setenv("PLRS_DB_URL", "sqlite:///:memory:")
        from plrs.api.db import get_db_key_store, PostgresKeyStore
        store = get_db_key_store()
        assert isinstance(store, PostgresKeyStore)

    def test_get_rate_limiter_redis_url_returns_redis(self, monkeypatch):
        """With PLRS_REDIS_URL set, returns RedisRateLimiter."""
        monkeypatch.setenv("PLRS_REDIS_URL", "redis://localhost:6379/0")
        from plrs.api.redis_rate_limit import get_rate_limiter, RedisRateLimiter
        limiter = get_rate_limiter()
        assert isinstance(limiter, RedisRateLimiter)
