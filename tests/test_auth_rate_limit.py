"""
tests/test_auth_rate_limit.py
==============================
Tests for API key auth and rate limiting.
Uses FastAPI dependency overrides for clean isolation — no module reloading.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
MAPS = ROOT / "data" / "knowledge_maps"


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def math_pipeline():
    from plrs.curriculum.loader import load_dag
    from plrs.pipeline import PLRSPipeline
    return PLRSPipeline(load_dag(MAPS / "math_dag.json"))


@pytest.fixture
def key_store():
    from plrs.api.auth import KeyStore
    return KeyStore()


@pytest.fixture
def limiter():
    from plrs.api.rate_limit import RateLimiter
    return RateLimiter()


@pytest.fixture
def client(key_store, limiter, math_pipeline):
    """
    Test client with auth enabled via FastAPI dependency overrides.
    Overrides get_key_store and get_limiter to inject fresh instances.
    Also patches DEV_MODE to False.
    """
    import sys; app_module = sys.modules["plrs.api.app"]
    from plrs.api.app import app, register_pipeline
    from plrs.api.auth import get_key_store
    from plrs.api.rate_limit import get_limiter

    # Register pipeline
    register_pipeline("math", math_pipeline)

    # Inject fresh instances via FastAPI dependency override
    app.dependency_overrides[get_key_store] = lambda: key_store
    app.dependency_overrides[get_limiter]   = lambda: limiter

    # Force auth on (DEV_MODE=False)
    original_dev_mode = app_module.DEV_MODE
    app_module.DEV_MODE = False

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    # Cleanup
    app.dependency_overrides.clear()
    app_module.DEV_MODE = original_dev_mode


@pytest.fixture
def valid_key(key_store):
    return key_store.create_key(name="test-app", tier="standard")


@pytest.fixture
def internal_key(key_store):
    return key_store.create_key(name="admin", tier="internal")


# ── KeyStore unit tests ───────────────────────────────────────────────────────

class TestKeyStore:
    def test_create_key_returns_plrs_prefix(self, key_store):
        key = key_store.create_key(name="test")
        assert key.startswith("plrs_")
        assert len(key) > 10

    def test_validate_valid_key(self, key_store):
        key = key_store.create_key(name="test")
        api_key = key_store.validate(key)
        assert api_key.name == "test"
        assert api_key.is_active is True

    def test_validate_invalid_key_raises(self, key_store):
        with pytest.raises(KeyError):
            key_store.validate("plrs_doesnotexist1234567890abcdef")

    def test_revoke_deactivates_key(self, key_store):
        key = key_store.create_key(name="test")
        key_store.revoke(key)
        with pytest.raises(ValueError, match="inactive"):
            key_store.validate(key)

    def test_delete_removes_key(self, key_store):
        key = key_store.create_key(name="test")
        key_store.delete(key)
        with pytest.raises(KeyError):
            key_store.validate(key)

    def test_list_keys_masks_raw_key(self, key_store):
        key_store.create_key(name="masked-test")
        keys = key_store.list_keys()
        for k in keys:
            assert "key_prefix" in k
            assert k["key_prefix"].endswith("...")
            assert "key_id" not in k

    def test_tier_limits_applied(self, key_store):
        free_key = key_store.create_key(name="free", tier="free")
        prem_key = key_store.create_key(name="premium", tier="premium")
        assert key_store.validate(free_key).requests_per_minute < \
               key_store.validate(prem_key).requests_per_minute

    def test_invalid_tier_raises(self, key_store):
        with pytest.raises(ValueError, match="Unknown tier"):
            key_store.create_key(name="bad", tier="galaxy_brain")

    def test_persist_and_reload(self, tmp_path):
        from plrs.api.auth import KeyStore
        path = tmp_path / "keys.json"
        store1 = KeyStore(persist_path=path)
        key = store1.create_key(name="persist-test")

        store2 = KeyStore(persist_path=path)
        api_key = store2.validate(key)
        assert api_key.name == "persist-test"


# ── RateLimiter unit tests ────────────────────────────────────────────────────

class TestRateLimiter:
    def test_first_request_allowed(self, limiter):
        result = limiter.check("test_key", requests_per_minute=10, requests_per_day=100)
        assert result.allowed is True

    def test_within_limits_allowed(self, limiter):
        for _ in range(5):
            r = limiter.check("key1", requests_per_minute=10, requests_per_day=100)
            assert r.allowed

    def test_exceeds_minute_limit_blocked(self, limiter):
        for _ in range(3):
            limiter.check("key2", requests_per_minute=3, requests_per_day=1000)
        result = limiter.check("key2", requests_per_minute=3, requests_per_day=1000)
        assert result.allowed is False
        assert result.limit_type == "minute"

    def test_exceeds_day_limit_blocked(self, limiter):
        for _ in range(5):
            limiter.check("key3", requests_per_minute=1000, requests_per_day=5)
        result = limiter.check("key3", requests_per_minute=1000, requests_per_day=5)
        assert result.allowed is False
        assert result.limit_type == "day"

    def test_retry_after_positive_when_blocked(self, limiter):
        for _ in range(2):
            limiter.check("key4", requests_per_minute=2, requests_per_day=1000)
        result = limiter.check("key4", requests_per_minute=2, requests_per_day=1000)
        assert result.retry_after > 0

    def test_headers_present(self, limiter):
        result = limiter.check("key5", requests_per_minute=60, requests_per_day=1000)
        headers = result.headers
        assert "X-RateLimit-Limit-Minute" in headers
        assert "X-RateLimit-Limit-Day" in headers
        assert "X-RateLimit-Remaining-Minute" in headers

    def test_reset_clears_window(self, limiter):
        for _ in range(3):
            limiter.check("key6", requests_per_minute=3, requests_per_day=1000)
        r = limiter.check("key6", requests_per_minute=3, requests_per_day=1000)
        assert r.allowed is False
        limiter.reset("key6")
        r2 = limiter.check("key6", requests_per_minute=3, requests_per_day=1000)
        assert r2.allowed is True

    def test_stats(self, limiter):
        limiter.check("key7", requests_per_minute=60, requests_per_day=1000)
        limiter.check("key7", requests_per_minute=60, requests_per_day=1000)
        stats = limiter.stats("key7")
        assert stats["requests_minute"] == 2
        assert stats["requests_day"] == 2

    def test_peek_does_not_consume(self, limiter):
        limiter.check("key8", requests_per_minute=2, requests_per_day=100, record=False)
        limiter.check("key8", requests_per_minute=2, requests_per_day=100, record=False)
        r = limiter.check("key8", requests_per_minute=2, requests_per_day=100, record=True)
        assert r.requests_minute == 1


# ── API endpoint auth tests ───────────────────────────────────────────────────

class TestAPIAuth:
    def test_health_no_auth_required(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["auth_enabled"] is True

    def test_root_no_auth_required(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_recommend_no_key_returns_401(self, client):
        r = client.post("/recommend", json={"domain": "math", "mastery_scores": {}})
        assert r.status_code == 401

    def test_recommend_invalid_key_returns_403(self, client):
        r = client.post(
            "/recommend",
            json={"domain": "math", "mastery_scores": {}},
            headers={"X-API-Key": "plrs_invalid000000000000000000000000"},
        )
        assert r.status_code == 403

    def test_recommend_valid_key_succeeds(self, client, valid_key):
        r = client.post(
            "/recommend",
            json={"domain": "math", "mastery_scores": {"whole_numbers": 0.9}},
            headers={"X-API-Key": valid_key},
        )
        assert r.status_code == 200
        assert "approved" in r.json()

    def test_rate_limit_headers_present(self, client, valid_key):
        r = client.post(
            "/recommend",
            json={"domain": "math", "mastery_scores": {}},
            headers={"X-API-Key": valid_key},
        )
        assert r.status_code == 200
        assert "x-ratelimit-limit-minute" in r.headers

    def test_revoked_key_returns_403(self, client, key_store, valid_key):
        key_store.revoke(valid_key)
        r = client.post(
            "/recommend",
            json={"domain": "math", "mastery_scores": {}},
            headers={"X-API-Key": valid_key},
        )
        assert r.status_code == 403

    def test_rate_limit_exceeded_returns_429(self, client, key_store, limiter):
        tiny_key = key_store.create_key(name="tiny", tier="free")
        # Pre-fill the window to the limit
        for _ in range(10):
            limiter.check(tiny_key, requests_per_minute=10, requests_per_day=100)
        r = client.post(
            "/recommend",
            json={"domain": "math", "mastery_scores": {}},
            headers={"X-API-Key": tiny_key},
        )
        assert r.status_code == 429
        assert "retry_after_s" in r.json()["detail"]

    def test_usage_endpoint(self, client, valid_key):
        r = client.get("/usage", headers={"X-API-Key": valid_key})
        assert r.status_code == 200
        data = r.json()
        assert "requests_minute" in data
        assert "remaining_day" in data
        assert data["tier"] == "standard"


class TestAdminEndpoints:
    def test_list_keys_requires_internal_tier(self, client, valid_key):
        r = client.get("/admin/keys", headers={"X-API-Key": valid_key})
        assert r.status_code == 403

    def test_list_keys_with_internal_tier(self, client, internal_key):
        r = client.get("/admin/keys", headers={"X-API-Key": internal_key})
        assert r.status_code == 200
        assert "keys" in r.json()

    def test_create_key_via_api(self, client, internal_key):
        r = client.post(
            "/admin/keys",
            json={"name": "api-created", "tier": "free"},
            headers={"X-API-Key": internal_key},
        )
        assert r.status_code == 201
        data = r.json()
        assert data["key"].startswith("plrs_")
        assert "warning" in data

    def test_revoke_key_via_api(self, client, internal_key, key_store):
        target = key_store.create_key(name="to-revoke")
        r = client.delete(
            f"/admin/keys/{target}",
            headers={"X-API-Key": internal_key},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "revoked"

    def test_revoke_nonexistent_returns_404(self, client, internal_key):
        r = client.delete(
            "/admin/keys/plrs_doesnotexist1234567890abcdef",
            headers={"X-API-Key": internal_key},
        )
        assert r.status_code == 404
