"""
scripts/docker_init.py
======================
First-run initialisation for the PLRS Docker stack.

Runs automatically on container startup (see Dockerfile CMD).
Safe to run multiple times — idempotent.

Actions:
  1. Wait for PostgreSQL to be ready
  2. Create database tables (api_keys, usage_logs)
  3. Create an internal admin key if none exists
  4. Verify Redis is reachable
  5. Print connection summary

Usage:
    python scripts/docker_init.py
    python scripts/docker_init.py --skip-redis
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def wait_for_postgres(db_url: str, max_retries: int = 30, delay: float = 2.0) -> bool:
    """Wait until PostgreSQL is accepting connections."""
    print("⏳ Waiting for PostgreSQL...")
    for attempt in range(1, max_retries + 1):
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"✅ PostgreSQL ready (attempt {attempt})")
            return True
        except Exception as e:
            if attempt < max_retries:
                print(f"   attempt {attempt}/{max_retries} — {e}")
                time.sleep(delay)
            else:
                print(f"❌ PostgreSQL not reachable after {max_retries} attempts")
                return False
    return False


def wait_for_redis(redis_url: str, max_retries: int = 15, delay: float = 2.0) -> bool:
    """Wait until Redis is accepting connections."""
    print("⏳ Waiting for Redis...")
    for attempt in range(1, max_retries + 1):
        try:
            import redis
            r = redis.from_url(redis_url, socket_connect_timeout=2)
            r.ping()
            print(f"✅ Redis ready (attempt {attempt})")
            return True
        except Exception as e:
            if attempt < max_retries:
                print(f"   attempt {attempt}/{max_retries} — {e}")
                time.sleep(delay)
            else:
                print(f"❌ Redis not reachable after {max_retries} attempts")
                return False
    return False


def init_database(db_url: str) -> str | None:
    """
    Create tables and seed admin key.
    Returns the raw admin key if newly created, None if already exists.
    """
    from plrs.api.db import Base, APIKeyRecord, PostgresKeyStore
    from sqlalchemy import create_engine

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    print("✅ Database tables created (or already exist)")

    store = PostgresKeyStore(db_url)

    # Check if any internal key exists
    existing = [k for k in store.list_keys() if "internal" in k.get("tier", "")]
    if existing:
        print(f"✅ Admin key already exists ({existing[0]['key_prefix']})")
        return None

    # Create the first internal admin key
    raw_key = store.create_key(
        name="admin",
        tier="internal",
        metadata={"created_by": "docker_init"},
    )
    print(f"\n🔑 Admin API key created (store this safely):")
    print(f"   {raw_key}")
    print(f"\n   Set as environment variable:")
    print(f"   PLRS_ADMIN_KEY={raw_key}\n")
    return raw_key


def main() -> None:
    parser = argparse.ArgumentParser(description="PLRS Docker initialisation")
    parser.add_argument("--skip-redis", action="store_true")
    parser.add_argument("--skip-postgres", action="store_true")
    args = parser.parse_args()

    db_url    = os.getenv("PLRS_DB_URL")
    redis_url = os.getenv("PLRS_REDIS_URL", "redis://redis:6379/0")

    print("\n🧠 PLRS Docker Init")
    print("=" * 40)

    success = True

    # PostgreSQL
    if not args.skip_postgres:
        if not db_url:
            print("❌ PLRS_DB_URL not set — cannot init database")
            sys.exit(1)

        if not wait_for_postgres(db_url):
            sys.exit(1)

        admin_key = init_database(db_url)
        if admin_key:
            # Write to file for docker-compose to pick up
            key_file = "/tmp/plrs_admin_key.txt"
            with open(key_file, "w") as f:
                f.write(admin_key)
            print(f"   (also written to {key_file})")

    # Redis
    if not args.skip_redis:
        if not wait_for_redis(redis_url):
            print("⚠️  Redis not reachable — rate limiting will use in-memory fallback")
            success = False

    print("\n" + "=" * 40)
    if success:
        print("✅ PLRS initialised successfully")
    else:
        print("⚠️  PLRS initialised with warnings")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
