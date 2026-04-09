"""
scripts/manage_keys.py
======================
CLI for managing PLRS API keys.

Usage:
    python scripts/manage_keys.py create --name "My App" --tier standard
    python scripts/manage_keys.py list
    python scripts/manage_keys.py revoke plrs_abc123...
    python scripts/manage_keys.py tiers
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEYS_PATH = ROOT / ".plrs_keys.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manage PLRS API keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage_keys.py create --name "Greenfield Academy" --tier standard
  python scripts/manage_keys.py create --name "Internal CI" --tier internal
  python scripts/manage_keys.py list
  python scripts/manage_keys.py revoke plrs_a3f8c2e1d4b7a9f0...
  python scripts/manage_keys.py tiers
        """,
    )
    p.add_argument(
        "--keys-file",
        default=os.getenv("PLRS_KEYS_PATH", str(DEFAULT_KEYS_PATH)),
        help="Path to keys JSON file (default: .plrs_keys.json)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # create
    c = sub.add_parser("create", help="Create a new API key")
    c.add_argument("--name", required=True, help="Human label for the key")
    c.add_argument("--tier", default="standard",
                   choices=["free", "standard", "premium", "internal"],
                   help="Rate limit tier (default: standard)")
    c.add_argument("--meta", default="{}", help="JSON metadata string")

    # list
    sub.add_parser("list", help="List all keys")

    # revoke
    r = sub.add_parser("revoke", help="Revoke a key")
    r.add_argument("key", help="Full raw API key to revoke")

    # delete
    d = sub.add_parser("delete", help="Permanently delete a key")
    d.add_argument("key", help="Full raw API key to delete")

    # tiers
    sub.add_parser("tiers", help="Show available tiers and their limits")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    from plrs.api.auth import TIERS, KeyStore

    store = KeyStore(persist_path=args.keys_file)

    if args.command == "create":
        try:
            metadata = json.loads(args.meta)
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON metadata: {args.meta}")
            return

        raw_key = store.create_key(name=args.name, tier=args.tier, metadata=metadata)
        api_key = store.validate(raw_key)

        print(f"\n✅ API key created")
        print(f"   Name  : {api_key.name}")
        print(f"   Tier  : {api_key.tier}")
        print(f"   Limits: {api_key.requests_per_minute} req/min · {api_key.requests_per_day} req/day")
        print(f"\n   Key   : {raw_key}")
        print(f"\n   ⚠️  Store this key safely — it will not be shown again.")
        print(f"   File  : {args.keys_file}\n")

    elif args.command == "list":
        keys = store.list_keys()
        if not keys:
            print("No keys found.")
            return

        print(f"\n{'Name':<25} {'Tier':<12} {'Active':<8} {'Req/min':<10} {'Req/day':<10} {'Key prefix'}")
        print("-" * 90)
        for k in keys:
            active = "✅" if k["is_active"] else "❌"
            print(
                f"{k['name']:<25} {k['tier']:<12} {active:<8} "
                f"{k['limits']['requests_per_minute']:<10} "
                f"{k['limits']['requests_per_day']:<10} "
                f"{k['key_prefix']}"
            )
        print(f"\nTotal: {len(keys)} key(s)  |  File: {args.keys_file}\n")

    elif args.command == "revoke":
        try:
            store.revoke(args.key)
            print(f"✅ Key revoked: {args.key[:16]}...")
        except KeyError:
            print(f"❌ Key not found: {args.key[:16]}...")

    elif args.command == "delete":
        confirm = input(f"Permanently delete key {args.key[:16]}...? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return
        try:
            store.delete(args.key)
            print(f"✅ Key deleted: {args.key[:16]}...")
        except KeyError:
            print(f"❌ Key not found.")

    elif args.command == "tiers":
        print("\nAvailable tiers:\n")
        print(f"{'Tier':<12} {'Req/min':<12} {'Req/day'}")
        print("-" * 36)
        for tier, limits in TIERS.items():
            print(f"{tier:<12} {limits['requests_per_minute']:<12} {limits['requests_per_day']:,}")
        print()


if __name__ == "__main__":
    main()
