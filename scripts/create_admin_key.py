#!/usr/bin/env python3
"""Create a new admin API key. Run from CServe root.

Usage:
  python scripts/create_admin_key.py [--db-path /var/lib/cserve/events.db]
"""
from __future__ import annotations

import argparse
import asyncio
import sys

# Add project root to path
sys.path.insert(0, ".")

from cserve.control_plane.db import EventLog


async def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new admin API key")
    parser.add_argument(
        "--db-path",
        default="/var/lib/cserve/events.db",
        help="Path to SQLite event log (default: /var/lib/cserve/events.db)",
    )
    args = parser.parse_args()

    db = EventLog(args.db_path)
    await db.open()
    try:
        raw_key, api_key = await db.create_api_key(
            user_id="admin",
            name="admin-key",
            role="admin",
        )
        print(f"\n{'='*60}")
        print("  ADMIN API KEY (store securely!)")
        print(f"  {raw_key}")
        print(f"{'='*60}\n")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
