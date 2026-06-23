#!/usr/bin/env python3
"""Keep gpt-oss-120b warm with periodic requests. Run in background.

Usage:
  python scripts/keep_gpt_oss_warm.py [--interval 45]

Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

API = "http://cosmos-9.ddns.ualr.edu:8002/v1/chat/completions"
KEY = os.environ.get("CSERVE_ADMIN_KEY", "")
MODEL = "gpt-oss-120b"


def req() -> tuple[str, str]:
    try:
        r = urllib.request.Request(
            API,
            method="POST",
            headers={
                "Authorization": f"Bearer {KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 20,
                }
            ).encode(),
        )
        with urllib.request.urlopen(r, timeout=120) as resp:
            d = json.loads(resp.read())
            content = (
                d.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            return "OK", content[:80]
    except Exception as e:
        return "ERR", str(e)[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep gpt-oss-120b warm")
    parser.add_argument(
        "--interval",
        type=int,
        default=45,
        help="Seconds between requests (default: 45)",
    )
    args = parser.parse_args()

    print(f"Keeping gpt-oss-120b warm — request every {args.interval}s (Ctrl+C to stop)")
    print("-" * 60)
    n = 0
    try:
        while True:
            n += 1
            ts = time.strftime("%H:%M:%S")
            status, msg = req()
            print(f"[{ts}] #{n} {status}: {msg}")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
