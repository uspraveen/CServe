#!/usr/bin/env python3
"""Merge SQLite ``ui_runtime_tuning`` into YAML files (offline export).

YAML remains the catalog; the dashboard persists tunables to SQLite.
Use this script to write the merged view back to disk for Git backup
after UI edits, without running the control plane.

Usage (from CServe repo root):
  python scripts/export_ui_tuning_to_yaml.py \\
    --cluster configs/cluster.yaml --models configs/models.yaml \\
    [--db-path /var/lib/cserve/events.db]
"""
from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, ".")

from cserve.common.config import (
    load_cluster_config,
    load_models_config,
    save_cluster_config,
    save_models_config_disk,
)
from cserve.common.ui_tuning import (
    apply_model_tuning_payload,
    apply_safety_payload,
    decode_payload_json,
)
from cserve.control_plane.db import EventLog


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export merged UI tunables (YAML + SQLite) to YAML files",
    )
    parser.add_argument("--cluster", required=True, help="Path to cluster.yaml")
    parser.add_argument("--models", required=True, help="Path to models.yaml")
    parser.add_argument(
        "--db-path",
        default="/var/lib/cserve/events.db",
        help="SQLite DB with ui_runtime_tuning (default: /var/lib/cserve/events.db)",
    )
    args = parser.parse_args()

    cluster = load_cluster_config(args.cluster)
    _, models = load_models_config(args.models)

    db = EventLog(args.db_path)
    await db.open()
    try:
        rows = await db.get_all_ui_runtime_tuning()
    finally:
        await db.close()

    for row in rows:
        scope = row["scope"]
        payload = decode_payload_json(row["payload_json"])
        if scope == "safety":
            cluster.safety = apply_safety_payload(cluster.safety, payload)
        elif scope == "model":
            name = row["model_name"]
            cfg = models.get(name)
            if cfg is not None:
                models[name] = apply_model_tuning_payload(cfg, payload)

    save_cluster_config(cluster, args.cluster)
    save_models_config_disk(args.models, models)
    print(f"Wrote merged config to {args.cluster} and {args.models} ({len(rows)} overlay row(s)).")


if __name__ == "__main__":
    asyncio.run(main())
