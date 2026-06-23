#!/usr/bin/env bash
# Control plane wrapper for the services production account (cosmos-9 head).
set -euo pipefail
cd /home/services/CServe
ulimit -n 1048576 2>/dev/null || true
exec /home/services/cserve-venv/bin/python3 -m cserve.control_plane.server \
  --cluster-config /home/services/CServe/configs/cluster.yaml \
  --models-config /home/services/CServe/configs/models.yaml \
  --db-path /var/lib/cserve/events.db \
  --port 8002
