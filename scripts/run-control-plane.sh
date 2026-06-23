#!/usr/bin/env bash
# DEPRECATED on cosmos-9: production runs as user 'services' via systemd (cserve-control.service).
# Use scripts/run-control-plane-services.sh or: sudo systemctl restart cserve-control
set -euo pipefail
echo "WARN: use services account / systemctl cserve-control on cosmos-9" >&2
cd /home/praveen/CServe
ulimit -n 1048576 2>/dev/null || true
exec /home/praveen/miniconda3/bin/python3 -m cserve.control_plane.server \
  --cluster-config /home/praveen/CServe/configs/cluster.yaml \
  --models-config /home/praveen/CServe/configs/models.yaml \
  --db-path /var/lib/cserve/events.db \
  --port 8002
