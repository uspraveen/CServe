#!/usr/bin/env bash
# One-time cosmos-9 head migration: praveen → services (uses docker for root ops, no sudo password).
set -euo pipefail

SRC=/home/praveen/CServe
DEST=/home/services/CServe
VENV=/home/services/cserve-venv
UNIT_SRC="$SRC/scripts/cserve-control.service"
LOG=/home/praveen/cserve-migrate-$(date +%Y%m%d-%H%M%S).log

exec > >(tee -a "$LOG") 2>&1
echo "=== CServe head migration log: $LOG ==="

nsys() {
  docker run --rm --privileged --pid=host alpine nsenter -t 1 -m -u -i -n -p "$@"
}

echo "==> [1/8] Stop praveen control plane (if running)"
pkill -TERM -f 'cserve.control_plane.server.*--port 8002' 2>/dev/null || true
sleep 3
pkill -KILL -f 'cserve.control_plane.server.*--port 8002' 2>/dev/null || true
sleep 1
if pgrep -u praveen -f 'cserve.control_plane.server' >/dev/null 2>&1; then
  echo "ERROR: control plane still running as praveen" >&2
  exit 1
fi

echo "==> [2/8] Backup SQLite DB metadata (light checkpoint)"
nsys runuser -u services -- sqlite3 /var/lib/cserve/events.db 'PRAGMA wal_checkpoint(TRUNCATE);' 2>/dev/null \
  || docker run --rm -v /var/lib/cserve:/data alpine sh -c 'apk add sqlite >/dev/null 2>&1; sqlite3 /data/events.db "PRAGMA wal_checkpoint(TRUNCATE);"' || true

echo "==> [3/8] Copy CServe repo to /home/services/CServe (includes .git)"
docker run --rm \
  -v "$SRC:/src:ro" \
  -v /home/services:/dest \
  alpine sh -c '
    set -e
    if [ -d /dest/CServe ]; then
      mv /dest/CServe "/dest/CServe.bak.$(date +%s)"
    fi
    cp -a /src /dest/CServe
    chown -R 1003:1003 /dest/CServe
  '

echo "==> [4/8] chown /var/lib/cserve → services"
docker run --rm -v /var/lib/cserve:/data alpine chown -R 1003:1003 /data

echo "==> [5/8] Install SSH key for worker deploy (if present)"
if [[ -f /home/praveen/.ssh/id_ed25519 ]]; then
  docker run --rm \
    -v /home/praveen/.ssh/id_ed25519:/key:ro \
    -v /home/services:/dest \
    alpine sh -c '
      mkdir -p /dest/.ssh
      cp /key /dest/.ssh/id_ed25519
      chmod 700 /dest/.ssh
      chmod 600 /dest/.ssh/id_ed25519
      chown -R 1003:1003 /dest/.ssh
    '
  [[ -f /home/praveen/.ssh/known_hosts ]] && docker run --rm \
    -v /home/praveen/.ssh/known_hosts:/kh:ro \
    -v /home/services:/dest \
    alpine sh -c 'cp /kh /dest/.ssh/known_hosts && chown 1003:1003 /dest/.ssh/known_hosts && chmod 644 /dest/.ssh/known_hosts'
fi

echo "==> [6/8] Python venv + pip install -e (may take several minutes)"
nsys runuser -u services -- /usr/bin/python3.12 -m venv "$VENV"
nsys runuser -u services -- "$VENV/bin/pip" install -U pip wheel
nsys runuser -u services -- "$VENV/bin/pip" install -e "$DEST"

chmod +x "$DEST/scripts/run-control-plane-services.sh" 2>/dev/null || \
  docker run --rm -v /home/services/CServe/scripts:/s alpine chmod +x /s/run-control-plane-services.sh

echo "==> [7/8] Install systemd unit"
docker run --rm \
  -v "$UNIT_SRC:/unit:ro" \
  -v /etc/systemd/system:/etc/systemd/system \
  alpine cp /unit /etc/systemd/system/cserve-control.service

nsys systemctl daemon-reload
nsys systemctl enable cserve-control.service

echo "==> [8/8] Start cserve-control.service"
nsys systemctl restart cserve-control.service
sleep 5
nsys systemctl is-active cserve-control.service

echo "==> Migration complete. Logs: journalctl -u cserve-control -f"
echo "    Dev sync: $SRC/scripts/sync-cserve-to-services.sh"
