#!/usr/bin/env bash
# Sync dev tree (praveen) → production tree (services) on cosmos-9.
# Both keep the same git remote; commit/push from praveen, then run this before restart.
set -euo pipefail

SRC=/home/praveen/CServe
DEST=/home/services/CServe
EXCLUDES=(
  --exclude '__pycache__'
  --exclude '.pytest_cache'
  --exclude 'node_modules'
  --exclude 'cserve-venv'
  --exclude '.venv'
)

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: missing $SRC" >&2
  exit 1
fi

echo "==> rsync $SRC → $DEST (via docker, preserves .git)"
docker run --rm \
  -v "$SRC:/src:ro" \
  -v /home/services:/dest \
  alpine sh -c '
    set -e
    mkdir -p /dest/CServe
    apk add --no-cache rsync >/dev/null
    rsync -a \
      --exclude __pycache__ \
      --exclude .pytest_cache \
      --exclude node_modules \
      --exclude cserve-venv \
      --exclude .venv \
      /src/ /dest/CServe/
    chown -R 1003:1003 /dest/CServe
  '

echo "==> done. Restart with: docker nsenter systemctl restart cserve-control"
echo "    Or on services account: systemctl --user is not used; use root/systemctl."
