#!/usr/bin/env bash
set -euo pipefail

CONTROL_PLANE_URL="http://cosmos-9.ddns.ualr.edu:8002"
AGENT_PORT=50051
CSERVE_SRC="/home/praveen/CServe"

declare -A NODE_HOSTS=(
  [cosmos-7]="144.167.35.154"
  [cosmos-8]="cosmos-8.ddns.ualr.edu"
  [cosmos-10]="cosmos-10.ddns.ualr.edu"
  [cosmos-11]="cosmos-11.ddns.ualr.edu"
  [cosmos-12]="cosmos-12.ddns.ualr.edu"
  [cosmos-13]="cosmos-13.ddns.ualr.edu"
)

declare -A NODE_IPS=(
  [cosmos-7]="144.167.35.154"
  [cosmos-8]="cosmos-8.ddns.ualr.edu"
  [cosmos-10]="cosmos-10.ddns.ualr.edu"
  [cosmos-11]="cosmos-11.ddns.ualr.edu"
  [cosmos-12]="cosmos-12.ddns.ualr.edu"
  [cosmos-13]="cosmos-13.ddns.ualr.edu"
)

declare -A NODE_CUDA_DEVICES=(
  [cosmos-7]="0,1,2,3,4,5,6,7"
  [cosmos-8]="0,1,2,3,4,5,6,7"
  [cosmos-10]="0,1,2,3"
  [cosmos-11]="0,1,2,3"
  [cosmos-12]="0,1,2,3"
  [cosmos-13]="0,1,2,3"
)

PIP="~/miniconda3/bin/pip"
PYTHON="~/miniconda3/bin/python3"

echo "========================================"
echo " CServe Node Agent Deployment"
echo "========================================"

for name in "${!NODE_HOSTS[@]}"; do
  host="${NODE_HOSTS[$name]}"
  cuda="${NODE_CUDA_DEVICES[$name]:-}"
  echo ""
  echo "──── Deploying to $name ($host) cuda=$cuda ────"

  echo "  [1/5] Syncing CServe source..."
  rsync -az --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'node_modules' \
    --exclude '.ruff_cache' \
    --exclude 'dashboard-ui/node_modules' \
    "${CSERVE_SRC}/" "praveen@${host}:~/CServe/"

  echo "  [2/5] Installing CServe package..."
  ssh praveen@"$host" "$PIP install -e ~/CServe/ -q 2>&1 | tail -3"

  echo "  [3/5] Checking vLLM..."
  HAS_VLLM=$(ssh praveen@"$host" "$PIP list 2>/dev/null | grep -c vllm" || echo "0")
  if [ "$HAS_VLLM" = "0" ]; then
    echo "  ⚠  vLLM not found — installing (this may take a few minutes)..."
    ssh praveen@"$host" "$PIP install vllm -q 2>&1 | tail -3"
  else
    echo "  ✓ vLLM already installed"
  fi

  echo "  [4/5] Stopping existing agent..."
  # [c]serve pattern avoids pkill matching this SSH bash -c line (which used to SIGKILL the session).
  ssh -o ConnectTimeout=15 praveen@"$host" \
    "timeout 12 bash -c 'pkill -f \"[m]cserve.node_agent.server\" 2>/dev/null; fuser -k $AGENT_PORT/tcp 2>/dev/null; true'" \
    || true

  echo "  [5/5] Starting agent (nohup)..."
  ssh -o ConnectTimeout=15 -f praveen@"$host" "cd ~/CServe && CSERVE_CUDA_DEVICES='$cuda' nohup $PYTHON -m cserve.node_agent.server \
    --node-name $name \
    --node-host $host \
    --control-plane $CONTROL_PLANE_URL \
    --port $AGENT_PORT \
    --transport http \
    > ~/cserve-agent.log 2>&1 &"

  echo "  ✓ $name deployed"
done

echo ""
echo "========================================"
echo " Waiting for agents to start..."
echo "========================================"
sleep 5

ok=0; fail=0
for name in "${!NODE_HOSTS[@]}"; do
  host="${NODE_HOSTS[$name]}"
  if curl -s --connect-timeout 3 "http://$host:$AGENT_PORT/node_status" > /dev/null 2>&1; then
    echo "  ✓ $name — UP"
    ok=$((ok+1))
  else
    echo "  ✗ $name — DOWN"
    fail=$((fail+1))
  fi
done

echo ""
echo "========================================"
echo " Deployment complete: $ok up, $fail down"
echo "========================================"
