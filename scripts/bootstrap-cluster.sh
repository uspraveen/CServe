#!/usr/bin/env bash
# Prepare L40 nodes and restart CServe with deployment precedence.
set -euo pipefail

CSERVE_ROOT="/home/praveen/CServe"
KEY=$(grep -o 'csk_[a-f0-9]*' "$CSERVE_ROOT/scripts/keep_gpt_oss_warm.py" | head -1)
API="http://127.0.0.1:8002"

echo "==> Pausing competing L40 models (qwen3-vl, gpt-oss) for gemma4 slot..."
curl -sf -X PUT "$API/admin/config" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"models":{"qwen3-vl-32b":{"autoscaling":{"min_replicas":0,"max_replicas":0}},"gpt-oss-120b":{"autoscaling":{"min_replicas":0,"max_replicas":0}}}}' \
  | python3 -m json.tool

echo "==> Waiting for replicas to drain..."
for i in $(seq 1 24); do
  n=$(curl -sf "$API/internal/registry" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(sum(1 for r in d['replicas'] if r['model'] in ('qwen3-vl-32b','gpt-oss-120b')))
")
  echo "  qwen/gpt replicas: $n"
  [[ "$n" == "0" ]] && break
  sleep 10
done

echo "==> Scrubbing vLLM on L40 workers..."
for host in cosmos-10.ddns.ualr.edu cosmos-11.ddns.ualr.edu cosmos-12.ddns.ualr.edu cosmos-13.ddns.ualr.edu; do
  ssh -o ConnectTimeout=6 praveen@"$host" \
    "pkill -f 'vllm.entrypoints' 2>/dev/null; pkill -f 'gemma-4-31B' 2>/dev/null; pkill -f 'Qwen3-VL' 2>/dev/null; pkill -f 'gpt-oss' 2>/dev/null; true" \
    && echo "  scrubbed $host" || echo "  skip $host"
done
sleep 5

echo "==> Upgrading vLLM 0.19 + transformers 5.5 on L40 nodes (for Gemma 4)..."
for host in cosmos-10.ddns.ualr.edu cosmos-11.ddns.ualr.edu cosmos-12.ddns.ualr.edu cosmos-13.ddns.ualr.edu; do
  ssh -o ConnectTimeout=8 praveen@"$host" \
    "/home/praveen/miniconda3/bin/pip install -q 'vllm==0.19.0' && \
     /home/praveen/miniconda3/bin/pip install -q 'transformers==5.5.0' 2>&1 | tail -2" \
    && echo "  upgraded $host" || echo "  upgrade failed $host"
done

echo "==> Restarting control plane..."
pkill -f 'cserve.control_plane.server' 2>/dev/null || true
sleep 2
nohup "$CSERVE_ROOT/scripts/run-control-plane.sh" >> /home/praveen/cserve-control.log 2>&1 &
sleep 12

curl -sf "$API/internal/health" | python3 -m json.tool

echo "==> Sync models.yaml (precedence + gemma4 TP4)..."
curl -sf -X POST "$API/admin/config/sync-from-yaml" \
  -H "Authorization: Bearer $KEY" | python3 -m json.tool

echo "==> Restore gpt-oss / qwen min replicas (precedence gates launch order)..."
curl -sf -X PUT "$API/admin/config" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"models":{"qwen3-vl-32b":{"autoscaling":{"min_replicas":1,"max_replicas":4}},"gpt-oss-120b":{"autoscaling":{"min_replicas":1,"max_replicas":2}}}}' \
  | python3 -m json.tool

echo "==> Waiting for gemma4-31b READY (up to 25 min)..."
for i in $(seq 1 50); do
  curl -sf "$API/internal/registry" | python3 -c "
import sys,json
d=json.load(sys.stdin)
g4=[r for r in d['replicas'] if r['model']=='gemma4-31b']
for r in g4:
    print(r['status'], r['node_name'], r.get('gpu_ids'))
if any(r['status']=='READY' for r in g4):
    raise SystemExit(0)
if not g4:
    print('no gemma4 replica yet')
raise SystemExit(1)
" && { echo "Gemma 4 READY"; exit 0; }
  sleep 30
done
echo "Gemma 4 not READY yet — check dashboard and ~/cserve-agent.log on worker"
exit 1
