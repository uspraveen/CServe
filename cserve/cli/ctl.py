"""cserve-ctl — command-line tool for inspecting and managing a CServe cluster.

Talks to the control plane's /internal/* endpoints over HTTP.

Usage:
  cserve-ctl status                  Show cluster overview
  cserve-ctl nodes                   List all nodes and their GPUs
  cserve-ctl replicas                List all active replicas
  cserve-ctl models                  List configured models
  cserve-ctl events [--limit N]      Show recent autoscale events
  cserve-ctl health [--limit N]      Show recent health incidents
  cserve-ctl jobs [--limit N]        Show recent job events
  cserve-ctl snapshot                Dump full registry snapshot as JSON
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=10.0)


def cmd_status(client: httpx.Client) -> None:
    resp = client.get("/internal/health")
    resp.raise_for_status()
    d = resp.json()
    print(f"{'Status:':<16} {'OK' if d.get('ok') else 'ERROR'}")
    print(f"{'Redis:':<16} {'connected' if d.get('redis') else 'disconnected'}")
    print(f"{'Models:':<16} {', '.join(d.get('models', []))}")
    print(f"{'Nodes:':<16} {d.get('nodes', 0)}")
    print(f"{'Replicas:':<16} {d.get('replicas', 0)}")

    snap = client.get("/internal/registry").json()
    gpu_summary = snap.get("gpu_summary", {})
    for gpu_type, (total, free) in gpu_summary.items():
        print(f"{'GPUs (' + gpu_type + '):':<16} {total} total, {free} free, {total - free} allocated")


def cmd_nodes(client: httpx.Client) -> None:
    snap = client.get("/internal/registry").json()
    nodes = snap.get("nodes", [])
    if not nodes:
        print("No nodes configured.")
        return

    for n in nodes:
        status_icon = {"ONLINE": "+", "OFFLINE": "x", "DEGRADED": "~"}.get(n["status"], "?")
        print(f"\n[{status_icon}] {n['name']} ({n['host']}) — {n['status']}")
        for gpu in n.get("gpus", []):
            state = gpu.get("state", "?")
            used = gpu.get("memory_used_mb", 0)
            total = gpu.get("memory_total_mb", 0)
            mem = f"{used}/{total} MB" if total else ""
            alloc = f" → {gpu['allocated_replica_id']}" if gpu.get("allocated_replica_id") else ""
            print(f"  GPU {gpu['index']:>2}: {state:<12} {mem:<20}{alloc}")


def cmd_replicas(client: httpx.Client) -> None:
    snap = client.get("/internal/registry").json()
    replicas = snap.get("replicas", [])
    if not replicas:
        print("No active replicas.")
        return

    print(f"{'REPLICA':<14} {'MODEL':<20} {'NODE':<12} {'GPUs':<10} {'STATUS':<10} {'INFLIGHT':<10}")
    print("-" * 76)
    for r in replicas:
        gpus = ",".join(str(g) for g in r.get("gpu_ids", []))
        print(
            f"{r['replica_id']:<14} {r['model']:<20} {r['node_name']:<12} "
            f"{gpus:<10} {r['status']:<10} {r.get('inflight_requests', 0):<10}"
        )


def cmd_models(client: httpx.Client) -> None:
    snap = client.get("/internal/registry").json()
    models = snap.get("models", [])
    if not models:
        print("No models configured.")
        return

    for m in models:
        auto = m.get("autoscaling", {})
        print(f"\n  {m['name']}")
        print(f"    served as:  {m.get('served_model_name', '—')}")
        print(f"    hf_model:   {m.get('hf_model', '—')}")
        print(f"    tp:         {m.get('tp', 1)}")
        print(f"    replicas:   {auto.get('min_replicas', 1)}-{auto.get('max_replicas', 1)}")
        print(f"    routing:    {m.get('routing_strategy', '—')}")


def cmd_events(client: httpx.Client, limit: int) -> None:
    resp = client.get(f"/dashboard/api/events/autoscale?limit={limit}")
    resp.raise_for_status()
    events = resp.json()
    if not events:
        print("No autoscale events.")
        return

    for e in events:
        import time
        ts = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
        reasons = ", ".join(e.get("reasons", []))
        print(f"[{ts}] {e['action']:<14} {e['model']:<20} {e['from_replicas']}→{e['to_replicas']}  {reasons}")


def cmd_health(client: httpx.Client, limit: int) -> None:
    resp = client.get(f"/dashboard/api/events/health?limit={limit}")
    resp.raise_for_status()
    events = resp.json()
    if not events:
        print("No health incidents.")
        return

    for e in events:
        import time
        ts = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
        node = e.get("node_name") or "—"
        print(f"[{ts}] {e['incident_type']:<20} node={node:<12} {e.get('details', '')}")


def cmd_jobs(client: httpx.Client, limit: int) -> None:
    resp = client.get(f"/dashboard/api/events/jobs?limit={limit}")
    resp.raise_for_status()
    events = resp.json()
    if not events:
        print("No job events.")
        return

    for e in events:
        import time
        ts = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
        meta = e.get("metadata", {})
        model = meta.get("model", "—")
        print(f"[{ts}] {e['event']:<12} {e['job_id'][:12]:<14} {model:<20} replica={e.get('replica_id', '—')}")


def cmd_snapshot(client: httpx.Client) -> None:
    resp = client.get("/internal/registry")
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="CServe Cluster CLI")
    parser.add_argument("--url", default="http://127.0.0.1:8002",
                        help="Control plane URL (default: http://127.0.0.1:8002)")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="Cluster overview")
    sub.add_parser("nodes", help="List nodes and GPUs")
    sub.add_parser("replicas", help="List active replicas")
    sub.add_parser("models", help="List configured models")

    ev = sub.add_parser("events", help="Recent autoscale events")
    ev.add_argument("--limit", type=int, default=20)

    hl = sub.add_parser("health", help="Recent health incidents")
    hl.add_argument("--limit", type=int, default=20)

    jb = sub.add_parser("jobs", help="Recent job events")
    jb.add_argument("--limit", type=int, default=20)

    sub.add_parser("snapshot", help="Full registry JSON dump")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = _client(args.url)

    try:
        cmds = {
            "status": lambda: cmd_status(client),
            "nodes": lambda: cmd_nodes(client),
            "replicas": lambda: cmd_replicas(client),
            "models": lambda: cmd_models(client),
            "events": lambda: cmd_events(client, args.limit),
            "health": lambda: cmd_health(client, args.limit),
            "jobs": lambda: cmd_jobs(client, args.limit),
            "snapshot": lambda: cmd_snapshot(client),
        }
        cmds[args.command]()
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to {args.url}")
        print("Is the control plane running?")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"ERROR: HTTP {e.response.status_code}")
        sys.exit(1)


if __name__ == "__main__":
    main()
