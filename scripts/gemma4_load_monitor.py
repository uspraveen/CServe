#!/usr/bin/env python3
"""30-minute Gemma 4 load test + stability monitor.

Runs N parallel chat-completion workers against the gateway while sampling:
  - dashboard snapshot (replica health, inflight)
  - health incidents API
  - control-plane log tail for gemma4 health/guard events
  - remote nvidia-smi on the replica node

Usage:
  python scripts/gemma4_load_monitor.py [--duration 1800] [--workers 15]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

API = "http://cosmos-9.ddns.ualr.edu:8002/v1/chat/completions"
SNAPSHOT = "http://cosmos-9.ddns.ualr.edu:8002/dashboard/api/snapshot"
HEALTH_EVENTS = "http://cosmos-9.ddns.ualr.edu:8002/dashboard/api/events/health"
KEY = os.environ.get("CSERVE_ADMIN_KEY", "")
MODEL = "gemma4-31b"
CONTROL_LOG = "/home/praveen/cserve-control.log"
SSH_NODE = "praveen@cosmos-11.ddns.ualr.edu"

PROMPT = (
    "Summarize in 3 bullet points why GPU memory tuning matters for "
    "large language model serving at scale."
)


@dataclass
class LoadStats:
    ok: int = 0
    err: int = 0
    latencies: list[float] = field(default_factory=list)
    last_error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_ok(self, latency: float) -> None:
        with self.lock:
            self.ok += 1
            if len(self.latencies) < 500:
                self.latencies.append(latency)

    def record_err(self, msg: str) -> None:
        with self.lock:
            self.err += 1
            self.last_error = msg[:200]


def _http_json(url: str, timeout: float = 12.0) -> dict | list | None:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"_error": str(e)}


def _chat_once(max_tokens: int = 128) -> tuple[bool, float, str]:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        API,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp.read()
        return True, time.monotonic() - t0, ""
    except Exception as e:
        return False, time.monotonic() - t0, str(e)


def worker_loop(stats: LoadStats, stop: threading.Event) -> None:
    while not stop.is_set():
        ok, lat, err = _chat_once()
        if ok:
            stats.record_ok(lat)
        else:
            stats.record_err(err)
        if stop.wait(0.5):
            break


def gemma4_replica(snapshot: dict) -> dict | None:
    for r in snapshot.get("replicas") or []:
        if isinstance(r, dict) and r.get("model") == MODEL:
            return r
    return None


def tail_gemma4_log(since_marker: str) -> list[str]:
    try:
        out = subprocess.run(
            ["grep", "-E", f"gemma4-31b|{since_marker}", CONTROL_LOG],
            capture_output=True,
            text=True,
            timeout=8,
        )
        lines = out.stdout.strip().splitlines() if out.stdout else []
        warn = [
            ln for ln in lines
            if any(
                k in ln
                for k in (
                    "health check failed",
                    "gpu_guard",
                    "DRAINING",
                    "force killing",
                    "replica removed",
                    "replica launch",
                    "in-place restart",
                )
            )
        ]
        return warn[-15:]
    except Exception:
        return []


def remote_gpu() -> str:
    cmd = [
        "ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes",
        SSH_NODE,
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader 2>/dev/null | head -1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (r.stdout or r.stderr or "ssh_failed").strip()
    except Exception as e:
        return f"ssh_error: {e}"


def pctl(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    i = int(len(s) * p / 100)
    return s[min(i, len(s) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=1800, help="seconds (default 30 min)")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--sample-interval", type=int, default=30)
    ap.add_argument("--log", default="/home/praveen/gemma4_load_monitor.log")
    args = ap.parse_args()

    stop = threading.Event()
    stats = LoadStats()
    threads = [
        threading.Thread(target=worker_loop, args=(stats, stop), daemon=True)
        for _ in range(args.workers)
    ]
    for t in threads:
        t.start()

    log_f = open(args.log, "a", encoding="utf-8")
    start = time.time()
    end = start + args.duration
    replica_id = ""
    incidents_seen: set[str] = set()
    sample_n = 0

    def log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}\n"
        log_f.write(line)
        log_f.flush()
        print(line, end="")

    log(f"=== START duration={args.duration}s workers={args.workers} ===")

    try:
        while time.time() < end:
            sample_n += 1
            snap = _http_json(SNAPSHOT)
            rep = gemma4_replica(snap) if isinstance(snap, dict) else None
            if rep:
                replica_id = rep.get("replica_id", replica_id)

            health_events = _http_json(f"{HEALTH_EVENTS}?limit=30")
            new_incidents: list[str] = []
            if isinstance(health_events, list):
                for ev in health_events:
                    if not isinstance(ev, dict):
                        continue
                    rid = ev.get("replica_id")
                    if rid and replica_id and rid != replica_id:
                        continue
                    if MODEL not in str(ev) and rid != replica_id:
                        if ev.get("replica_id") not in (replica_id, None):
                            continue
                    key = json.dumps(ev, sort_keys=True)
                    if key not in incidents_seen and (
                        rid == replica_id
                        or (ev.get("incident_type") or "").startswith("replica")
                        or "gemma4" in str(ev).lower()
                    ):
                        incidents_seen.add(key)
                        new_incidents.append(key[:300])

            gpu_line = remote_gpu()
            warn_lines = tail_gemma4_log(replica_id) if replica_id else []

            with stats.lock:
                ok, err = stats.ok, stats.err
                lats = list(stats.latencies)
                last_err = stats.last_error

            rep_summary = "no_replica"
            if rep:
                rep_summary = (
                    f"id={rep.get('replica_id')} status={rep.get('status')} "
                    f"node={rep.get('node_name')} inflight={rep.get('inflight_requests')} "
                    f"health_ok={rep.get('last_health_ok')} "
                    f"consec_fail={rep.get('consecutive_health_failures')}"
                )

            log(
                f"SAMPLE #{sample_n} | {rep_summary} | "
                f"load ok={ok} err={err} p50={pctl(lats,50)} p95={pctl(lats,95)} | "
                f"gpu0={gpu_line}"
            )
            if new_incidents:
                log(f"  NEW_HEALTH_INCIDENTS: {new_incidents[:3]}")
            if warn_lines:
                log(f"  CP_WARN: {warn_lines[-3:]}")
            if err and last_err:
                log(f"  LAST_LOAD_ERR: {last_err}")

            time.sleep(args.sample_interval)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)

    with stats.lock:
        ok, err = stats.ok, stats.err
        lats = list(stats.latencies)

    snap = _http_json(SNAPSHOT)
    rep = gemma4_replica(snap) if isinstance(snap, dict) else None
    final_status = rep.get("status") if rep else "MISSING"
    final_fail = rep.get("consecutive_health_failures") if rep else -1
    same_replica = rep and rep.get("replica_id") == replica_id

    log(
        f"=== END ok={ok} err={err} p50={pctl(lats,50)} p95={pctl(lats,95)} "
        f"final_status={final_status} same_replica={same_replica} "
        f"consec_fail={final_fail} replica={replica_id} ==="
    )
    log_f.close()
    return 0 if err == 0 and final_status == "READY" and same_replica else 1


if __name__ == "__main__":
    sys.exit(main())
