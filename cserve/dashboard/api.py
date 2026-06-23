"""Dashboard API — REST endpoints + WebSocket for the real-time UI.

Serves:
  1. /dashboard/ — the single-page HTML dashboard.
  2. /dashboard/ws — WebSocket for live cluster state pushes (1Hz).
  3. /dashboard/api/* — JSON endpoints for historical data.

The WebSocket broadcasts a full cluster snapshot every second, including:
  - Nodes (status, GPUs, memory)
  - Replicas (status, model, inflight, health)
  - Queue depths per model
  - Recent autoscale events
  - Recent health incidents

This is intentionally a simple server-push model.  The frontend opens one
WebSocket and renders the state — no client-side polling needed.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from cserve.common.logging import get_logger
from cserve.common.ui_tuning import (
    AUTOSCALE_UI_FIELDS,
    ENGINE_UI_FIELDS,
    SAFETY_UI_FIELDS,
)

log = get_logger("dashboard")

BROADCAST_INTERVAL_S = 1.0


def _derive_model_capabilities(cfg) -> list[str]:
    """Infer the playground capabilities exposed by a model."""
    hf_model = cfg.hf_model.lower()
    served_name = cfg.served_model_name.lower()
    runner = (cfg.engine.runner or "").lower()
    convert = (cfg.engine.convert or "").lower()

    if runner == "pooling" or convert == "embed" or "embedding" in hf_model:
        return ["embeddings"]
    if "whisper" in hf_model or "whisper" in served_name:
        return ["transcription"]

    capabilities = ["chat"]
    if any(token in hf_model for token in ("vl", "vision", "llava", "idefics", "gemma-3")):
        capabilities.append("vision")
    return capabilities


class DashboardAPI:
    """Dashboard REST + WebSocket API."""

    def __init__(self, registry, db, queue, models_config=None,
                 gpu_guard=None, health_manager=None,
                 node_client=None) -> None:
        self.registry = registry
        self.db = db
        self.queue = queue
        self.models_config = models_config or {}
        self.gpu_guard = gpu_guard
        self.health_manager = health_manager
        self.node_client = node_client
        self._ws_clients: set[WebSocket] = set()
        self._broadcast_task: asyncio.Task | None = None

        self.router = APIRouter(prefix="/dashboard")
        self._register_routes()

    async def start(self) -> None:
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        log.info("dashboard started")

    async def stop(self) -> None:
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        for ws in list(self._ws_clients):
            try:
                await ws.close()
            except Exception:
                pass
        log.info("dashboard stopped")

    @staticmethod
    def static_assets_dir() -> Path | None:
        """Return the path to built dashboard assets, or None if not found."""
        static_root = Path(__file__).parent / "static"
        for candidate in (static_root / "dist" / "assets", static_root / "assets"):
            if candidate.is_dir():
                return candidate
        return None

    def _register_routes(self) -> None:
        router = self.router
        static_root = Path(__file__).parent / "static"
        static_dirs = [static_root / "dist", static_root]

        def _serve_spa() -> HTMLResponse:
            for static_dir in static_dirs:
                html_path = static_dir / "index.html"
                if html_path.exists():
                    return HTMLResponse(
                        html_path.read_text(),
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
                    )
            return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)

        @router.get("/", response_class=HTMLResponse)
        async def dashboard_page():
            return _serve_spa()

        @router.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self._ws_clients.add(ws)
            log.info("dashboard client connected", total=len(self._ws_clients))
            try:
                # Send initial snapshot immediately
                snapshot = await self._build_snapshot()
                await ws.send_text(json.dumps(snapshot, default=str))
                # Keep connection alive, listen for client messages
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                        # Clients can request specific data
                        if msg == "snapshot":
                            snapshot = await self._build_snapshot()
                            await ws.send_text(json.dumps(snapshot, default=str))
                    except TimeoutError:
                        # Send ping to keep alive
                        await ws.send_text('{"type":"ping"}')
            except WebSocketDisconnect:
                pass
            except Exception as e:
                log.warning("ws error", error=str(e))
            finally:
                self._ws_clients.discard(ws)
                log.info("dashboard client disconnected", total=len(self._ws_clients))

        @router.get("/api/snapshot")
        async def api_snapshot():
            return await self._build_snapshot()

        @router.get("/api/events/jobs")
        async def api_job_events(limit: int = 50, model: str | None = None):
            return await self.db.get_recent_job_events(limit=limit, model=model)

        @router.get("/api/events/autoscale")
        async def api_autoscale_events(
            model: str | None = None,
            limit: int = Query(500, ge=1, le=5000),
            since_ts: float | None = None,
            until_ts: float | None = None,
            window_s: float | None = None,
        ):
            """Scale events in [since_ts, until_ts], or last window_s seconds, default last 24h."""
            now = time.time()
            if since_ts is not None and until_ts is not None:
                lo, hi = float(since_ts), float(until_ts)
                if hi < lo:
                    lo, hi = hi, lo
                events = await self.db.get_autoscale_events(
                    model=model, limit=limit, since=lo, until=hi,
                )
            elif window_s is not None:
                w = max(60.0, float(window_s))
                events = await self.db.get_autoscale_events(
                    model=model, limit=limit, since=now - w, until=now,
                )
            else:
                events = await self.db.get_autoscale_events(
                    model=model, limit=limit, since=now - 86400.0, until=now,
                )
            return [e for e in events if e.get("action") != "HOLD"]

        @router.get("/api/events/health")
        async def api_health_incidents(limit: int = 50):
            return await self.db.get_recent_health_incidents(limit=limit)

        @router.get("/api/latency/{model}")
        async def api_latency(model: str, window_s: float = 300):
            return await self.db.get_job_latency_stats(model, window_s)

        @router.get("/api/users")
        async def api_users():
            return await self.db.list_users()

        @router.get("/api/keys")
        async def api_keys(user_id: str | None = None):
            return await self.db.list_api_keys(user_id=user_id)

        @router.get("/api/usage")
        async def api_usage(
            user_id: str | None = None,
            window_s: float = 86400,
            since_ts: float | None = None,
            until_ts: float | None = None,
        ):
            return await self.db.get_usage_by_user(
                user_id=user_id,
                window_s=window_s,
                since_ts=since_ts,
                until_ts=until_ts,
            )

        @router.get("/api/usage/timeseries")
        async def api_usage_timeseries(
            user_id: str | None = None,
            model: str | None = None,
            bucket_s: int = 3600,
            window_s: float = 86400,
            since_ts: float | None = None,
            until_ts: float | None = None,
        ):
            return await self.db.get_usage_timeseries(
                user_id=user_id,
                model=model,
                bucket_s=bucket_s,
                window_s=window_s,
                since_ts=since_ts,
                until_ts=until_ts,
            )

        @router.get("/api/admin_config")
        async def api_admin_config():
            """Expose admin config for the dashboard (read-only, no auth)."""
            model_configs = {}
            for name, cfg in self.models_config.items():
                model_configs[name] = {
                    "engine": {k: getattr(cfg.engine, k) for k in ENGINE_UI_FIELDS},
                    "autoscaling": {k: getattr(cfg.autoscaling, k) for k in AUTOSCALE_UI_FIELDS},
                }
            guard_safety = self.gpu_guard.safety if self.gpu_guard else None
            safety_defaults = {
                "gpu_memory_limit": 0.79,
                "gpu_warn_threshold": 0.70,
                "gpu_danger_threshold": 0.79,
                "gpu_compute_sustain_threshold": 0.99,
                "gpu_compute_sustain_duration_s": 900.0,
                "guard_mitigation_window_s": 600.0,
                "guard_check_interval_s": 20.0,
            }
            return {
                "safety": {
                    k: getattr(guard_safety, k) if guard_safety else safety_defaults[k]
                    for k in SAFETY_UI_FIELDS
                },
                "models": model_configs,
            }

        @router.get("/api/gpu_guard")
        async def api_gpu_guard():
            if not self.gpu_guard:
                return {"entries": [], "events": []}
            return {
                "entries": self.gpu_guard.get_all_entries(),
                "events": self.gpu_guard.get_recent_events(limit=50),
            }

        @router.get("/api/models/{model_name:path}/activity")
        async def api_model_activity(model_name: str):
            """Per-model job tail, DB health rows, and in-memory remediation events."""
            name = unquote(model_name)
            jobs = await self.db.get_recent_job_events(limit=50, model=name)
            reps = [r for r in self.registry.get_all_replicas() if r.model == name]
            rids = [r.replica_id for r in reps]
            health = await self.db.get_health_incidents_for_replicas(rids, limit=40)
            remediation: list[dict] = []
            if self.health_manager:
                remediation = [
                    {**row}
                    for row in self.health_manager.recent_incidents
                    if row.get("model") == name
                ][-35:]
            remediation.reverse()
            return {
                "model": name,
                "recent_jobs": jobs,
                "health_incidents": health,
                "remediation": remediation,
            }

        async def _fetch_replica_vllm_logs(
            node_name: str,
            rid: str,
            *,
            log_offset: int = 0,
            max_lines: int = 200,
        ) -> dict:
            vllm: dict = {
                "output_tail": "",
                "exit_code": None,
                "vllm_metrics": {},
                "agent_error": None,
                "log_offset": 0,
                "log_size": 0,
                "log_path": "",
            }
            if not self.node_client:
                vllm["agent_error"] = "node_client_not_configured"
                return vllm
            try:
                raw = await self.node_client.get_replica_status(
                    node_name, rid, log_offset=log_offset, max_lines=max_lines,
                )
                if raw.get("error"):
                    vllm["agent_error"] = raw["error"]
                elif raw:
                    vllm["output_tail"] = raw.get("output_tail") or ""
                    vllm["exit_code"] = raw.get("exit_code")
                    vllm["vllm_metrics"] = raw.get("vllm_metrics") or {}
                    vllm["log_offset"] = raw.get("log_offset", 0)
                    vllm["log_size"] = raw.get("log_size", 0)
                    vllm["log_path"] = raw.get("log_path") or ""
                else:
                    vllm["agent_error"] = "empty_response_from_agent"
            except Exception as e:
                vllm["agent_error"] = str(e)
            return vllm

        @router.get("/api/replicas/{replica_id}/logs")
        async def api_replica_logs(
            replica_id: str,
            offset: int = Query(0, ge=0),
            max_lines: int = Query(200, ge=20, le=500),
        ):
            """Incremental vLLM log tail from the worker node agent (~2s polling)."""
            rid = unquote(replica_id)
            rep = self.registry.get_replica(rid)
            if not rep:
                return {"ok": False, "error": "unknown_replica", "replica_id": rid}
            vllm = await _fetch_replica_vllm_logs(
                rep.node_name, rid, log_offset=offset, max_lines=max_lines,
            )
            return {
                "ok": vllm.get("agent_error") is None,
                "replica_id": rid,
                "node_name": rep.node_name,
                "text": vllm.get("output_tail") or "",
                "offset": vllm.get("log_offset", 0),
                "size": vllm.get("log_size", 0),
                "log_path": vllm.get("log_path") or "",
                "agent_error": vllm.get("agent_error"),
            }

        @router.get("/api/control-plane/logs")
        async def api_control_plane_logs(
            lines: int = Query(200, ge=10, le=2000),
        ):
            """Recent control-plane journal lines (cosmos-9 systemd cserve-control)."""
            try:
                proc = await asyncio.create_subprocess_exec(
                    "journalctl",
                    "-u",
                    "cserve-control",
                    "-n",
                    str(lines),
                    "--no-pager",
                    "-o",
                    "short-iso",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    return {
                        "ok": False,
                        "text": "",
                        "error": (stderr or b"").decode(errors="replace").strip()
                        or f"journalctl exit {proc.returncode}",
                    }
                return {
                    "ok": True,
                    "text": stdout.decode("utf-8", errors="replace"),
                    "source": "journalctl -u cserve-control",
                }
            except FileNotFoundError:
                return {"ok": False, "text": "", "error": "journalctl not available"}
            except Exception as e:
                return {"ok": False, "text": "", "error": str(e)}

        @router.get("/api/replicas/{replica_id}/diagnostics")
        async def api_replica_diagnostics(
            replica_id: str,
            offset: int = Query(0, ge=0),
        ):
            """vLLM process tail from node agent (HTTP) + CServe DB rows for replica."""
            rid = unquote(replica_id)
            rep = self.registry.get_replica(rid)
            if not rep:
                return {
                    "ok": False,
                    "error": "unknown_replica",
                    "replica_id": rid,
                }

            vllm = await _fetch_replica_vllm_logs(
                rep.node_name, rid, log_offset=offset,
            )

            job_events = await self.db.get_job_events_for_replica(rid, limit=80)
            health = await self.db.get_health_incidents_for_replicas([rid], limit=50)
            remediation: list[dict] = []
            if self.health_manager:
                remediation = [
                    {**row}
                    for row in self.health_manager.recent_incidents
                    if row.get("replica_id") == rid
                ][-25:]
            remediation.reverse()

            return {
                "ok": True,
                "replica_id": rid,
                "node_name": rep.node_name,
                "model": rep.model,
                "http_endpoint": rep.http_endpoint,
                "status": (
                    rep.status.value
                    if hasattr(rep.status, "value")
                    else str(rep.status)
                ),
                "vllm": vllm,
                "cserve_job_events": job_events,
                "cserve_health_incidents": health,
                "remediation": remediation,
            }

        @router.get("/{catch_all:path}", include_in_schema=False)
        async def dashboard_spa_fallback(catch_all: str):
            """Catch-all: serve static assets if path matches, else SPA index."""
            if catch_all.startswith("assets/"):
                for static_dir in static_dirs:
                    asset_path = static_dir / catch_all
                    if asset_path.is_file():
                        media = "application/javascript" if catch_all.endswith(".js") \
                            else "text/css" if catch_all.endswith(".css") \
                            else "application/octet-stream"
                        return FileResponse(asset_path, media_type=media)
            return _serve_spa()

    async def _broadcast_loop(self) -> None:
        """Push cluster state to all connected WebSocket clients at 1Hz."""
        while True:
            try:
                if self._ws_clients:
                    snapshot = await self._build_snapshot()
                    snapshot["type"] = "state"
                    payload = json.dumps(snapshot, default=str)
                    dead: list[WebSocket] = []
                    for ws in list(self._ws_clients):
                        try:
                            await ws.send_text(payload)
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        self._ws_clients.discard(ws)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("broadcast error", error=str(e))
            await asyncio.sleep(BROADCAST_INTERVAL_S)

    async def _build_snapshot(self) -> dict:
        """Build a comprehensive cluster snapshot for the dashboard."""
        registry_data = self.registry.snapshot()

        # Queue depths
        try:
            queue_depths = await self.queue.queue_depths_all()
        except Exception:
            queue_depths = {}

        # Recent autoscale events (last 20, non-HOLD)
        try:
            autoscale_events = await self.db.get_autoscale_events(limit=50)
            autoscale_events = [e for e in autoscale_events if e.get("action") != "HOLD"]
        except Exception:
            autoscale_events = []

        # Recent health incidents
        try:
            health_incidents = await self.db.get_recent_health_incidents(limit=10)
        except Exception:
            health_incidents = []

        # Recent job events
        try:
            recent_jobs = await self.db.get_recent_job_events(limit=30)
        except Exception:
            recent_jobs = []

        # Aggregate stats
        total_gpus = 0
        free_gpus = 0
        for _, (t, f) in registry_data.get("gpu_summary", {}).items():
            total_gpus += t
            free_gpus += f

        total_replicas = len(registry_data.get("replicas", []))
        ready_replicas = sum(
            1 for r in registry_data.get("replicas", [])
            if r.get("status") == "READY"
        )
        total_inflight = sum(
            r.get("inflight_requests", 0)
            for r in registry_data.get("replicas", [])
        )

        # Model configurations for the dashboard
        model_configs = {}
        for name, cfg in self.models_config.items():
            model_configs[name] = {
                "served_model_name": cfg.served_model_name,
                "hf_model": cfg.hf_model,
                "tp": cfg.tp,
                "node_type_required": cfg.node_type_required,
                "node_types_allowed": cfg.node_types_allowed,
                "routing_strategy": cfg.routing_strategy.value,
                "capabilities": _derive_model_capabilities(cfg),
                "engine": {
                    "max_model_len": cfg.engine.max_model_len,
                    "max_num_seqs": cfg.engine.max_num_seqs,
                    "gpu_memory_utilization": cfg.engine.gpu_memory_utilization,
                    "dtype": cfg.engine.dtype,
                },
                "autoscaling": {
                    "min_replicas": cfg.autoscaling.min_replicas,
                    "max_replicas": cfg.autoscaling.max_replicas,
                    "allow_scale_to_zero": cfg.autoscaling.allow_scale_to_zero,
                    "idle_timeout_s": cfg.autoscaling.idle_timeout_s,
                    "target_inflight": cfg.autoscaling.target_inflight,
                    "replica_startup_timeout_s": cfg.autoscaling.replica_startup_timeout_s,
                },
            }

        # GPU Guard state
        guard_entries: list[dict] = []
        guard_events: list[dict] = []
        compute_notifications: list[dict] = []
        if self.gpu_guard:
            guard_entries = self.gpu_guard.get_all_entries()
            guard_events = self.gpu_guard.get_recent_events(limit=20)
            compute_notifications = (
                self.gpu_guard.get_compute_pressure_notifications()
            )

        return {
            "timestamp": time.time(),
            "nodes": registry_data.get("nodes", []),
            "replicas": registry_data.get("replicas", []),
            "models": registry_data.get("models", []),
            "model_configs": model_configs,
            "gpu_summary": registry_data.get("gpu_summary", {}),
            "queue_depths": queue_depths,
            "autoscale_events": autoscale_events,
            "health_incidents": health_incidents,
            "recent_jobs": recent_jobs,
            "gpu_guard": {
                "memory_limit": self.gpu_guard.safety.gpu_memory_limit if self.gpu_guard else 0.79,
                "compute_sustain_threshold": (
                    self.gpu_guard.safety.gpu_compute_sustain_threshold
                    if self.gpu_guard else 0.99
                ),
                "compute_sustain_duration_s": (
                    self.gpu_guard.safety.gpu_compute_sustain_duration_s
                    if self.gpu_guard else 900.0
                ),
                "mitigation_window_s": (
                    self.gpu_guard.safety.guard_mitigation_window_s
                    if self.gpu_guard else 600
                ),
                "entries": guard_entries,
                "events": guard_events,
            },
            "gpu_compute_notifications": compute_notifications,
            "remediation_log": (
                self.health_manager.recent_incidents[-30:]
                if self.health_manager else []
            ),
            "launch_failures": registry_data.get("launch_failures", []),
            "stats": {
                "total_gpus": total_gpus,
                "free_gpus": free_gpus,
                "total_replicas": total_replicas,
                "ready_replicas": ready_replicas,
                "total_inflight": total_inflight,
                "total_queue_depth": sum(queue_depths.values()),
            },
        }
