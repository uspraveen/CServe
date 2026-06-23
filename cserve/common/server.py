"""CServe Control Plane Server — the main entry point.

Wires together all control plane components and starts the unified HTTP server.

Components started:
  1. ClusterRegistry (in-memory state)
  2. EventLog (SQLite)
  3. JobQueue (Redis)
  4. Gateway (FastAPI — public-facing /v1/ API)
  5. Scheduler (background — drains queues to replicas)
  6. Autoscaler (background — scaling decisions)
  7. HealthManager (background — node/replica/GPU checks)
  8. MetricsCollector (background — scrapes vLLM /metrics)
  9. NodeAgentClient (HTTP client to talk to node agents)
  10. Internal API (heartbeats, status updates from agents)
  11. Prometheus /metrics endpoint

Everything runs in a single asyncio event loop inside one process.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid

import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest

from cserve.common.auth import AuthenticatedUser, KeyRole
from cserve.common.config import load_cluster_config, load_models_config, save_cluster_config
from cserve.common.config_sync import (
    persist_ui_tuning_to_yaml,
    should_apply_yaml_on_startup,
)
from cserve.common.demand import DemandTracker
from cserve.common.logging import get_logger
from cserve.common.metrics import BUILD_INFO
from cserve.common.models import NodeConfig, NodeStatus, ReplicaState, ReplicaStatus, SshConfig
from cserve.common.rate_limit import RateLimiter
from cserve.common.ui_tuning import (
    AUTOSCALE_UI_FIELDS,
    ENGINE_UI_FIELDS,
    SAFETY_UI_FIELDS,
    apply_model_tuning_payload,
    apply_safety_payload,
    decode_payload_json,
    model_tuning_payload_from_config,
    safety_payload_from_config,
)
from cserve.control_plane.autoscaler import Autoscaler
from cserve.control_plane.db import EventLog
from cserve.control_plane.gateway import Gateway
from cserve.control_plane.gpu_guard import GpuMemoryGuard
from cserve.control_plane.health import HealthManager
from cserve.control_plane.metrics_collector import MetricsCollector
from cserve.control_plane.node_client import NodeAgentClient
from cserve.control_plane.orchestrator import Orchestrator
from cserve.control_plane.placement import find_placement
from cserve.control_plane.queue import JobQueue
from cserve.control_plane.registry import ClusterRegistry
from cserve.control_plane.scheduler import Scheduler
from cserve.control_plane.ssh_manager import deploy_agent, probe_node, stop_agent
from cserve.dashboard.api import DashboardAPI

log = get_logger("server")


class ControlPlaneServer:
    """Orchestrates all control plane components."""

    def __init__(self, cluster_yaml: str, models_yaml: str, db_path: str | None = None) -> None:
        # Load config
        self._cluster_yaml = cluster_yaml   # kept for live config saves
        self._models_yaml = models_yaml
        self.cluster_config = load_cluster_config(cluster_yaml)
        self.global_config, self.models_config = load_models_config(models_yaml)

        # Core state
        self.registry = ClusterRegistry(self.cluster_config)
        self.registry.load_models(self.models_config)

        # Subsystems
        self.db = EventLog(db_path or "/var/lib/cserve/events.db")
        self.queue = JobQueue(self.cluster_config.redis)
        self.node_client = NodeAgentClient(self.registry)
        self.demand_tracker = DemandTracker(window_s=60)

        # Redis client for rate limiting (shares config with queue)
        rc = self.cluster_config.redis
        self._redis_client = aioredis.Redis(
            host=rc.host, port=rc.port, db=rc.db, password=rc.password,
            decode_responses=True,
        )
        self.rate_limiter = RateLimiter(self._redis_client)

        # Build the models_config dict keyed by served_model_name for gateway
        models_by_served_name = {}
        for name, cfg in self.models_config.items():
            models_by_served_name[cfg.served_model_name] = cfg
            if name != cfg.served_model_name:
                models_by_served_name[name] = cfg

        self.gateway = Gateway(
            self.registry, self.queue, self.db, models_by_served_name,
            demand_tracker=self.demand_tracker,
            rate_limiter=self.rate_limiter,
        )
        self.scheduler = Scheduler(self.registry, self.queue, self.db)
        node_cuda_devices: dict[str, list[int]] = {}
        for n in self.cluster_config.nodes:
            if not n.cuda_devices:
                continue
            node_cuda_devices[n.name] = [
                int(x.strip()) for x in str(n.cuda_devices).split(",") if x.strip()
            ]
        self.autoscaler = Autoscaler(
            self.registry, self.queue, self.db,
            demand_tracker=self.demand_tracker,
            launch_callback=self._launch_replica,
            stop_callback=self._stop_replica,
            node_client=self.node_client,
            node_cuda_devices=node_cuda_devices,
        )
        self.gpu_guard = GpuMemoryGuard(
            self.registry,
            safety=self.cluster_config.safety,
            migrate_callback=self._migrate_replica,
            pause_callback=self._pause_replica_for_mitigation,
            resume_callback=self._resume_replica_after_mitigation,
        )
        self.health_manager = HealthManager(
            self.registry, self.db,
            safety_config=self.cluster_config.safety,
            node_agent_client=self.node_client,
            gpu_guard=self.gpu_guard,
            restart_callback=self._restart_replica_in_place,
            migrate_callback=self._migrate_replica,
            models_config=self.models_config,
        )
        self.metrics_collector = MetricsCollector(self.registry)
        self.dashboard = DashboardAPI(
            self.registry, self.db, self.queue,
            models_by_served_name, gpu_guard=self.gpu_guard,
            health_manager=self.health_manager,
            node_client=self.node_client,
        )

        # Build the top-level FastAPI app
        self.app = FastAPI(title="CServe Control Plane", version="0.1.0", docs_url=None, redoc_url=None)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._register_internal_routes()
        self.app.include_router(self.dashboard.router)

        from fastapi.responses import RedirectResponse

        @self.app.get("/dashboard", include_in_schema=False)
        async def _redirect_dashboard():
            return RedirectResponse(url="/dashboard/", status_code=301)

        assets_dir = self.dashboard.static_assets_dir()
        if assets_dir:
            from fastapi.staticfiles import StaticFiles
            self.app.mount(
                "/dashboard/assets",
                StaticFiles(directory=str(assets_dir)),
                name="dashboard-assets",
            )

        self.app.mount("/", self.gateway.app)

        BUILD_INFO.info({"version": "0.1.0", "component": "control_plane"})

    def _rebuild_gateway_models_config(self) -> None:
        """Rebuild served-name map after ``ModelConfig`` instances are replaced."""
        d = self.gateway.models_config
        d.clear()
        for name, cfg in self.models_config.items():
            d[cfg.served_model_name] = cfg
            if name != cfg.served_model_name:
                d[name] = cfg

    async def _apply_sqlite_tuning_overlay(self, rows: list[dict]) -> None:
        """Overlay SQLite ``ui_runtime_tuning`` on YAML-loaded config."""
        if not rows:
            return
        for row in rows:
            scope = row["scope"]
            payload = decode_payload_json(row["payload_json"])
            if scope == "safety":
                self.cluster_config.safety = apply_safety_payload(
                    self.cluster_config.safety, payload)
            elif scope == "model":
                mn = row["model_name"]
                cfg = self.models_config.get(mn)
                if cfg is not None:
                    self.models_config[mn] = apply_model_tuning_payload(cfg, payload)
        self.gpu_guard.safety = self.cluster_config.safety
        self.health_manager.safety = self.cluster_config.safety
        self.registry.load_models(self.models_config)
        self._rebuild_gateway_models_config()
        log.info("applied ui_runtime_tuning from SQLite", row_count=len(rows))

    async def _seed_sqlite_tuning_from_memory(self) -> None:
        """First boot: persist YAML-loaded tunables so overlay table is populated."""
        ts = time.time()
        rows: list[tuple[str, str, dict, float, str]] = [
            (
                "safety",
                "",
                safety_payload_from_config(self.cluster_config.safety),
                ts,
                "file",
            ),
        ]
        for name, cfg in self.models_config.items():
            rows.append(
                (
                    "model",
                    name,
                    model_tuning_payload_from_config(cfg),
                    ts,
                    "file",
                ),
            )
        await self.db.replace_all_ui_runtime_tuning(rows)
        log.info("seeded ui_runtime_tuning from YAML", row_count=len(rows))

    async def _reload_tuning_from_yaml(self, *, persist_sqlite: bool = True) -> list[str]:
        """Reload UI-tunable fields from YAML files into memory (and optionally SQLite)."""
        fresh_cluster = load_cluster_config(self._cluster_yaml)
        _, fresh_models = load_models_config(self._models_yaml)

        s = self.cluster_config.safety
        fs = fresh_cluster.safety
        for field in SAFETY_UI_FIELDS:
            setattr(s, field, getattr(fs, field))
        self.gpu_guard.safety = s
        self.health_manager.safety = s

        updated: list[str] = []
        for name, cfg in list(self.models_config.items()):
            if name not in fresh_models:
                continue
            fm = fresh_models[name]
            self.models_config[name] = cfg.model_copy(
                update={
                    "tp": fm.tp,
                    "deploy_priority": fm.deploy_priority,
                    "node_type_required": fm.node_type_required,
                    "node_types_allowed": list(fm.node_types_allowed),
                    "nodes_allowed": list(fm.nodes_allowed),
                    "gpu_guard_exempt": fm.gpu_guard_exempt,
                    "engine": fm.engine.model_copy(deep=True),
                    "autoscaling": fm.autoscaling.model_copy(deep=True),
                },
            )
            updated.append(name)

        self.registry.load_models(self.models_config)
        self._rebuild_gateway_models_config()

        if persist_sqlite:
            ts = time.time()
            rows: list[tuple[str, str, dict, float, str]] = [
                (
                    "safety",
                    "",
                    safety_payload_from_config(self.cluster_config.safety),
                    ts,
                    "file",
                ),
            ]
            for name, cfg in self.models_config.items():
                rows.append(
                    (
                        "model",
                        name,
                        model_tuning_payload_from_config(cfg),
                        ts,
                        "file",
                    ),
                )
            await self.db.replace_all_ui_runtime_tuning(rows)

        return updated

    async def _resolve_startup_tuning(self) -> None:
        """Pick YAML vs SQLite overlay using file mtimes (see config_sync)."""
        rows = await self.db.get_all_ui_runtime_tuning()
        if should_apply_yaml_on_startup(self._cluster_yaml, self._models_yaml, rows):
            await self._reload_tuning_from_yaml(persist_sqlite=True)
            log.info("startup tuning: YAML newer than SQLite — loaded from disk")
        elif rows:
            await self._apply_sqlite_tuning_overlay(rows)
        else:
            await self._seed_sqlite_tuning_from_memory()

    async def _persist_admin_tuning_to_db(self, request_data: dict) -> list[str]:
        """Persist current in-memory tunables for keys present in the PUT body."""
        ts = time.time()
        written: list[str] = []
        if "safety" in request_data:
            await self.db.upsert_ui_runtime_tuning(
                "safety",
                "",
                safety_payload_from_config(self.cluster_config.safety),
                "ui",
                ts,
            )
            written.append("ui_runtime_tuning: safety")
        if "models" in request_data:
            for model_name in request_data["models"]:
                cfg = self.models_config.get(model_name)
                if not cfg:
                    continue
                await self.db.upsert_ui_runtime_tuning(
                    "model",
                    model_name,
                    model_tuning_payload_from_config(cfg),
                    "ui",
                    ts,
                )
                written.append(f"ui_runtime_tuning: model/{model_name}")
        return written

    def _register_internal_routes(self) -> None:
        """Routes used by node agents and internal tooling."""
        app = self.app

        @app.post("/internal/heartbeat")
        async def heartbeat(request: Request):
            data = await request.json()
            node_name = data.get("node_name")
            if not node_name:
                return {"ok": False, "error": "missing node_name"}

            # Update heartbeat
            self.registry.record_heartbeat(node_name)

            # Update GPU info
            from cserve.common.models import GpuInfo
            gpus = [GpuInfo(**g) for g in data.get("gpus", [])]
            if gpus:
                self.registry.update_gpu_info(node_name, gpus)

            # Tell the agent which replica IDs the control plane still owns on this node.
            # The agent reconciles local vLLM + ~/.cserve/replicas.json against this set.
            from cserve.common.models import ReplicaStatus

            expected = [
                r.replica_id
                for r in self.registry.get_all_replicas()
                if r.node_name == node_name
                and r.status not in (ReplicaStatus.FAILED, ReplicaStatus.STOPPING)
            ]
            return {"ok": True, "expected_replica_ids": expected}

        @app.post("/internal/replica_status")
        async def replica_status(request: Request):
            """Called by node agents when a replica becomes ready or fails."""
            data = await request.json()
            replica_id = data.get("replica_id")
            status_str = data.get("status", "")
            endpoint = data.get("endpoint", "")

            if not replica_id:
                return {"ok": False, "error": "missing replica_id"}

            replica = self.registry.get_replica(replica_id)
            if not replica:
                return {"ok": False, "error": f"unknown replica: {replica_id}"}

            if status_str == "READY":
                if endpoint:
                    port = int(endpoint.rsplit(":", 1)[-1]) if ":" in endpoint else 0
                    self.registry.update_replica_endpoint(replica_id, endpoint, port, replica.pid)
                self.registry.set_replica_status(replica_id, ReplicaStatus.READY)
                log.info("replica ready", replica=replica_id, endpoint=endpoint)
            elif status_str == "FAILED":
                try:
                    self.registry.set_replica_status(replica_id, ReplicaStatus.FAILED)
                except Exception:
                    pass
                log.error("replica failed to start", replica=replica_id)

            return {"ok": True}

        @app.get("/internal/registry")
        async def registry_snapshot():
            return self.registry.snapshot()

        @app.get("/metrics")
        async def prometheus_metrics():
            return Response(
                content=generate_latest(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

        @app.get("/internal/health")
        async def internal_health():
            redis_ok = await self.queue.ping()
            return {
                "ok": True,
                "redis": redis_ok,
                "models": list(self.models_config.keys()),
                "nodes": len(self.registry.get_all_nodes()),
                "replicas": len(self.registry.get_all_replicas()),
            }

        # ── Admin API (key + user management) ─────────────────────────

        async def _require_admin(request: Request) -> AuthenticatedUser | JSONResponse:
            """Auth guard that requires admin role."""
            auth = await self.gateway._authenticate(request)
            if isinstance(auth, JSONResponse):
                return auth
            if auth.role != KeyRole.ADMIN:
                return JSONResponse(
                    {"error": {"message": "Admin access required", "type": "auth_error"}},
                    status_code=403,
                )
            return auth

        @app.post("/admin/keys")
        async def create_key(request: Request):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            data = await request.json()
            raw_key, api_key = await self.db.create_api_key(
                user_id=data.get("user_id", ""),
                name=data.get("name", ""),
                role=data.get("role", "user"),
                rate_limit_rpm=data.get("rate_limit_rpm", 0),
            )
            return {
                "key": raw_key,
                "key_id": api_key.key_id,
                "key_prefix": api_key.key_prefix,
                "user_id": api_key.user_id,
                "name": api_key.name,
                "role": api_key.role,
                "rate_limit_rpm": api_key.rate_limit_rpm,
                "warning": "Store this key securely — it will NOT be shown again.",
            }

        @app.get("/admin/keys")
        async def list_keys(request: Request, user_id: str | None = None):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            return await self.db.list_api_keys(user_id=user_id)

        @app.delete("/admin/keys/{key_id}")
        async def revoke_key(request: Request, key_id: str):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            revoked = await self.db.revoke_api_key(key_id)
            return {"ok": revoked, "key_id": key_id}

        @app.put("/admin/keys/{key_id}")
        async def update_key(request: Request, key_id: str):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            data = await request.json()
            updated = await self.db.update_api_key(
                key_id,
                name=data.get("name"),
                rate_limit_rpm=data.get("rate_limit_rpm"),
                enabled=data.get("enabled"),
            )
            return {"ok": updated, "key_id": key_id}

        @app.get("/admin/users")
        async def list_users(request: Request):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            return await self.db.list_users()

        @app.get("/admin/usage")
        async def get_usage(
            request: Request,
            user_id: str | None = None,
            window_s: float = 86400,
            since_ts: float | None = None,
            until_ts: float | None = None,
        ):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            return await self.db.get_usage_by_user(
                user_id=user_id,
                window_s=window_s,
                since_ts=since_ts,
                until_ts=until_ts,
            )

        @app.get("/admin/usage/timeseries")
        async def get_usage_timeseries(
            request: Request,
            user_id: str | None = None,
            model: str | None = None,
            bucket_s: int = 3600,
            window_s: float = 86400,
            since_ts: float | None = None,
            until_ts: float | None = None,
        ):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            return await self.db.get_usage_timeseries(
                user_id=user_id,
                model=model,
                bucket_s=bucket_s,
                window_s=window_s,
                since_ts=since_ts,
                until_ts=until_ts,
            )

        # ── Admin Controls (runtime config management) ────────────

        @app.get("/admin/config")
        async def get_admin_config(request: Request):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            safety = self.cluster_config.safety
            model_configs = {}
            for name, cfg in self.models_config.items():
                model_configs[name] = {
                    "engine": {k: getattr(cfg.engine, k) for k in ENGINE_UI_FIELDS},
                    "autoscaling": {k: getattr(cfg.autoscaling, k) for k in AUTOSCALE_UI_FIELDS},
                }
            return {
                "safety": {k: getattr(safety, k) for k in SAFETY_UI_FIELDS},
                "models": model_configs,
            }

        @app.put("/admin/config")
        async def update_admin_config(request: Request):
            """Update runtime configuration. Engine changes require replica restart."""
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            data = await request.json()

            changes: list[str] = []
            requires_restart: list[str] = []

            if "safety" in data:
                s = data["safety"]
                safety = self.cluster_config.safety
                if "gpu_memory_limit" in s:
                    old = safety.gpu_memory_limit
                    safety.gpu_memory_limit = float(s["gpu_memory_limit"])
                    if self.gpu_guard:
                        self.gpu_guard.safety = safety
                    changes.append(f"gpu_memory_limit: {old:.2f} -> {safety.gpu_memory_limit:.2f}")
                if "gpu_warn_threshold" in s:
                    safety.gpu_warn_threshold = float(s["gpu_warn_threshold"])
                    changes.append(f"gpu_warn_threshold -> {safety.gpu_warn_threshold:.2f}")
                if "gpu_danger_threshold" in s:
                    safety.gpu_danger_threshold = float(s["gpu_danger_threshold"])
                    changes.append(f"gpu_danger_threshold -> {safety.gpu_danger_threshold:.2f}")
                if "gpu_compute_sustain_threshold" in s:
                    safety.gpu_compute_sustain_threshold = float(
                        s["gpu_compute_sustain_threshold"])
                    changes.append(
                        "gpu_compute_sustain_threshold -> "
                        f"{safety.gpu_compute_sustain_threshold:.2f}"
                    )
                if "gpu_compute_sustain_duration_s" in s:
                    safety.gpu_compute_sustain_duration_s = float(
                        s["gpu_compute_sustain_duration_s"])
                    changes.append(
                        "gpu_compute_sustain_duration_s -> "
                        f"{safety.gpu_compute_sustain_duration_s:.0f}"
                    )
                if "guard_mitigation_window_s" in s:
                    safety.guard_mitigation_window_s = float(s["guard_mitigation_window_s"])
                    changes.append(f"guard_mitigation_window_s -> {safety.guard_mitigation_window_s}")
                if "guard_check_interval_s" in s:
                    safety.guard_check_interval_s = float(s["guard_check_interval_s"])
                    changes.append(
                        f"guard_check_interval_s -> {safety.guard_check_interval_s:.0f}"
                    )
                if self.gpu_guard:
                    self.gpu_guard.safety = safety

            if "models" in data:
                for model_name, model_data in data["models"].items():
                    cfg = self.models_config.get(model_name)
                    if not cfg:
                        continue

                    if "autoscaling" in model_data:
                        a = model_data["autoscaling"]
                        policy = cfg.autoscaling
                        for field in AUTOSCALE_UI_FIELDS:
                            if field in a:
                                old_val = getattr(policy, field)
                                setattr(policy, field, type(old_val)(a[field]))
                                changes.append(
                                    f"{model_name}.autoscaling.{field}: {old_val} -> {a[field]}"
                                )

                    if "engine" in model_data:
                        e = model_data["engine"]
                        eng = cfg.engine
                        for field in ENGINE_UI_FIELDS:
                            if field in e:
                                old_val = getattr(eng, field)
                                setattr(eng, field, type(old_val)(e[field]))
                                if old_val != type(old_val)(e[field]):
                                    requires_restart.append(
                                        f"{model_name}.engine.{field}: {old_val} -> {e[field]}")
                                    changes.append(requires_restart[-1])

            log.info("admin config updated", changes=changes,
                     requires_restart=requires_restart)

            persisted_sqlite = await self._persist_admin_tuning_to_db(data)
            persisted_yaml: list[str] = []
            try:
                persisted_yaml = persist_ui_tuning_to_yaml(
                    self._cluster_yaml,
                    self._models_yaml,
                    self.cluster_config,
                    self.models_config,
                )
            except Exception as exc:
                log.error("failed to mirror admin tuning to YAML", error=str(exc))

            return {
                "ok": True,
                "changes": changes,
                "requires_restart": requires_restart,
                "persisted_to_sqlite": persisted_sqlite,
                "persisted_to_yaml": persisted_yaml,
                "warning": (
                    "Engine changes require a rolling restart of affected model replicas. "
                    "Use POST /admin/restart/{model} to restart. Inflight requests will be "
                    "drained before each replica is restarted — no requests will be lost."
                ) if requires_restart else None,
            }

        @app.post("/admin/config/sync-from-yaml")
        async def admin_config_sync_from_yaml(request: Request):
            """Reload UI-tunable fields from YAML and replace SQLite overlay (file wins)."""
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin

            models_updated = await self._reload_tuning_from_yaml(persist_sqlite=True)

            log.info("admin config synced from YAML; SQLite ui_runtime_tuning rewritten")

            return {
                "ok": True,
                "message": (
                    "Reloaded safety and per-model engine/autoscaling from YAML and "
                    "replaced ui_runtime_tuning in SQLite. In-memory state matches files."
                ),
                "models_updated": models_updated,
            }

        @app.post("/admin/restart/{model_name}")
        async def admin_restart_model(request: Request, model_name: str):
            """Rolling restart: drain each replica, then restart one at a time."""
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            if model_name not in self.models_config:
                return JSONResponse(
                    {"error": {"message": f"Unknown model: {model_name}",
                               "type": "invalid_request_error"}},
                    status_code=404,
                )

            replicas = self.registry.get_replicas_for_model(model_name)
            ready = [r for r in replicas if r.status == ReplicaStatus.READY]
            if not ready:
                return JSONResponse(
                    {"error": {"message": "No READY replicas to restart",
                               "type": "invalid_request_error"}},
                    status_code=400,
                )

            asyncio.create_task(
                self._rolling_restart(model_name, [r.replica_id for r in ready])
            )

            return {
                "ok": True,
                "model": model_name,
                "replicas_queued": len(ready),
                "message": (
                    f"Rolling restart initiated for {len(ready)} replica(s). "
                    "Each replica will be drained (inflight requests completed), "
                    "stopped, and relaunched with updated engine settings. "
                    "Other replicas continue serving during the process."
                ),
            }

        @app.post("/admin/queues/purge")
        async def admin_purge_queues(request: Request):
            """Drop all (or selected) CServe Redis job queues for a clean slate.

            Does **not** clear vLLM in-process queues — use rolling restart
            (``POST /admin/restart/<model>``) or restart node agents after
            purging so workers start empty.
            """
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            try:
                data = await request.json()
            except Exception:
                data = {}
            raw_models = data.get("models")
            if raw_models is not None and not isinstance(raw_models, list):
                return JSONResponse(
                    {
                        "error": {
                            "message": "models must be a list of model names or omitted",
                            "type": "invalid_request_error",
                        },
                    },
                    status_code=400,
                )
            models: list[str] | None = None
            if isinstance(raw_models, list):
                models = [str(m) for m in raw_models]
                for m in models:
                    if m not in self.models_config:
                        return JSONResponse(
                            {
                                "error": {
                                    "message": f"Unknown model: {m}",
                                    "type": "invalid_request_error",
                                },
                            },
                            status_code=404,
                        )

            if models is None:
                purge_cb = bool(data.get("purge_callback_keys", True))
                purge_cx = bool(data.get("purge_cancelled_keys", True))
            else:
                purge_cb = bool(data.get("purge_callback_keys", False))
                purge_cx = bool(data.get("purge_cancelled_keys", False))

            result = await self.queue.purge_queues(
                models=models,
                purge_callback_keys=purge_cb,
                purge_cancelled_keys=purge_cx,
            )
            log.info(
                "admin purged redis queues",
                admin_user=admin.user_id,
                **result,
            )
            return {
                "ok": True,
                **result,
                "next_steps": (
                    "Redis CServe queues are empty. Restart replicas "
                    "(POST /admin/restart/{model} per model) or restart the "
                    "control plane so vLLM processes and any in-flight routing "
                    "state align with the empty queues. Clients with pending "
                    "requests may need to retry."
                ),
            }

        @app.post("/admin/cluster/stop")
        async def admin_cluster_stop(request: Request):
            """Stop every replica, optionally purge Redis queues, then sweep GPUs.

            GPU cleanup uses only indices from cluster.yaml ``cuda_devices`` for
            each node. By default it **SIGKILLs every compute process** nvidia-smi
            reports on those GPUs (not only vLLM). Set ``vllm_only_gpu_sweep`` to
            true for the older, narrower behavior on shared nodes.

            By default sweeps are **attempted even if the registry says OFFLINE**,
            so a stale offline flag does not skip cleanup (set
            ``sweep_offline_nodes`` false to restore the old skip).
            """
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            try:
                data = await request.json()
            except Exception:
                data = {}
            purge_queues = bool(data.get("purge_queues", True))
            wait_after = float(data.get("wait_after_stop_s", 8.0))
            second_wait = float(data.get("second_wait_s", 0.0))
            cleanup = bool(data.get("cleanup_orphans", True))
            force = bool(data.get("force", False))
            resume_after = bool(data.get("resume_autoscale_after", False))
            verify_gpus = bool(data.get("verify_gpus", True))
            verify_max_used_mb = int(data.get("verify_max_used_mb", 768))
            verify_max_frac = float(data.get("verify_max_frac", 0.025))
            retry_verify = bool(data.get("retry_sweep_on_verify_fail", True))
            vllm_only_sweep = bool(data.get("vllm_only_gpu_sweep", False))
            sweep_offline = bool(data.get("sweep_offline_nodes", True))

            report = await self._run_cluster_stop(
                purge_queues=purge_queues,
                wait_after_stop_s=wait_after,
                second_wait_s=second_wait,
                cleanup_orphans=cleanup,
                force=force,
                resume_autoscale_after=resume_after,
                verify_gpus=verify_gpus,
                verify_max_used_mb=verify_max_used_mb,
                verify_max_frac=verify_max_frac,
                retry_sweep_on_verify_fail=retry_verify,
                vllm_only_gpu_sweep=vllm_only_sweep,
                sweep_offline_nodes=sweep_offline,
            )
            report["ok"] = True
            report["admin_user"] = admin.user_id
            return report

        @app.post("/admin/cluster/resume")
        async def admin_cluster_resume(request: Request):
            """Re-enable autoscaling after ``/admin/cluster/stop`` (if it was left paused)."""
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            self.autoscaler.set_paused(False)
            return {"ok": True, "autoscaler_paused": False}

        # ── SSH Config ────────────────────────────────────────────────

        @app.get("/admin/ssh_config")
        async def get_ssh_config(request: Request):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            s = self.cluster_config.ssh
            return {
                "username": s.username,
                "key_path": s.key_path,
                "password": None,         # never echo the raw password back over the API
                "has_password": bool(s.password),
                "port": s.port,
                "timeout_s": s.timeout_s,
                "cserve_src": s.cserve_src,
                "python_path": s.python_path,
                "pip_path": s.pip_path,
            }

        @app.put("/admin/ssh_config")
        async def update_ssh_config(request: Request):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            data = await request.json()
            ssh = self.cluster_config.ssh
            for field_name in ("username", "key_path", "password", "port", "timeout_s",
                               "cserve_src", "python_path", "pip_path"):
                if field_name in data:
                    val = data[field_name]
                    # Allow clearing the password by sending null or empty string
                    if field_name == "password" and not val:
                        val = None
                    setattr(ssh, field_name, val)
            save_cluster_config(self.cluster_config, self._cluster_yaml)
            log.info("SSH config updated and saved", username=ssh.username)
            return {"ok": True, "message": "SSH configuration saved."}

        # ── Node management ───────────────────────────────────────────

        @app.get("/admin/nodes")
        async def list_nodes(request: Request):
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            nodes = self.registry.get_all_nodes()
            result = []
            for n in nodes:
                replica_count = len([
                    r for r in self.registry.get_all_replicas()
                    if r.node_name == n.name
                ])
                result.append({
                    "name": n.name,
                    "host": n.host,
                    "status": n.status.value,
                    "gpu_type": n.gpu_type,
                    "gpu_count": len(n.gpus),
                    "agent_endpoint": n.agent_endpoint,
                    "last_heartbeat": n.last_heartbeat,
                    "replica_count": replica_count,
                    "schedulable": n.schedulable,
                    "labels": n.labels,
                })
            return {"nodes": result}

        @app.patch("/admin/nodes/{node_name}")
        async def patch_node(request: Request, node_name: str):
            """Update runtime node flags (e.g. schedulable) and mirror to cluster.yaml."""
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            data = await request.json()
            if "schedulable" not in data:
                return JSONResponse(
                    {"error": {"message": "schedulable is required", "type": "invalid_request_error"}},
                    status_code=400,
                )
            schedulable = bool(data["schedulable"])
            if not self.registry.set_node_schedulable(node_name, schedulable):
                return JSONResponse(
                    {"error": {"message": f"Unknown node: {node_name}",
                               "type": "invalid_request_error"}},
                    status_code=404,
                )
            for nc in self.cluster_config.nodes:
                if nc.name == node_name:
                    nc.schedulable = schedulable
                    break
            try:
                save_cluster_config(self.cluster_config, self._cluster_yaml)
            except Exception as exc:
                log.error("failed to persist schedulable to cluster.yaml", error=str(exc))
                return JSONResponse(
                    {"error": {"message": f"Updated in memory but YAML write failed: {exc}",
                               "type": "server_error"}},
                    status_code=500,
                )
            log.info("node schedulable updated", node=node_name, schedulable=schedulable)
            return {"ok": True, "node": node_name, "schedulable": schedulable}

        @app.post("/admin/nodes/probe")
        async def probe_node_endpoint(request: Request):
            """SSH into a host and detect available GPUs."""
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            data = await request.json()
            host = data.get("host", "").strip()
            if not host:
                return JSONResponse(
                    {"error": {"message": "host is required", "type": "invalid_request_error"}},
                    status_code=400,
                )
            # Allow per-request SSH credential overrides
            ssh_cfg = SshConfig(**{
                **self.cluster_config.ssh.model_dump(),
                **{k: v for k, v in data.items()
                   if k in ("username", "key_path", "port", "timeout_s")},
            })
            result = await probe_node(host, ssh_cfg)
            return {
                "hostname": result.hostname,
                "os_info": result.os_info,
                "error": result.error,
                "gpus": [
                    {
                        "index": g.index,
                        "name": g.name,
                        "memory_total_mb": g.memory_total_mb,
                        "utilization_pct": g.utilization_pct,
                    }
                    for g in result.gpus
                ],
            }

        @app.post("/admin/nodes")
        async def add_node_endpoint(request: Request):
            """Add a new node to the cluster, deploy the agent, and hot-register."""
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            data = await request.json()

            # Required fields
            node_name = (data.get("name") or "").strip()
            host = (data.get("host") or "").strip()
            cuda_devices = (data.get("cuda_devices") or "").strip()
            gpu_type = (data.get("gpu_type") or "").strip()
            if not node_name or not host:
                return JSONResponse(
                    {"error": {"message": "name and host are required",
                               "type": "invalid_request_error"}},
                    status_code=400,
                )
            if not cuda_devices:
                return JSONResponse(
                    {"error": {"message": "cuda_devices is required (e.g. '0,1,2')",
                               "type": "invalid_request_error"}},
                    status_code=400,
                )
            if self.registry.get_node(node_name):
                return JSONResponse(
                    {"error": {"message": f"Node '{node_name}' already exists",
                               "type": "invalid_request_error"}},
                    status_code=409,
                )

            gpu_count = len([x for x in cuda_devices.split(",") if x.strip()])
            labels = data.get("labels") or {}
            sync_code = data.get("sync_code", True)

            # Build NodeConfig and hot-register so the agent can reach the CP
            node_cfg = NodeConfig(
                name=node_name,
                host=host,
                gpu_count=gpu_count,
                gpu_type=gpu_type,
                cuda_devices=cuda_devices,
                labels=labels,
            )
            self.registry.add_node(node_cfg, self.cluster_config.node_agent.port)

            # Determine the control-plane URL as the agent sees it
            cp_host = self.cluster_config.head.host or "localhost"
            cp_port = self.cluster_config.gateway.port
            control_plane_url = f"http://{cp_host}:{cp_port}"

            result = await deploy_agent(
                host=host,
                node_name=node_name,
                cuda_devices=cuda_devices,
                agent_port=self.cluster_config.node_agent.port,
                control_plane_url=control_plane_url,
                ssh_cfg=self.cluster_config.ssh,
                sync_code=sync_code,
            )

            if not result.ok:
                # Rollback registry entry
                self.registry.remove_node(node_name, force=True)
                return JSONResponse(
                    {
                        "ok": False,
                        "error": result.error,
                        "log": result.log,
                    },
                    status_code=500,
                )

            # Persist to cluster.yaml
            if node_cfg not in self.cluster_config.nodes:
                self.cluster_config.nodes.append(node_cfg)
            save_cluster_config(self.cluster_config, self._cluster_yaml)

            log.info("node added successfully", node=node_name, host=host, cuda=cuda_devices)
            return {
                "ok": True,
                "node": node_name,
                "host": host,
                "cuda_devices": cuda_devices,
                "gpu_type": gpu_type,
                "log": result.log,
                "message": (
                    f"Node '{node_name}' deployed and registered. "
                    f"The agent will connect back and become ONLINE within ~30 seconds."
                ),
            }

        @app.delete("/admin/nodes/{node_name}")
        async def remove_node_endpoint(request: Request, node_name: str):
            """Drain replicas, stop the agent, and remove the node from the cluster."""
            admin = await _require_admin(request)
            if isinstance(admin, JSONResponse):
                return admin
            data: dict = {}
            try:
                data = await request.json()
            except Exception:
                pass
            force = bool(data.get("force", False))

            node = self.registry.get_node(node_name)
            if not node:
                return JSONResponse(
                    {"error": {"message": f"Node '{node_name}' not found",
                               "type": "not_found"}},
                    status_code=404,
                )

            # Attempt graceful agent stop
            stop_result = await stop_agent(
                node.host, self.cluster_config.node_agent.port, self.cluster_config.ssh
            )
            stop_log = stop_result.log

            ok, reason = self.registry.remove_node(node_name, force=force)
            if not ok:
                return JSONResponse(
                    {"ok": False, "error": reason, "stop_log": stop_log},
                    status_code=409,
                )

            # Remove from cluster.yaml
            self.cluster_config.nodes = [
                n for n in self.cluster_config.nodes if n.name != node_name
            ]
            save_cluster_config(self.cluster_config, self._cluster_yaml)

            log.info("node removed", node=node_name, force=force)
            return {
                "ok": True,
                "node": node_name,
                "stop_log": stop_log,
                "message": f"Node '{node_name}' has been stopped and removed from the cluster.",
            }

    async def _rolling_restart(self, model_name: str, replica_ids: list[str]) -> None:
        """Restart replicas one at a time, draining each before stopping."""
        for rid in replica_ids:
            replica = self.registry.get_replica(rid)
            if not replica:
                continue

            log.info("rolling restart: draining replica",
                     model=model_name, replica=rid,
                     inflight=replica.inflight_requests)

            model_cfg = self.models_config.get(model_name)
            drain_timeout = int(model_cfg.autoscaling.drain_timeout_s) if model_cfg else 60

            try:
                if replica.status == ReplicaStatus.READY:
                    self.registry.set_replica_status(rid, ReplicaStatus.DRAINING)

                await self.node_client.drain_replica(
                    replica.node_name, rid, timeout_s=drain_timeout)

                polled = 0
                while polled < drain_timeout:
                    r = self.registry.get_replica(rid)
                    if not r or r.inflight_requests == 0:
                        break
                    await asyncio.sleep(1)
                    polled += 1

                log.info("rolling restart: stopping replica",
                         model=model_name, replica=rid)

                node_name = replica.node_name
                gpu_ids = list(replica.gpu_ids)

                if replica.status == ReplicaStatus.DRAINING:
                    self.registry.set_replica_status(rid, ReplicaStatus.STOPPING)
                try:
                    await self.node_client.stop_replica(node_name, rid, force=False)
                except Exception as stop_exc:
                    log.warning(
                        "rolling restart: graceful stop failed, forcing stop",
                        model=model_name, replica=rid, error=str(stop_exc),
                    )
                    await self.node_client.stop_replica(node_name, rid, force=True)
                self.registry.remove_replica(rid)

                await asyncio.sleep(3)

                log.info("rolling restart: relaunching on same GPUs",
                         model=model_name, node=node_name, gpus=gpu_ids)
                await self._restart_replica_in_place(
                    rid, model_name, node_name, gpu_ids)

                await asyncio.sleep(5)

            except Exception as e:
                log.error("rolling restart failed for replica",
                          model=model_name, replica=rid, error=str(e))

    async def start(self) -> None:
        """Start all background services."""
        await self.db.open()
        await self._resolve_startup_tuning()
        await self.queue.connect()

        # Bootstrap: create admin key on first startup if no keys exist
        key_count = await self.db.get_key_count()
        if key_count == 0:
            raw_key, api_key = await self.db.create_api_key(
                user_id="admin", name="bootstrap-admin", role="admin",
            )
            log.info(
                "BOOTSTRAP: created admin API key — store this securely!",
                key=raw_key, key_id=api_key.key_id,
            )
            print(f"\n{'='*60}")
            print("  BOOTSTRAP ADMIN API KEY (store securely!)")
            print(f"  {raw_key}")
            print(f"{'='*60}\n")

        await self.node_client.start()
        await self.gateway.startup()
        await self.scheduler.start()
        await self.autoscaler.start()
        await self.health_manager.start()
        await self.metrics_collector.start()
        await self.dashboard.start()

        log.info("control plane started",
                 host=self.cluster_config.gateway.host,
                 port=self.cluster_config.gateway.port,
                 models=list(self.models_config.keys()))

    async def stop(self) -> None:
        """Stop all background services."""
        log.info("control plane shutting down")
        await self.dashboard.stop()
        await self.metrics_collector.stop()
        await self.health_manager.stop()
        await self.autoscaler.stop()
        await self.scheduler.stop()
        await self.gateway.shutdown()
        await self.node_client.stop()
        await self.queue.close()
        await self._redis_client.aclose()
        await self.db.close()
        log.info("control plane stopped")

    # ─── Autoscaler callbacks ─────────────────────────────────────────────

    async def _launch_replica(
        self, model_name: str, avoid_node: str | None = None,
    ) -> None:
        """Called by autoscaler to launch a new replica.

        ``avoid_node`` is set when replacing a failed replica so placement prefers
        a different machine when the cluster has capacity elsewhere.
        """
        model_cfg = self.models_config.get(model_name)
        if not model_cfg:
            raise ValueError(f"Unknown model: {model_name}")

        # Find placement — use get_available_nodes() which filters out nodes
        # whose launch circuit breaker is open (unreachable, repeated failures).
        nodes = self.registry.get_available_nodes()
        existing_counts = {}
        for node in nodes:
            count = sum(
                1 for r in self.registry.get_replicas_for_model(model_name)
                if r.node_name == node.name
            )
            existing_counts[node.name] = count

        effective_avoid = avoid_node
        if not effective_avoid:
            for ev in reversed(self.registry.get_recent_launch_failures(within_s=900.0)):
                if ev.get("model") == model_name and ev.get("node"):
                    effective_avoid = ev["node"]
                    log.info(
                        "placement avoiding recent failure node",
                        model=model_name, avoid_node=effective_avoid,
                    )
                    break

        placement = find_placement(
            model_cfg, nodes, existing_counts, avoid_node=effective_avoid,
        )
        if not placement:
            raise RuntimeError(f"No placement available for {model_name} (tp={model_cfg.tp})")

        # Create replica state
        replica_id = uuid.uuid4().hex[:12]
        replica = ReplicaState(
            replica_id=replica_id,
            model=model_name,
            node_name=placement.node_name,
            gpu_ids=placement.gpu_indices,
            tp_size=model_cfg.tp,
        )

        # Reserve GPUs and register replica
        self.registry.allocate_gpus(placement.node_name, placement.gpu_indices, replica_id)
        self.registry.add_replica(replica)

        engine_args = Orchestrator._build_engine_args(model_cfg)

        env_vars = {}
        if model_cfg.hf_token:
            env_vars["HF_TOKEN"] = model_cfg.hf_token

        # Tell the node agent to launch
        try:
            result = await self.node_client.launch_replica(
                node_name=placement.node_name,
                replica_id=replica_id,
                model_name=model_name,
                hf_model=model_cfg.hf_model,
                served_model_name=model_cfg.served_model_name,
                variant="default",
                gpu_ids=placement.gpu_indices,
                tp_size=model_cfg.tp,
                engine_args=engine_args,
                env_vars=env_vars,
                health_timeout_s=model_cfg.autoscaling.replica_startup_timeout_s,
            )
            self.registry.update_replica_endpoint(
                replica_id, result.get("http_endpoint", ""),
                0, result.get("pid", 0),
            )
            # Successful launch → close circuit breaker for this node
            self.registry.record_launch_success(placement.node_name)
            log.info("replica launch initiated",
                     model=model_name, replica=replica_id,
                     node=placement.node_name, gpus=placement.gpu_indices)
        except Exception as e:
            log.error("replica launch failed", model=model_name,
                      replica=replica_id, error=str(e))
            self.registry.remove_replica(replica_id)
            self.registry.record_launch_failure_event(
                model_name, placement.node_name, str(e)
            )
            # Open circuit for node-level failures so we don't hammer a broken node.
            # Includes: network errors, "too many open files" (errno 24), OOM, etc.
            err_str = str(e).lower()
            node_level_tokens = (
                "timeout", "connect", "refused", "unreachable",
                "errno 24", "too many open files", "out of memory", "oom",
            )
            if any(tok in err_str for tok in node_level_tokens):
                self.registry.record_launch_failure(placement.node_name)
            raise

    async def _stop_replica(self, replica_id: str) -> None:
        """Called by autoscaler to stop a replica."""
        replica = self.registry.get_replica(replica_id)
        if not replica:
            return

        # Drain first
        try:
            self.registry.set_replica_status(replica_id, ReplicaStatus.DRAINING)
            await self.node_client.drain_replica(
                replica.node_name, replica_id,
                timeout_s=int(self.models_config.get(replica.model, None).autoscaling.drain_timeout_s
                              if self.models_config.get(replica.model) else 60),
            )
        except Exception as e:
            log.warning("drain failed, force stopping", replica=replica_id, error=str(e))

        # Stop
        try:
            self.registry.set_replica_status(replica_id, ReplicaStatus.STOPPING)
            await self.node_client.stop_replica(replica.node_name, replica_id)
        except Exception as e:
            log.error("stop failed", replica=replica_id, error=str(e))
        finally:
            self.registry.remove_replica(replica_id)

    async def _cluster_stop_single_replica(
        self, replica_id: str, *, force: bool,
    ) -> dict:
        """Drain+stop one replica with valid registry transitions."""
        detail: dict = {"replica_id": replica_id}
        replica = self.registry.get_replica(replica_id)
        if not replica:
            detail["note"] = "already_removed"
            return detail

        node_name = replica.node_name
        st = replica.status
        detail["status_before"] = st.value

        drain_timeout = 60
        mc = self.models_config.get(replica.model)
        if mc:
            drain_timeout = int(mc.autoscaling.drain_timeout_s)

        try:
            if not force and st == ReplicaStatus.READY:
                try:
                    self.registry.set_replica_status(
                        replica_id, ReplicaStatus.DRAINING,
                    )
                    await self.node_client.drain_replica(
                        node_name, replica_id, timeout_s=drain_timeout,
                    )
                except Exception as e:
                    detail["drain_warning"] = str(e)
            elif not force and st == ReplicaStatus.DRAINING:
                try:
                    await self.node_client.drain_replica(
                        node_name, replica_id, timeout_s=drain_timeout,
                    )
                except Exception as e:
                    detail["drain_warning"] = str(e)

            cur = self.registry.get_replica(replica_id)
            if cur and cur.status in (
                ReplicaStatus.DRAINING,
                ReplicaStatus.READY,
            ):
                if cur.status == ReplicaStatus.READY:
                    self.registry.set_replica_status(
                        replica_id, ReplicaStatus.DRAINING,
                    )
                self.registry.set_replica_status(
                    replica_id, ReplicaStatus.STOPPING,
                )

            use_force = force or st == ReplicaStatus.STARTING
            await self.node_client.stop_replica(
                node_name, replica_id, force=use_force,
            )
            detail["stopped"] = True
        except Exception as e:
            detail["stop_error"] = str(e)
            log.error("cluster stop: replica stop failed",
                      replica=replica_id, error=str(e))
        finally:
            try:
                self.registry.remove_replica(replica_id)
            except Exception:
                pass
        return detail

    async def _orphan_gpu_sweep(
        self,
        *,
        vllm_only: bool = False,
        sweep_offline_nodes: bool = True,
    ) -> dict[str, dict]:
        """Kill GPU compute processes on CServe-managed indices (``cuda_devices``).

        When ``vllm_only`` is false (default for cluster stop), every process
        nvidia-smi lists on those GPUs is SIGKILLed — matching an
        *inference cluster* shutdown. When true, only processes whose reported
        name looks like vLLM/python (legacy shared-node behavior).

        When ``sweep_offline_nodes`` is true, the sweep is still attempted if the
        registry says OFFLINE so stale health state cannot skip cleanup.
        """
        out: dict[str, dict] = {}
        for node in self.registry.get_all_nodes():
            allowed = [g.index for g in node.gpus]
            if not allowed:
                out[node.name] = {
                    "skipped": True,
                    "reason": "no_cuda_devices_in_cluster_config",
                }
                continue
            offline = node.status != NodeStatus.ONLINE
            if offline and not sweep_offline_nodes:
                out[node.name] = {
                    "skipped": True,
                    "reason": "node_offline",
                    "would_target_gpu_indices": allowed,
                }
                continue
            try:
                res = await self.node_client.kill_gpu_processes(
                    node.name, allowed, vllm_only=vllm_only,
                )
                entry: dict = {"ok": True, "response": res}
                if offline:
                    entry["note"] = (
                        "registry showed OFFLINE; sweep was attempted anyway "
                        "(sweep_offline_nodes=true)"
                    )
                out[node.name] = entry
            except Exception as e:
                log.error("cluster stop: orphan GPU sweep failed",
                          node=node.name, error=str(e))
                out[node.name] = {"ok": False, "error": str(e)}
        return out

    def _gpu_idle_threshold_mb(self, total_mb: int, max_used_mb: int, max_frac: float) -> int:
        return max(int(max_used_mb), int(total_mb * max_frac))

    async def _verify_managed_gpus_idle(
        self,
        *,
        max_used_mb: int,
        max_frac: float,
    ) -> dict:
        """Poll each ONLINE node via agent; confirm managed GPUs look idle after stop.

        Uses the same GPU indices as ``cuda_devices`` (``node.gpus`` in the registry).
        """
        nodes_out: dict[str, dict] = {}
        all_passed = True
        for node in self.registry.get_all_nodes():
            allowed = {g.index for g in node.gpus}
            if not allowed:
                nodes_out[node.name] = {"skipped": True, "reason": "no_managed_gpus"}
                continue
            if node.status != NodeStatus.ONLINE:
                nodes_out[node.name] = {
                    "skipped": True,
                    "reason": "node_offline",
                    "managed_indices": sorted(allowed),
                }
                continue
            try:
                status = await self.node_client.get_node_status(node.name)
            except Exception as e:
                all_passed = False
                nodes_out[node.name] = {"ok": False, "error": str(e)}
                continue
            gpus = status.get("gpus") or []
            by_idx = {int(g["index"]): g for g in gpus if "index" in g}
            checked: list[dict] = []
            issues: list[str] = []
            for idx in sorted(allowed):
                g = by_idx.get(idx)
                if not g:
                    issues.append(f"GPU {idx}: missing from agent status")
                    all_passed = False
                    continue
                total = int(g.get("memory_total_mb") or 0)
                used = int(g.get("memory_used_mb") or 0)
                thr = self._gpu_idle_threshold_mb(total, max_used_mb, max_frac)
                st = str(g.get("state") or "")
                rep_id = g.get("allocated_replica_id") or ""
                ok_mem = used <= thr
                ok_alloc = st != "ALLOCATED" or not rep_id
                entry = {
                    "index": idx,
                    "memory_used_mb": used,
                    "memory_total_mb": total,
                    "idle_threshold_mb": thr,
                    "memory_ok": ok_mem,
                    "state": st,
                    "allocated_replica_id": rep_id,
                    "allocation_ok": ok_alloc,
                }
                checked.append(entry)
                if not ok_mem:
                    issues.append(
                        f"GPU {idx}: {used}MB used > idle threshold {thr}MB",
                    )
                    all_passed = False
                if not ok_alloc:
                    issues.append(
                        f"GPU {idx}: still ALLOCATED to replica {rep_id}",
                    )
                    all_passed = False
            nodes_out[node.name] = {
                "ok": len(issues) == 0,
                "checked": checked,
                "issues": issues,
            }
        return {
            "all_passed": all_passed,
            "nodes": nodes_out,
        }

    async def _run_cluster_stop(
        self,
        *,
        purge_queues: bool,
        wait_after_stop_s: float,
        second_wait_s: float,
        cleanup_orphans: bool,
        force: bool,
        resume_autoscale_after: bool,
        verify_gpus: bool,
        verify_max_used_mb: int,
        verify_max_frac: float,
        retry_sweep_on_verify_fail: bool,
        vllm_only_gpu_sweep: bool,
        sweep_offline_nodes: bool,
    ) -> dict:
        """Orchestrate full cluster inference shutdown + optional GPU sweep."""
        log.warning(
            "cluster stop initiated",
            purge_queues=purge_queues,
            force=force,
            resume_autoscale_after=resume_autoscale_after,
            vllm_only_gpu_sweep=vllm_only_gpu_sweep,
            sweep_offline_nodes=sweep_offline_nodes,
        )
        self.autoscaler.set_paused(True)
        report: dict = {
            "replicas": [],
            "queues": None,
            "orphan_cleanup_passes": [],
            "gpu_sweep_policy": {
                "vllm_only": vllm_only_gpu_sweep,
                "sweep_offline_nodes": sweep_offline_nodes,
                "scope": (
                    "all processes on cuda_devices (nvidia-smi)"
                    if not vllm_only_gpu_sweep
                    else "vllm/python-named processes only"
                ),
            },
            "gpu_verification": None,
            "autoscaler_paused": True,
            "resume_autoscale_after": resume_autoscale_after,
            "control_plane_process": (
                "still running — this endpoint stops replicas and clears Redis; "
                "stop the systemd/supervisor unit separately if you need the "
                "head process down."
            ),
        }
        try:
            if purge_queues:
                report["queues"] = await self.queue.purge_queues(
                    models=None,
                    purge_callback_keys=True,
                    purge_cancelled_keys=True,
                )

            replica_ids = [r.replica_id for r in self.registry.get_all_replicas()]
            for rid in list(replica_ids):
                entry = await self._cluster_stop_single_replica(rid, force=force)
                report["replicas"].append(entry)

            await asyncio.sleep(max(0.0, wait_after_stop_s))

            if cleanup_orphans:
                report["orphan_cleanup_passes"].append(
                    await self._orphan_gpu_sweep(
                        vllm_only=vllm_only_gpu_sweep,
                        sweep_offline_nodes=sweep_offline_nodes,
                    ),
                )
                if second_wait_s > 0:
                    await asyncio.sleep(second_wait_s)
                    report["orphan_cleanup_passes"].append(
                        await self._orphan_gpu_sweep(
                            vllm_only=vllm_only_gpu_sweep,
                            sweep_offline_nodes=sweep_offline_nodes,
                        ),
                    )

            if verify_gpus:
                v1 = await self._verify_managed_gpus_idle(
                    max_used_mb=verify_max_used_mb,
                    max_frac=verify_max_frac,
                )
                report["gpu_verification"] = v1
                if (
                    not v1.get("all_passed")
                    and cleanup_orphans
                    and retry_sweep_on_verify_fail
                ):
                    log.warning(
                        "cluster stop: GPU verification failed, retrying orphan sweep",
                    )
                    await asyncio.sleep(4.0)
                    report["orphan_cleanup_passes"].append(
                        await self._orphan_gpu_sweep(
                            vllm_only=vllm_only_gpu_sweep,
                            sweep_offline_nodes=sweep_offline_nodes,
                        ),
                    )
                    await asyncio.sleep(2.0)
                    report["gpu_verification"] = await self._verify_managed_gpus_idle(
                        max_used_mb=verify_max_used_mb,
                        max_frac=verify_max_frac,
                    )

            # Drop stale GPU-guard samples/timers so Alerts/Topology do not keep
            # showing pre-stop compute pressure until the next guard interval.
            if self.gpu_guard:
                self.gpu_guard.reset_per_gpu_tracking_after_cluster_stop()
                report["gpu_guard_tracking_reset"] = True

            if resume_autoscale_after:
                self.autoscaler.set_paused(False)
                report["autoscaler_paused"] = False
            elif report["autoscaler_paused"]:
                report["hint"] = (
                    "Autoscaling is paused — no new replicas until "
                    "POST /admin/cluster/resume."
                )

            return report
        except Exception as e:
            log.error("cluster stop failed", error=str(e))
            report["fatal_error"] = str(e)
            return report

    # ─── Health Manager callbacks ────────────────────────────────────────

    async def _restart_replica_in_place(
        self, replica_id: str, model_name: str,
        node_name: str, gpu_ids: list[int],
    ) -> None:
        """Restart a replica on the same GPUs after an in-place recoverable failure."""
        model_cfg = self.models_config.get(model_name)
        if not model_cfg:
            raise ValueError(f"Unknown model for restart: {model_name}")

        try:
            self.registry.remove_replica(replica_id)
        except Exception:
            pass

        new_id = uuid.uuid4().hex[:12]
        replica = ReplicaState(
            replica_id=new_id,
            model=model_name,
            node_name=node_name,
            gpu_ids=gpu_ids,
            tp_size=model_cfg.tp,
        )

        self.registry.allocate_gpus(node_name, gpu_ids, new_id)
        self.registry.add_replica(replica)

        engine_args = Orchestrator._build_engine_args(model_cfg)
        env_vars = {}
        if model_cfg.hf_token:
            env_vars["HF_TOKEN"] = model_cfg.hf_token

        try:
            result = await self.node_client.launch_replica(
                node_name=node_name,
                replica_id=new_id,
                model_name=model_name,
                hf_model=model_cfg.hf_model,
                served_model_name=model_cfg.served_model_name,
                variant="default",
                gpu_ids=gpu_ids,
                tp_size=model_cfg.tp,
                engine_args=engine_args,
                env_vars=env_vars,
                health_timeout_s=model_cfg.autoscaling.replica_startup_timeout_s,
            )
            self.registry.update_replica_endpoint(
                new_id, result.get("http_endpoint", ""),
                0, result.get("pid", 0),
            )
            log.info("in-place restart launched",
                     model=model_name, old_replica=replica_id,
                     new_replica=new_id, node=node_name, gpus=gpu_ids)
        except Exception as e:
            log.error("in-place restart launch failed",
                      model=model_name, new_replica=new_id, error=str(e))
            self.registry.remove_replica(new_id)
            raise

    # ─── GPU Guard callbacks ─────────────────────────────────────────────

    async def _pause_replica_for_mitigation(self, replica_id: str) -> None:
        """Pause a replica by setting it to DRAINING so the gateway stops routing."""
        replica = self.registry.get_replica(replica_id)
        if not replica:
            return
        if replica.status == ReplicaStatus.READY:
            self.registry.set_replica_status(replica_id, ReplicaStatus.DRAINING)
            log.info("replica paused for GPU mitigation",
                     replica=replica_id, model=replica.model,
                     node=replica.node_name)
            await self.db.log_health_incident(
                incident_type="gpu_guard_pause",
                node_name=replica.node_name,
                replica_id=replica_id,
                details=(f"Paused {replica.model} for GPU memory "
                         f"mitigation on {replica.node_name}"),
            )

    async def _resume_replica_after_mitigation(self, replica_id: str) -> None:
        """Resume a paused replica after GPU memory recovered."""
        replica = self.registry.get_replica(replica_id)
        if not replica:
            return
        if replica.status == ReplicaStatus.DRAINING:
            self.registry.set_replica_status(replica_id, ReplicaStatus.READY)
            log.info("replica resumed after GPU self-heal",
                     replica=replica_id, model=replica.model,
                     node=replica.node_name)
            await self.db.log_health_incident(
                incident_type="gpu_guard_resume",
                node_name=replica.node_name,
                replica_id=replica_id,
                details=(f"Resumed {replica.model} after GPU memory "
                         f"recovered on {replica.node_name}"),
            )

    async def _migrate_replica(self, replica_id: str) -> None:
        """Live-migrate a replica: drain → stop → find new placement → launch.

        The model's other replicas keep serving during this process, so
        users experience no downtime — just temporarily reduced capacity.
        """
        replica = self.registry.get_replica(replica_id)
        if not replica:
            raise ValueError(f"Cannot migrate unknown replica: {replica_id}")

        model_name = replica.model
        old_node = replica.node_name
        old_gpus = list(replica.gpu_ids)
        model_cfg = self.models_config.get(model_name)
        if not model_cfg:
            raise ValueError(f"Unknown model config: {model_name}")

        log.info("live migration starting",
                 replica=replica_id, model=model_name,
                 from_node=old_node, from_gpus=old_gpus)

        await self.db.log_health_incident(
            incident_type="gpu_guard_migrate_start",
            node_name=old_node,
            replica_id=replica_id,
            details=(f"Migrating {model_name} away from "
                     f"{old_node} GPUs {old_gpus}"),
        )

        # 1. Stop the old replica (drain + kill)
        try:
            if replica.status not in (ReplicaStatus.STOPPING, ReplicaStatus.FAILED):
                if replica.status == ReplicaStatus.READY:
                    self.registry.set_replica_status(
                        replica_id, ReplicaStatus.DRAINING)
                drain_timeout = int(model_cfg.autoscaling.drain_timeout_s)
                await self.node_client.drain_replica(
                    old_node, replica_id, timeout_s=drain_timeout)
        except Exception as e:
            log.warning("drain during migration failed",
                        replica=replica_id, error=str(e))

        try:
            if replica.status == ReplicaStatus.DRAINING:
                self.registry.set_replica_status(
                    replica_id, ReplicaStatus.STOPPING)
            await self.node_client.stop_replica(
                old_node, replica_id, force=True)
        except Exception as e:
            log.warning("stop during migration failed",
                        replica=replica_id, error=str(e))
        finally:
            self.registry.remove_replica(replica_id)

        # 2. Launch a replacement (prefer a node other than the hot one)
        try:
            await self._launch_replica(model_name, avoid_node=old_node)
            log.info("live migration complete",
                     model=model_name, from_node=old_node,
                     from_gpus=old_gpus)
            await self.db.log_health_incident(
                incident_type="gpu_guard_migrate_done",
                node_name=old_node,
                replica_id=replica_id,
                details=(f"Migrated {model_name} from "
                         f"{old_node} GPUs {old_gpus}"),
            )
        except Exception as e:
            log.error("replacement launch during migration failed",
                      model=model_name, error=str(e))
            await self.db.log_health_incident(
                incident_type="gpu_guard_migrate_failed",
                node_name=old_node,
                replica_id=replica_id,
                details=f"Failed to launch replacement: {e}",
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="CServe Control Plane")
    parser.add_argument("--cluster-config", default="configs/cluster.yaml",
                        help="Path to cluster.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml",
                        help="Path to models.yaml")
    parser.add_argument("--db-path", default="/var/lib/cserve/events.db",
                        help="Path to SQLite event log")
    parser.add_argument("--host", default=None, help="Override gateway host")
    parser.add_argument("--port", type=int, default=None, help="Override gateway port")
    args = parser.parse_args()

    server = ControlPlaneServer(args.cluster_config, args.models_config, args.db_path)

    host = args.host or server.cluster_config.gateway.host
    port = args.port or server.cluster_config.gateway.port

    config = uvicorn.Config(
        app=server.app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    uvicorn_server = uvicorn.Server(config)

    async def run():
        await server.start()
        try:
            await uvicorn_server.serve()
        finally:
            await server.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
