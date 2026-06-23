"""Node Agent — the worker-side gRPC server.

Runs on every GPU node.  Implements the NodeAgent gRPC service defined in
proto/node_agent.proto.  Since we don't want to require proto compilation
at install time, this uses a lightweight JSON-over-HTTP approach instead of
compiled protobuf stubs.

The agent:
  - Manages vLLM replica lifecycle via the Launcher.
  - Reports GPU metrics via the GpuMonitor.
  - Sends heartbeats to the control plane.
  - Cleans up zombie GPU processes.

Wire protocol: JSON-over-HTTP (FastAPI), mapped 1:1 to the proto messages.
This is simpler to debug than binary gRPC and avoids the proto compilation step.
When we need sub-millisecond latencies on the control channel, we can switch
to gRPC — the proto file is ready.
"""

from __future__ import annotations

import asyncio
import os
import platform
import time

import httpx
from fastapi import FastAPI

from cserve.common.logging import get_logger
from cserve.node_agent.gpu_monitor import GpuMonitor
from cserve.node_agent.launcher import Launcher

log = get_logger("node_agent")


class NodeAgent:
    """The agent running on each GPU node."""

    def __init__(
        self,
        node_name: str,
        node_host: str,
        control_plane_url: str,
        agent_port: int = 50051,
        heartbeat_interval_s: int = 10,
    ) -> None:
        self.node_name = node_name
        self.node_host = node_host
        self.control_plane_url = control_plane_url.rstrip("/")
        self.agent_port = agent_port
        self.heartbeat_interval_s = heartbeat_interval_s

        self.launcher = Launcher(node_name, node_host)
        self.gpu_monitor = GpuMonitor(node_name)
        self._started_at = time.time()
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._http_client: httpx.AsyncClient | None = None

        self.app = FastAPI(title=f"CServe Agent - {node_name}", docs_url=None, redoc_url=None)
        self._register_routes()

    async def startup(self) -> None:
        self._running = True
        await self._cleanup_stale_vllm()
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0))
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        log.info("node agent started", node=self.node_name, port=self.agent_port)

    async def _cleanup_stale_vllm(self) -> None:
        """Do not kill vLLM just because the node agent restarted.

        The control plane includes expected replica IDs in heartbeats; orphan
        reconciliation can clean truly stale processes after the agent has that
        authoritative view.  Eager startup scrubbing made agent restarts look
        like model crashes and caused avoidable user-facing downtime.
        """
        log.info("startup vLLM scrub skipped; heartbeat reconciliation will clean true orphans")

    @staticmethod
    def _managed_gpu_indices() -> list[int]:
        """Return list of GPU indices this agent manages.

        Reads from CSERVE_CUDA_DEVICES env var (set from cluster.yaml
        cuda_devices) or returns [] to mean 'all'.
        """
        raw = os.environ.get("CSERVE_CUDA_DEVICES", "")
        if not raw:
            return []
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            return []

    async def shutdown(self) -> None:
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self.launcher.stop_all()
        if self._http_client:
            await self._http_client.aclose()
        log.info("node agent stopped", node=self.node_name)

    # ─── Heartbeat ────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the control plane with GPU status."""
        while self._running:
            try:
                gpus = await self.gpu_monitor.query_gpus()
                payload = {
                    "node_name": self.node_name,
                    "gpus": [g.model_dump() for g in gpus],
                    "uptime_s": time.time() - self._started_at,
                    "replicas": self._replica_summaries(),
                }
                resp = await self._http_client.post(
                    f"{self.control_plane_url}/internal/heartbeat",
                    json=payload,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    body = resp.json()
                    expected = set(body.get("expected_replica_ids", []))
                    report = await self.launcher.reconcile_orphans(
                        expected, self.gpu_monitor,
                    )
                    if any(report.values()):
                        log.warning("orphan reconcile", **report)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("heartbeat failed", error=str(e))
            await asyncio.sleep(self.heartbeat_interval_s)

    def _replica_summaries(self) -> list[dict]:
        """Build a summary of all replicas on this node."""
        result = []
        for rp in self.launcher.all_replicas():
            alive = self.launcher.is_alive(rp.replica_id)
            result.append({
                "replica_id": rp.replica_id,
                "model_name": rp.model_name,
                "variant": rp.variant,
                "gpu_ids": rp.gpu_ids,
                "port": rp.port,
                "pid": rp.pid,
                "alive": alive,
                "uptime_s": time.time() - rp.started_at if rp.started_at else 0,
            })
        return result

    # ─── HTTP routes (matching proto RPCs) ────────────────────────────────

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/health")
        async def health():
            return {"ok": True, "node": self.node_name}

        @app.get("/ping")
        async def ping():
            return {
                "node_name": self.node_name,
                "hostname": platform.node(),
                "uptime_s": time.time() - self._started_at,
            }

        @app.post("/launch_replica")
        async def launch_replica(req: dict):
            return await self._handle_launch(req)

        @app.post("/stop_replica")
        async def stop_replica(req: dict):
            return await self._handle_stop(req)

        @app.post("/drain_replica")
        async def drain_replica(req: dict):
            return await self._handle_drain(req)

        @app.get("/node_status")
        async def node_status():
            return await self._handle_node_status()

        @app.get("/replica_status/{replica_id}")
        async def replica_status(
            replica_id: str,
            offset: int = 0,
            max_lines: int = 200,
        ):
            return await self._handle_replica_status(
                replica_id, log_offset=offset, max_lines=max_lines,
            )

        @app.post("/kill_gpu_processes")
        async def kill_gpu_processes(req: dict):
            return await self._handle_kill_gpu_processes(req)

    async def _handle_launch(self, req: dict) -> dict:
        """Launch a vLLM replica."""
        try:
            rp = await self.launcher.launch(
                replica_id=req["replica_id"],
                model_name=req["model_name"],
                hf_model=req["hf_model"],
                served_model_name=req.get("served_model_name", req["model_name"]),
                variant=req.get("variant", "default"),
                gpu_ids=req["gpu_ids"],
                tp_size=req.get("tp_size", len(req["gpu_ids"])),
                port=req.get("port", 0),
                engine_args=req.get("engine_args"),
                env_vars=req.get("env_vars"),
            )

            # Wait for health in background so we can return the PID immediately
            endpoint = f"http://{self.node_host}:{rp.port}"
            asyncio.create_task(self._wait_and_report_ready(rp.replica_id, endpoint))

            return {
                "ok": True,
                "replica_id": rp.replica_id,
                "http_endpoint": endpoint,
                "pid": rp.pid,
            }
        except Exception as e:
            log.error("launch failed", error=str(e))
            return {"ok": False, "error": str(e)}

    async def _wait_and_report_ready(self, replica_id: str, endpoint: str) -> None:
        """Wait for vLLM to become healthy, then notify the control plane."""
        healthy = await self.launcher.wait_for_health(replica_id)
        status = "READY" if healthy else "FAILED"

        try:
            await self._http_client.post(
                f"{self.control_plane_url}/internal/replica_status",
                json={
                    "node_name": self.node_name,
                    "replica_id": replica_id,
                    "status": status,
                    "endpoint": endpoint if healthy else "",
                },
                timeout=5.0,
            )
        except Exception as e:
            log.error("failed to report replica status", replica=replica_id, error=str(e))

    async def _handle_stop(self, req: dict) -> dict:
        replica_id = req.get("replica_id", "")
        force = req.get("force", False)
        try:
            ok = await self.launcher.stop(replica_id, force=force)
            return {"ok": ok}
        except Exception as e:
            log.error("stop failed", replica=replica_id, error=str(e))
            return {"ok": False, "error": str(e)}

    async def _handle_drain(self, req: dict) -> dict:
        replica_id = req.get("replica_id", "")
        timeout_s = req.get("timeout_s", 60)

        rp = self.launcher.get_replica(replica_id)
        if not rp:
            return {"ok": False, "error": f"Unknown replica: {replica_id}"}

        # Draining means we stop accepting new requests.
        # We wait until in-flight drops to 0 (checked via vLLM /metrics).
        log.info("draining replica", replica=replica_id, timeout_s=timeout_s)
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"http://127.0.0.1:{rp.port}/metrics")
                    if "vllm:num_requests_running" in resp.text:
                        for line in resp.text.split("\n"):
                            if line.startswith("vllm:num_requests_running"):
                                val = float(line.split()[-1])
                                if val == 0:
                                    return {"ok": True, "drained_requests": 0}
            except Exception:
                pass
            await asyncio.sleep(2.0)

        return {"ok": True, "drained_requests": 0, "timed_out": True}

    async def _handle_node_status(self) -> dict:
        gpus = await self.gpu_monitor.query_gpus()
        load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0

        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            ram_total = ram_used = 0
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    ram_total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    ram_used = ram_total - int(line.split()[1]) // 1024
        except Exception:
            ram_total = ram_used = 0

        return {
            "node_name": self.node_name,
            "gpus": [g.model_dump() for g in gpus],
            "replicas": self._replica_summaries(),
            "uptime_s": time.time() - self._started_at,
            "load_avg_1m": load_avg,
            "ram_used_mb": ram_used,
            "ram_total_mb": ram_total,
        }

    # Keep node-agent status responsive.  The control plane treats an alive
    # large-model process with slow /health as serving-capable, so this timeout
    # should not stall health loops for tens of seconds per replica.
    VLLM_HEALTH_PROBE_TIMEOUT_S = 8.0

    async def _probe_vllm_health(
        self, port: int, timeout_s: float | None = None,
    ) -> bool:
        if timeout_s is None:
            timeout_s = self.VLLM_HEALTH_PROBE_TIMEOUT_S
        if not port:
            return False
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/health")
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _replica_log_path(replica_id: str, rp) -> str:
        from pathlib import Path

        if rp and getattr(rp, "stdout_log_path", ""):
            return rp.stdout_log_path
        return str(Path.home() / ".cserve" / "logs" / f"vllm-{replica_id}.log")

    async def _handle_replica_status(
        self,
        replica_id: str,
        *,
        log_offset: int = 0,
        max_lines: int = 200,
    ) -> dict:
        rp = self.launcher.get_replica(replica_id)
        store_entry = self.launcher.store.get(replica_id) if not rp else None

        if not rp and store_entry:
            # Agent restarted but vLLM may still be running (persistent store).
            import os
            pid_alive = False
            if store_entry.pid:
                try:
                    os.kill(store_entry.pid, 0)
                    pid_alive = True
                except OSError:
                    pid_alive = False
            health_ok = pid_alive and await self._probe_vllm_health(store_entry.port)
            from cserve.node_agent.launcher import read_replica_log

            log_path = self._replica_log_path(replica_id, None)
            log_chunk = read_replica_log(
                log_path, offset=log_offset, max_lines=max_lines,
            )
            return {
                "replica": {
                    "replica_id": replica_id,
                    "model_name": store_entry.model_name,
                    "gpu_ids": store_entry.gpu_ids,
                    "port": store_entry.port,
                    "pid": store_entry.pid,
                    "status": "READY" if health_ok else "FAILED",
                    "health_ok": health_ok,
                    "alive": pid_alive,
                    "uptime_s": time.time() - store_entry.launched_at,
                    "tracked_in_memory": False,
                },
                "exit_code": None,
                "output_tail": log_chunk.get("text") or "",
                "log_offset": log_chunk.get("offset", 0),
                "log_size": log_chunk.get("size", 0),
                "log_path": log_chunk.get("log_path") or log_path,
                "vllm_metrics": {},
            }

        if not rp:
            return {"error": f"Unknown replica: {replica_id}"}

        alive = self.launcher.is_alive(replica_id)
        health_ok = alive and await self._probe_vllm_health(rp.port)
        metrics = {}

        if alive and rp.port:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"http://127.0.0.1:{rp.port}/metrics")
                    if resp.status_code == 200:
                        for line in resp.text.split("\n"):
                            if line and not line.startswith("#"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        metrics[parts[0]] = float(parts[1])
                                    except ValueError:
                                        pass
            except Exception:
                pass

        from cserve.node_agent.launcher import read_replica_log

        exit_code = rp.proc.returncode if (not alive and rp.proc) else None
        log_path = self._replica_log_path(replica_id, rp)
        log_chunk = read_replica_log(
            log_path, offset=log_offset, max_lines=max_lines,
        )
        output_tail = log_chunk.get("text") or ""

        return {
            "replica": {
                "replica_id": rp.replica_id,
                "model_name": rp.model_name,
                "variant": rp.variant,
                "gpu_ids": rp.gpu_ids,
                "port": rp.port,
                "pid": rp.pid,
                "status": "READY" if health_ok else "FAILED",
                "health_ok": health_ok,
                "alive": alive,
                "uptime_s": time.time() - rp.started_at,
            },
            "exit_code": exit_code,
            "output_tail": output_tail,
            "log_offset": log_chunk.get("offset", 0),
            "log_size": log_chunk.get("size", 0),
            "log_path": log_chunk.get("log_path") or log_path,
            "vllm_metrics": metrics,
        }

    async def _handle_kill_gpu_processes(self, req: dict) -> dict:
        gpu_ids = req.get("gpu_ids", [])
        vllm_only = req.get("vllm_only", False)
        try:
            killed = await self.gpu_monitor.kill_processes(gpu_ids, vllm_only=vllm_only)
            return {"ok": True, "killed_pids": killed}
        except Exception as e:
            return {"ok": False, "error": str(e), "killed_pids": []}
