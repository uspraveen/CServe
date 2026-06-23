"""gRPC server for the Node Agent.

Wraps the existing agent logic (Launcher, GpuMonitor) behind compiled
protobuf stubs.  This replaces the JSON-over-HTTP transport for deployments
that need binary efficiency and typed contracts.

Both transports can coexist: the agent starts gRPC on one port and HTTP on
another if needed.  The gRPC port is the primary production interface.
"""

from __future__ import annotations

import asyncio
import os
import platform
import time
from concurrent import futures

import grpc

from cserve.common.logging import get_logger
from cserve.generated import node_agent_pb2 as pb
from cserve.generated import node_agent_pb2_grpc as pb_grpc
from cserve.node_agent.gpu_monitor import GpuMonitor
from cserve.node_agent.launcher import Launcher

log = get_logger("grpc_agent")


class NodeAgentServicer(pb_grpc.NodeAgentServicer):
    """Implements the NodeAgent gRPC service."""

    def __init__(
        self, node_name: str, node_host: str,
        control_plane_url: str, launcher: Launcher, gpu_monitor: GpuMonitor,
    ) -> None:
        self.node_name = node_name
        self.node_host = node_host
        self.control_plane_url = control_plane_url
        self.launcher = launcher
        self.gpu_monitor = gpu_monitor
        self._started_at = time.time()

    async def Ping(self, request, context):
        return pb.PingResponse(
            node_name=self.node_name,
            hostname=platform.node(),
            uptime_s=time.time() - self._started_at,
        )

    async def LaunchReplica(self, request, context):
        try:
            rp = await self.launcher.launch(
                replica_id=request.replica_id,
                model_name=request.model_name,
                hf_model=request.hf_model,
                served_model_name=request.served_model_name,
                variant=request.variant,
                gpu_ids=list(request.gpu_ids),
                tp_size=request.tp_size,
                port=request.port,
                engine_args=dict(request.engine_args),
                env_vars=dict(request.env_vars),
            )
            endpoint = f"http://{self.node_host}:{rp.port}"
            return pb.LaunchReplicaResponse(
                ok=True, replica_id=rp.replica_id,
                http_endpoint=endpoint, pid=rp.pid,
            )
        except Exception as e:
            log.error("grpc launch failed", error=str(e))
            return pb.LaunchReplicaResponse(ok=False, error=str(e))

    async def StopReplica(self, request, context):
        try:
            ok = await self.launcher.stop(request.replica_id, force=request.force)
            return pb.StopReplicaResponse(ok=ok)
        except Exception as e:
            return pb.StopReplicaResponse(ok=False, error=str(e))

    async def DrainReplica(self, request, context):
        import httpx

        rp = self.launcher.get_replica(request.replica_id)
        if not rp:
            return pb.DrainReplicaResponse(ok=False, error=f"Unknown replica: {request.replica_id}")

        timeout_s = request.timeout_s or 60
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
                                    return pb.DrainReplicaResponse(ok=True, drained_requests=0)
            except Exception:
                pass
            await asyncio.sleep(2.0)

        return pb.DrainReplicaResponse(ok=True, drained_requests=0)

    async def GetNodeStatus(self, request, context):
        gpus = await self.gpu_monitor.query_gpus()
        load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0

        ram_total = ram_used = 0
        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    ram_total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    ram_used = ram_total - int(line.split()[1]) // 1024
        except Exception:
            pass

        gpu_infos = [
            pb.GpuInfo(
                index=g.index, uuid=g.uuid, name=g.name,
                memory_used_mb=g.memory_used_mb, memory_total_mb=g.memory_total_mb,
                utilization_pct=g.utilization_pct, temperature_c=g.temperature_c,
                state=g.state.value, allocated_replica_id=g.allocated_replica_id or "",
            )
            for g in gpus
        ]

        replica_infos = []
        for rp in self.launcher.all_replicas():
            alive = self.launcher.is_alive(rp.replica_id)
            replica_infos.append(pb.ReplicaInfo(
                replica_id=rp.replica_id, model_name=rp.model_name,
                variant=rp.variant, gpu_ids=rp.gpu_ids, port=rp.port,
                pid=rp.pid, status="READY" if alive else "FAILED",
                health_ok=alive, uptime_s=time.time() - rp.started_at,
            ))

        return pb.NodeStatusResponse(
            node_name=self.node_name, gpus=gpu_infos, replicas=replica_infos,
            uptime_s=time.time() - self._started_at,
            load_avg_1m=load_avg, ram_used_mb=ram_used, ram_total_mb=ram_total,
        )

    async def GetReplicaStatus(self, request, context):
        import httpx

        rp = self.launcher.get_replica(request.replica_id)
        if not rp:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown replica: {request.replica_id}")
            return pb.ReplicaStatusResponse()

        alive = self.launcher.is_alive(request.replica_id)
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

        info = pb.ReplicaInfo(
            replica_id=rp.replica_id, model_name=rp.model_name,
            variant=rp.variant, gpu_ids=rp.gpu_ids, port=rp.port,
            pid=rp.pid, status="READY" if alive else "FAILED",
            health_ok=alive, uptime_s=time.time() - rp.started_at,
        )
        return pb.ReplicaStatusResponse(replica=info, vllm_metrics=metrics)

    async def KillGpuProcesses(self, request, context):
        try:
            killed = await self.gpu_monitor.kill_processes(
                list(request.gpu_ids), vllm_only=request.vllm_only,
            )
            return pb.KillGpuProcessesResponse(ok=True, killed_pids=killed)
        except Exception as e:
            return pb.KillGpuProcessesResponse(ok=False, error=str(e))


async def serve_grpc(
    node_name: str, node_host: str, control_plane_url: str,
    port: int = 50051,
) -> grpc.aio.Server:
    """Create and start the gRPC server."""
    launcher = Launcher(node_name, node_host)
    gpu_monitor = GpuMonitor(node_name)

    servicer = NodeAgentServicer(
        node_name, node_host, control_plane_url, launcher, gpu_monitor,
    )

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_NodeAgentServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    await server.start()
    log.info("gRPC server started", node=node_name, port=port)
    return server
