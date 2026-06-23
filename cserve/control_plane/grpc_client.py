"""gRPC client for communicating with node agents.

Drop-in replacement for NodeAgentClient (JSON-over-HTTP).  Both implement the
same interface so the rest of the control plane doesn't care which transport
is active.  Use this when binary efficiency matters (production deployments).
"""

from __future__ import annotations

import grpc

from cserve.common.logging import get_logger
from cserve.generated import node_agent_pb2 as pb
from cserve.generated import node_agent_pb2_grpc as pb_grpc

log = get_logger("grpc_client")


class GrpcNodeAgentClient:
    """gRPC client — same interface as NodeAgentClient (HTTP)."""

    def __init__(self, registry) -> None:
        self.registry = registry
        self._channels: dict[str, grpc.aio.Channel] = {}

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        for ch in self._channels.values():
            await ch.close()
        self._channels.clear()

    def _stub(self, node_name: str) -> pb_grpc.NodeAgentStub:
        node = self.registry.get_node(node_name)
        if not node:
            raise ValueError(f"Unknown node: {node_name}")
        endpoint = node.agent_endpoint
        if endpoint not in self._channels:
            self._channels[endpoint] = grpc.aio.insecure_channel(endpoint)
        return pb_grpc.NodeAgentStub(self._channels[endpoint])

    async def ping(self, node_name: str) -> bool:
        try:
            stub = self._stub(node_name)
            resp = await stub.Ping(pb.PingRequest(), timeout=5.0)
            return bool(resp.node_name)
        except Exception:
            return False

    async def launch_replica(
        self,
        node_name: str,
        replica_id: str,
        model_name: str,
        hf_model: str,
        served_model_name: str,
        variant: str,
        gpu_ids: list[int],
        tp_size: int,
        port: int = 0,
        engine_args: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> dict:
        stub = self._stub(node_name)
        req = pb.LaunchReplicaRequest(
            replica_id=replica_id, model_name=model_name,
            hf_model=hf_model, served_model_name=served_model_name,
            variant=variant, tp_size=tp_size, gpu_ids=gpu_ids,
            port=port, engine_args=engine_args or {},
            env_vars=env_vars or {},
        )
        resp = await stub.LaunchReplica(req, timeout=30.0)
        if not resp.ok:
            raise RuntimeError(f"Launch failed on {node_name}: {resp.error}")
        return {
            "ok": True,
            "replica_id": resp.replica_id,
            "http_endpoint": resp.http_endpoint,
            "pid": resp.pid,
        }

    async def stop_replica(
        self, node_name: str, replica_id: str, force: bool = False,
    ) -> dict:
        stub = self._stub(node_name)
        resp = await stub.StopReplica(
            pb.StopReplicaRequest(replica_id=replica_id, force=force),
            timeout=30.0,
        )
        return {"ok": resp.ok, "error": resp.error}

    async def drain_replica(
        self, node_name: str, replica_id: str, timeout_s: int = 60,
    ) -> dict:
        stub = self._stub(node_name)
        resp = await stub.DrainReplica(
            pb.DrainReplicaRequest(replica_id=replica_id, timeout_s=timeout_s),
            timeout=timeout_s + 10,
        )
        return {"ok": resp.ok, "drained_requests": resp.drained_requests}

    async def get_node_status(self, node_name: str) -> dict:
        stub = self._stub(node_name)
        resp = await stub.GetNodeStatus(pb.NodeStatusRequest(), timeout=10.0)
        return {
            "node_name": resp.node_name,
            "gpus": [
                {
                    "index": g.index, "uuid": g.uuid, "name": g.name,
                    "memory_used_mb": g.memory_used_mb,
                    "memory_total_mb": g.memory_total_mb,
                    "utilization_pct": g.utilization_pct,
                    "temperature_c": g.temperature_c,
                    "state": g.state,
                    "allocated_replica_id": g.allocated_replica_id,
                }
                for g in resp.gpus
            ],
            "replicas": [
                {
                    "replica_id": r.replica_id, "model_name": r.model_name,
                    "variant": r.variant, "gpu_ids": list(r.gpu_ids),
                    "port": r.port, "pid": r.pid, "status": r.status,
                    "health_ok": r.health_ok, "uptime_s": r.uptime_s,
                }
                for r in resp.replicas
            ],
            "uptime_s": resp.uptime_s,
            "load_avg_1m": resp.load_avg_1m,
            "ram_used_mb": resp.ram_used_mb,
            "ram_total_mb": resp.ram_total_mb,
        }

    async def get_replica_status(self, node_name: str, replica_id: str) -> dict:
        stub = self._stub(node_name)
        resp = await stub.GetReplicaStatus(
            pb.ReplicaStatusRequest(replica_id=replica_id),
            timeout=10.0,
        )
        r = resp.replica
        return {
            "replica_id": r.replica_id,
            "model_name": r.model_name,
            "gpu_ids": list(r.gpu_ids),
            "port": r.port,
            "pid": r.pid,
            "status": r.status,
            "health_ok": r.health_ok,
            "uptime_s": r.uptime_s,
            "vllm_metrics": dict(resp.vllm_metrics),
            "output_tail": "",
            "exit_code": None,
        }

    async def kill_gpu_processes(
        self, node_name: str, gpu_ids: list[int], vllm_only: bool = False,
    ) -> dict:
        stub = self._stub(node_name)
        resp = await stub.KillGpuProcesses(
            pb.KillGpuProcessesRequest(gpu_ids=gpu_ids, vllm_only=vllm_only),
            timeout=15.0,
        )
        return {"ok": resp.ok, "error": resp.error, "killed_pids": list(resp.killed_pids)}
