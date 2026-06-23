"""Node Agent Client — how the control plane talks to node agents.

Each node agent exposes a JSON-over-HTTP API.  This client wraps those
calls with retries, timeouts, and structured error handling.

Used by:
  - Autoscaler (launch/stop replicas)
  - Health manager (ping, kill GPU processes)
  - Server (initial bootstrap)
"""

from __future__ import annotations

import httpx

from cserve.common.logging import get_logger

log = get_logger("node_client")


class NodeAgentClient:
    """HTTP client for communicating with node agents."""

    def __init__(self, registry) -> None:
        self.registry = registry
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    def _endpoint(self, node_name: str) -> str:
        node = self.registry.get_node(node_name)
        if not node:
            raise ValueError(f"Unknown node: {node_name}")
        return f"http://{node.agent_endpoint}"

    async def ping(self, node_name: str) -> bool:
        try:
            resp = await self._client.get(f"{self._endpoint(node_name)}/ping")
            return resp.status_code == 200
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
        health_timeout_s: float | None = None,
    ) -> dict:
        """Tell a node agent to launch a vLLM replica."""
        payload = {
            "replica_id": replica_id,
            "model_name": model_name,
            "hf_model": hf_model,
            "served_model_name": served_model_name,
            "variant": variant,
            "gpu_ids": gpu_ids,
            "tp_size": tp_size,
            "port": port,
            "engine_args": engine_args or {},
            "env_vars": env_vars or {},
        }
        if health_timeout_s is not None:
            payload["health_timeout_s"] = float(health_timeout_s)
        endpoint = f"{self._endpoint(node_name)}/launch_replica"
        try:
            resp = await self._client.post(endpoint, json=payload)
        except Exception as e:
            raise RuntimeError(
                f"Launch HTTP request to {node_name} failed: {type(e).__name__}: {e}"
            ) from e
        try:
            result = resp.json()
        except Exception:
            raise RuntimeError(
                f"Launch on {node_name}: non-JSON response (status={resp.status_code}): "
                f"{resp.text[:500]}"
            )
        if not result.get("ok"):
            raise RuntimeError(
                f"Launch failed on {node_name}: {result.get('error', 'unknown')}"
            )
        return result

    async def stop_replica(
        self, node_name: str, replica_id: str, force: bool = False,
    ) -> dict:
        resp = await self._client.post(
            f"{self._endpoint(node_name)}/stop_replica",
            json={"replica_id": replica_id, "force": force},
            # Graceful stop waits up to 30s inside the node agent before it
            # escalates to SIGKILL. Give the agent room to return the final
            # result so the control plane can remove/relaunch the replica.
            timeout=90.0,
        )
        return resp.json()

    async def drain_replica(
        self, node_name: str, replica_id: str, timeout_s: int = 60,
    ) -> dict:
        resp = await self._client.post(
            f"{self._endpoint(node_name)}/drain_replica",
            json={"replica_id": replica_id, "timeout_s": timeout_s},
            timeout=timeout_s + 10,
        )
        return resp.json()

    async def get_node_status(self, node_name: str) -> dict:
        resp = await self._client.get(f"{self._endpoint(node_name)}/node_status")
        return resp.json()

    async def get_replica_status(
        self,
        node_name: str,
        replica_id: str,
        *,
        log_offset: int = 0,
        max_lines: int = 200,
        timeout_s: float = 30.0,
    ) -> dict:
        try:
            resp = await self._client.get(
                f"{self._endpoint(node_name)}/replica_status/{replica_id}",
                params={"offset": log_offset, "max_lines": max_lines},
                timeout=timeout_s,
            )
            return resp.json()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    async def kill_gpu_processes(
        self, node_name: str, gpu_ids: list[int], vllm_only: bool = False,
    ) -> dict:
        resp = await self._client.post(
            f"{self._endpoint(node_name)}/kill_gpu_processes",
            json={"gpu_ids": gpu_ids, "vllm_only": vllm_only},
        )
        return resp.json()
