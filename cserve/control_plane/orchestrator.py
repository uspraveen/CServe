"""Orchestrator — replica lifecycle management with retries and rollback.

This module handles the full lifecycle of launching and stopping replicas
with proper error handling, retries, and cleanup on failure.

Called by:
  - Autoscaler (scale up/down decisions)
  - Server (initial startup, crash recovery)
  - CLI (manual replica management)

The orchestrator is the single point where GPU allocation + node agent calls
+ registry updates happen atomically (or get rolled back on failure).
"""

from __future__ import annotations

import asyncio
import uuid

from cserve.common.logging import get_logger
from cserve.common.models import (
    JobEvent,
    JobEventRecord,
    ModelConfig,
    ReplicaState,
    ReplicaStatus,
)
from cserve.control_plane.placement import find_placement

log = get_logger("orchestrator")

MAX_LAUNCH_RETRIES = 2
LAUNCH_RETRY_DELAY_S = 5.0


class Orchestrator:
    """Manages replica lifecycle with retries and proper cleanup."""

    def __init__(self, registry, node_client, db, models_config) -> None:
        self.registry = registry
        self.node_client = node_client
        self.db = db
        self.models_config = models_config

    async def launch_replica(self, model_name: str) -> str | None:
        """Launch a new replica for a model. Returns replica_id on success, None on failure."""
        model_cfg = self.models_config.get(model_name)
        if not model_cfg:
            log.error("unknown model", model=model_name)
            return None

        for attempt in range(MAX_LAUNCH_RETRIES + 1):
            try:
                replica_id = await self._try_launch(model_name, model_cfg)
                return replica_id
            except Exception as e:
                log.warning("launch attempt failed",
                            model=model_name, attempt=attempt + 1,
                            max_attempts=MAX_LAUNCH_RETRIES + 1, error=str(e))
                if attempt < MAX_LAUNCH_RETRIES:
                    await asyncio.sleep(LAUNCH_RETRY_DELAY_S)

        log.error("all launch attempts exhausted", model=model_name)
        return None

    async def _try_launch(self, model_name: str, model_cfg: ModelConfig) -> str:
        """Single launch attempt. Raises on failure."""
        # Find placement
        nodes = self.registry.get_available_nodes()
        existing_counts = {}
        for node in nodes:
            existing_counts[node.name] = sum(
                1 for r in self.registry.get_replicas_for_model(model_name)
                if r.node_name == node.name
            )

        placement = find_placement(model_cfg, nodes, existing_counts)
        if not placement:
            raise RuntimeError(f"No placement available for {model_name} (tp={model_cfg.tp})")

        replica_id = uuid.uuid4().hex[:12]
        replica = ReplicaState(
            replica_id=replica_id,
            model=model_name,
            node_name=placement.node_name,
            gpu_ids=placement.gpu_indices,
            tp_size=model_cfg.tp,
        )

        # Allocate GPUs
        self.registry.allocate_gpus(placement.node_name, placement.gpu_indices, replica_id)
        self.registry.add_replica(replica)

        try:
            engine_args = self._build_engine_args(model_cfg)
            env_vars = {}
            if model_cfg.hf_token:
                env_vars["HF_TOKEN"] = model_cfg.hf_token

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

            log.info("replica launch initiated",
                     model=model_name, replica=replica_id,
                     node=placement.node_name, gpus=placement.gpu_indices)

            return replica_id

        except Exception as e:
            # Rollback: release GPUs and remove replica
            log.error("launch failed, rolling back", replica=replica_id, error=str(e))
            self.registry.remove_replica(replica_id)
            raise

    async def stop_replica(self, replica_id: str, force: bool = False) -> bool:
        """Stop a replica with drain → stop → cleanup lifecycle.

        Returns True if successfully stopped.
        """
        replica = self.registry.get_replica(replica_id)
        if not replica:
            log.warning("stop requested for unknown replica", replica=replica_id)
            return False

        model_cfg = self.models_config.get(replica.model)
        drain_timeout = int(model_cfg.autoscaling.drain_timeout_s) if model_cfg else 60

        # Drain phase (skip if force)
        if not force and replica.status.can_accept_requests():
            try:
                self.registry.set_replica_status(replica_id, ReplicaStatus.DRAINING)
                await self.node_client.drain_replica(
                    replica.node_name, replica_id, timeout_s=drain_timeout,
                )
                log.info("replica drained", replica=replica_id)
            except Exception as e:
                log.warning("drain failed", replica=replica_id, error=str(e))

        # Stop phase
        try:
            if replica.status == ReplicaStatus.DRAINING:
                self.registry.set_replica_status(replica_id, ReplicaStatus.STOPPING)
            elif replica.status.can_accept_requests():
                self.registry.set_replica_status(replica_id, ReplicaStatus.DRAINING)
                self.registry.set_replica_status(replica_id, ReplicaStatus.STOPPING)

            await self.node_client.stop_replica(
                replica.node_name, replica_id, force=force,
            )
            log.info("replica stopped", replica=replica_id)
        except Exception as e:
            log.error("stop failed", replica=replica_id, error=str(e))

        # Cleanup: always remove from registry
        self.registry.remove_replica(replica_id)

        await self.db.log_job_event(JobEventRecord(
            job_id=f"lifecycle-{replica_id}",
            event=JobEvent.COMPLETED,
            replica_id=replica_id,
            node_name=replica.node_name,
            metadata={"action": "stop", "model": replica.model, "force": force},
        ))

        return True

    async def ensure_min_replicas(self) -> None:
        """Ensure each model has at least min_replicas running. Used at startup."""
        for model_name, model_cfg in self.models_config.items():
            current = self.registry.count_replicas(model_name)
            needed = model_cfg.autoscaling.min_replicas - current

            if needed <= 0:
                continue

            log.info("bootstrapping replicas", model=model_name,
                     current=current, target=model_cfg.autoscaling.min_replicas)

            for _ in range(needed):
                result = await self.launch_replica(model_name)
                if not result:
                    log.error("failed to bootstrap replica", model=model_name)
                    break

    @staticmethod
    def _build_engine_args(model_cfg: ModelConfig) -> dict[str, str]:
        eng = model_cfg.engine
        args: dict[str, str] = {
            "max_model_len": str(eng.max_model_len),
            "max_num_seqs": str(eng.max_num_seqs),
            "gpu_memory_utilization": str(eng.gpu_memory_utilization),
            "dtype": eng.dtype,
        }
        if eng.trust_remote_code:
            args["trust_remote_code"] = "true"
        if eng.enable_chunked_prefill:
            args["enable_chunked_prefill"] = "true"
        if eng.enable_prefix_caching:
            args["enable_prefix_caching"] = "true"
        if eng.disable_custom_all_reduce:
            args["disable_custom_all_reduce"] = "true"
        if eng.reasoning_parser:
            args["reasoning_parser"] = eng.reasoning_parser
        if eng.runner:
            args["runner"] = eng.runner
        if eng.convert:
            args["convert"] = eng.convert
        if eng.kv_cache_dtype:
            args["kv_cache_dtype"] = eng.kv_cache_dtype
        return args
