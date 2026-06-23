"""Integration tests — exercise multiple components together.

These tests don't require Redis or GPUs.  They use fakes/stubs for
external dependencies and verify that the core components wire up
correctly.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from cserve.common.models import (
    AutoscaleAction,
    AutoscalePolicy,
    ClusterConfig,
    GatewayConfig,
    HeadConfig,
    ModelConfig,
    NodeAgentConfig,
    NodeConfig,
    NodeStatus,
    RedisConfig,
    ReplicaState,
    ReplicaStatus,
    SafetyConfig,
)
from cserve.control_plane.autoscaler import Autoscaler, ModelScaleState
from cserve.control_plane.orchestrator import Orchestrator
from cserve.control_plane.registry import ClusterRegistry


def _cluster_config() -> ClusterConfig:
    return ClusterConfig(
        head=HeadConfig(name="head", host="10.0.0.1"),
        nodes=[
            NodeConfig(
                name="worker1", host="10.0.0.2",
                gpu_count=4, gpu_type="a40",
                cuda_devices="0,1,2,3", agent_port=50051,
            ),
            NodeConfig(
                name="worker2", host="10.0.0.3",
                gpu_count=4, gpu_type="a40",
                cuda_devices="0,1,2,3", agent_port=50051,
            ),
        ],
        gateway=GatewayConfig(port=8002),
        redis=RedisConfig(),
        safety=SafetyConfig(),
        node_agent=NodeAgentConfig(),
    )


def _models_config() -> dict[str, ModelConfig]:
    return {
        "llama": ModelConfig(
            name="llama",
            served_model_name="llama-3",
            hf_model="meta-llama/Llama-3-8B",
            tp=2,
            node_type_required="a40",
            autoscaling=AutoscalePolicy(min_replicas=1, max_replicas=4),
        ),
        "embed": ModelConfig(
            name="embed",
            served_model_name="embed-v1",
            hf_model="BAAI/bge-base-en-v1.5",
            tp=1,
            autoscaling=AutoscalePolicy(min_replicas=0, max_replicas=2, allow_scale_to_zero=True),
        ),
    }


class TestRegistryModelIntegration:
    """Test that registry + models config + placement work together."""

    def test_load_models_and_query(self):
        reg = ClusterRegistry(_cluster_config())
        models = _models_config()
        reg.load_models(models)

        assert reg.get_model_config("llama") is not None
        assert reg.get_model_config("llama").tp == 2
        assert reg.get_model_config("embed") is not None
        assert reg.get_autoscale_policy("llama").max_replicas == 4

    def test_multi_model_replica_tracking(self):
        reg = ClusterRegistry(_cluster_config())
        reg.load_models(_models_config())
        reg.set_node_status("worker1", NodeStatus.ONLINE)
        reg.set_node_status("worker2", NodeStatus.ONLINE)

        # Add a llama replica on GPUs 0,1 of worker1
        r1 = ReplicaState(replica_id="r1", model="llama", node_name="worker1",
                          gpu_ids=[0, 1], tp_size=2)
        reg.allocate_gpus("worker1", [0, 1], "r1")
        reg.add_replica(r1)
        reg.set_replica_status("r1", ReplicaStatus.READY)

        # Add an embed replica on GPU 2 of worker1
        r2 = ReplicaState(replica_id="r2", model="embed", node_name="worker1",
                          gpu_ids=[2], tp_size=1)
        reg.allocate_gpus("worker1", [2], "r2")
        reg.add_replica(r2)
        reg.set_replica_status("r2", ReplicaStatus.READY)

        assert reg.count_replicas("llama") == 1
        assert reg.count_replicas("embed") == 1
        assert reg.count_ready_replicas("llama") == 1

        # Check GPU allocation
        free = reg.get_free_gpus("worker1")
        assert len(free) == 1  # Only GPU 3 is free
        assert free[0].index == 3

        # Full cleanup
        reg.remove_replica("r1")
        reg.remove_replica("r2")
        assert reg.count_replicas("llama") == 0
        assert len(reg.get_free_gpus("worker1")) == 4


class TestOrchestratorIntegration:
    """Test the orchestrator with a mocked node client."""

    @pytest.fixture
    def setup(self):
        reg = ClusterRegistry(_cluster_config())
        models = _models_config()
        reg.load_models(models)
        reg.set_node_status("worker1", NodeStatus.ONLINE)
        reg.set_node_status("worker2", NodeStatus.ONLINE)

        node_client = AsyncMock()
        node_client.launch_replica = AsyncMock(return_value={
            "ok": True, "replica_id": "test", "http_endpoint": "http://10.0.0.2:8100", "pid": 12345,
        })
        node_client.stop_replica = AsyncMock(return_value={"ok": True})
        node_client.drain_replica = AsyncMock(return_value={"ok": True, "drained_requests": 0})

        db = AsyncMock()
        db.log_job_event = AsyncMock()

        orch = Orchestrator(reg, node_client, db, models)
        return orch, reg, node_client

    @pytest.mark.asyncio
    async def test_launch_allocates_gpus(self, setup):
        orch, reg, node_client = setup
        replica_id = await orch.launch_replica("llama")

        assert replica_id is not None
        replica = reg.get_replica(replica_id)
        assert replica is not None
        assert replica.model == "llama"
        assert replica.tp_size == 2
        assert len(replica.gpu_ids) == 2

        # Node client was called
        node_client.launch_replica.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_rollback_on_failure(self, setup):
        orch, reg, node_client = setup
        node_client.launch_replica = AsyncMock(side_effect=RuntimeError("connection refused"))

        replica_id = await orch.launch_replica("llama")

        # Should fail after retries
        assert replica_id is None
        # No replicas should remain
        assert reg.count_replicas("llama") == 0
        # GPUs should be free
        assert len(reg.get_free_gpus("worker1")) == 4

    @pytest.mark.asyncio
    async def test_stop_drains_then_stops(self, setup):
        orch, reg, node_client = setup
        replica_id = await orch.launch_replica("llama")
        reg.set_replica_status(replica_id, ReplicaStatus.READY)

        ok = await orch.stop_replica(replica_id)
        assert ok is True
        assert reg.get_replica(replica_id) is None
        node_client.drain_replica.assert_called_once()
        node_client.stop_replica.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_spreads_across_nodes(self, setup):
        orch, reg, node_client = setup

        # Launch two llama replicas — should go to different nodes
        r1 = await orch.launch_replica("llama")
        r2 = await orch.launch_replica("llama")

        assert r1 is not None
        assert r2 is not None

        rep1 = reg.get_replica(r1)
        rep2 = reg.get_replica(r2)
        assert rep1.node_name != rep2.node_name, "Replicas should spread across nodes"


class TestAutoscalerDecisionChain:
    """Test a sequence of autoscaling decisions over time."""

    def test_scale_up_then_cool_down_then_scale_down(self):
        autoscaler = Autoscaler.__new__(Autoscaler)
        policy = AutoscalePolicy(
            min_replicas=1, max_replicas=4,
            queue_depth_threshold=3,
            upscale_cooldown_s=10,
            idle_timeout_s=5,
            idle_inflight_threshold=0.5,
            idle_cache_threshold=0.3,
            downscale_cooldown_s=0,
        )
        state = ModelScaleState()
        state.last_scale_up_at = 0.0
        state.last_scale_down_at = 0.0
        state.last_request_at = time.time()

        # Phase 1: Heavy load → scale up
        signals = {
            "current_replicas": 1.0,
            "ready_replicas": 1.0,
            "rps": 0.0,
            "demand_avg_inflight": 0.0,
            "demand_peak_inflight": 0.0,
            "sustained_pressure": 0.0,
            "avg_inflight_per_replica": 8.0,
            "total_inflight": 8.0,
            "queue_depth": 10.0,
            "oldest_queue_age_ms": 5000.0,
            "ttft_p95_ms": 0.0,
            "avg_gpu_cache_usage": 0.0,
            "time_since_last_request_s": 0.0,
        }
        action, reasons, target = autoscaler._decide("test", policy, state, signals)
        assert action == AutoscaleAction.SCALE_UP
        assert target > 1

        # Simulate: we scaled up, cooldown kicks in
        state.last_scale_up_at = time.time()
        signals["current_replicas"] = float(target)

        # Phase 2: Still loaded, but cooldown prevents immediate scale-up
        action2, _, _ = autoscaler._decide("test", policy, state, signals)
        assert action2 == AutoscaleAction.HOLD

        # Phase 3: Load drops, time passes → scale down
        state.last_scale_up_at = time.time() - 60
        state.last_scale_down_at = 0.0
        signals.update({
            "current_replicas": 3.0,
            "queue_depth": 0.0,
            "oldest_queue_age_ms": 0.0,
            "avg_inflight_per_replica": 0.0,
            "avg_gpu_cache_usage": 0.0,
            "time_since_last_request_s": 30.0,
            "rps": 0.0,
        })
        action3, _, target3 = autoscaler._decide("test", policy, state, signals)
        assert action3 == AutoscaleAction.SCALE_DOWN
        assert target3 < 3


class TestEventCallbacks:
    """Test that registry events fire correctly."""

    def test_callbacks_fire_on_state_changes(self):
        reg = ClusterRegistry(_cluster_config())
        reg.load_models(_models_config())
        reg.set_node_status("worker1", NodeStatus.ONLINE)

        events: list[tuple[str, dict]] = []
        reg.register_event_callback(lambda t, d: events.append((t, d)))

        r = ReplicaState(replica_id="cb-test", model="llama",
                         node_name="worker1", gpu_ids=[0, 1], tp_size=2)
        reg.allocate_gpus("worker1", [0, 1], "cb-test")
        reg.add_replica(r)
        reg.set_replica_status("cb-test", ReplicaStatus.READY)
        reg.remove_replica("cb-test")

        event_types = [e[0] for e in events]
        assert "replica_added" in event_types
        assert "replica_status_change" in event_types
        assert "replica_removed" in event_types
