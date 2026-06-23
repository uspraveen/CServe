"""Tests for the dashboard API (snapshot building)."""

from __future__ import annotations

import pytest

from cserve.common.models import (
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
from cserve.control_plane.registry import ClusterRegistry
from cserve.dashboard.api import DashboardAPI


class FakeDB:
    async def get_recent_job_events(self, **kwargs):
        return []

    async def get_job_events_for_replica(self, *args, **kwargs):
        return []

    async def get_health_incidents_for_replicas(self, *args, **kwargs):
        return []

    async def get_autoscale_events(self, **kwargs):
        return []

    async def get_recent_health_incidents(self, **kwargs):
        return []


class FakeQueue:
    async def queue_depths_all(self):
        return {"test-model": 3}


def _make_setup():
    cfg = ClusterConfig(
        head=HeadConfig(name="head", host="10.0.0.1"),
        nodes=[NodeConfig(name="w1", host="10.0.0.2", gpu_count=2,
                          gpu_type="a40", cuda_devices="0,1")],
        gateway=GatewayConfig(), redis=RedisConfig(),
        safety=SafetyConfig(), node_agent=NodeAgentConfig(),
    )
    reg = ClusterRegistry(cfg)
    models = {"test-model": ModelConfig(
        name="test-model", served_model_name="test-model",
        hf_model="org/test", tp=1,
    )}
    reg.load_models(models)
    reg.set_node_status("w1", NodeStatus.ONLINE)

    r = ReplicaState(replica_id="r1", model="test-model",
                     node_name="w1", gpu_ids=[0], tp_size=1)
    reg.allocate_gpus("w1", [0], "r1")
    reg.add_replica(r)
    reg.set_replica_status("r1", ReplicaStatus.READY)

    dash = DashboardAPI(reg, FakeDB(), FakeQueue(), models_config=models)
    return dash


class TestDashboardSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_structure(self):
        dash = _make_setup()
        snap = await dash._build_snapshot()

        assert "timestamp" in snap
        assert "nodes" in snap
        assert "replicas" in snap
        assert "models" in snap
        assert "stats" in snap
        assert "queue_depths" in snap

    @pytest.mark.asyncio
    async def test_snapshot_stats(self):
        dash = _make_setup()
        snap = await dash._build_snapshot()
        stats = snap["stats"]

        assert stats["total_gpus"] == 2
        assert stats["free_gpus"] == 1
        assert stats["total_replicas"] == 1
        assert stats["ready_replicas"] == 1
        assert stats["total_queue_depth"] == 3

    @pytest.mark.asyncio
    async def test_snapshot_contains_nodes(self):
        dash = _make_setup()
        snap = await dash._build_snapshot()

        assert len(snap["nodes"]) == 1
        assert snap["nodes"][0]["name"] == "w1"
        assert snap["nodes"][0]["status"] == "ONLINE"

    @pytest.mark.asyncio
    async def test_snapshot_contains_replicas(self):
        dash = _make_setup()
        snap = await dash._build_snapshot()

        assert len(snap["replicas"]) == 1
        assert snap["replicas"][0]["replica_id"] == "r1"
        assert snap["replicas"][0]["status"] == "READY"

    @pytest.mark.asyncio
    async def test_snapshot_contains_model_capabilities(self):
        dash = _make_setup()
        snap = await dash._build_snapshot()

        assert snap["model_configs"]["test-model"]["capabilities"] == ["chat"]
