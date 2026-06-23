"""Tests for the Cluster Registry."""

import pytest

from cserve.common.models import (
    ClusterConfig,
    GatewayConfig,
    GpuState,
    HeadConfig,
    NodeAgentConfig,
    NodeConfig,
    NodeStatus,
    RedisConfig,
    ReplicaState,
    ReplicaStatus,
    SafetyConfig,
)
from cserve.control_plane.registry import (
    ClusterRegistry,
    InvalidTransitionError,
    RegistryError,
)


def _make_cluster_config() -> ClusterConfig:
    return ClusterConfig(
        head=HeadConfig(name="head", host="10.0.0.1", node_ip="10.0.0.1"),
        nodes=[
            NodeConfig(
                name="w1", host="10.0.0.2", gpu_count=4,
                gpu_type="a40", cuda_devices="0,1,2,3",
            ),
            NodeConfig(
                name="w2", host="10.0.0.3", gpu_count=4,
                gpu_type="l40", cuda_devices="0,1,2,3",
            ),
        ],
        gateway=GatewayConfig(),
        redis=RedisConfig(),
        safety=SafetyConfig(),
        node_agent=NodeAgentConfig(),
    )


class TestNodeManagement:
    def test_nodes_initialized_offline(self):
        reg = ClusterRegistry(_make_cluster_config())
        nodes = reg.get_all_nodes()
        assert len(nodes) == 2
        assert all(n.status == NodeStatus.OFFLINE for n in nodes)

    def test_node_success_brings_online(self):
        reg = ClusterRegistry(_make_cluster_config())
        reg.record_node_success("w1")
        assert reg.get_node("w1").status == NodeStatus.ONLINE

    def test_heartbeat_does_not_bring_online(self):
        reg = ClusterRegistry(_make_cluster_config())
        reg.record_heartbeat("w1")
        # Heartbeats update last_heartbeat but do NOT bring nodes online
        # (only outbound health-check success does)
        assert reg.get_node("w1").status == NodeStatus.OFFLINE

    def test_set_node_status(self):
        reg = ClusterRegistry(_make_cluster_config())
        reg.set_node_status("w1", NodeStatus.DEGRADED)
        assert reg.get_node("w1").status == NodeStatus.DEGRADED

    def test_unknown_node_raises(self):
        reg = ClusterRegistry(_make_cluster_config())
        with pytest.raises(RegistryError, match="Unknown node"):
            reg.set_node_status("nonexistent", NodeStatus.ONLINE)


class TestGpuAllocation:
    def test_gpus_initialized_free(self):
        reg = ClusterRegistry(_make_cluster_config())
        free = reg.get_free_gpus("w1")
        assert len(free) == 4
        assert all(g.state == GpuState.FREE for g in free)

    def test_allocate_and_release(self):
        reg = ClusterRegistry(_make_cluster_config())
        reg.allocate_gpus("w1", [0, 1], "r1")
        free = reg.get_free_gpus("w1")
        assert len(free) == 2
        assert set(g.index for g in free) == {2, 3}

        reg.release_gpus("w1", [0, 1])
        free = reg.get_free_gpus("w1")
        assert len(free) == 4

    def test_double_allocate_raises(self):
        reg = ClusterRegistry(_make_cluster_config())
        reg.allocate_gpus("w1", [0], "r1")
        with pytest.raises(RegistryError, match="cannot allocate"):
            reg.allocate_gpus("w1", [0], "r2")


class TestReplicaLifecycle:
    def test_add_and_get_replica(self):
        reg = ClusterRegistry(_make_cluster_config())
        r = ReplicaState(
            replica_id="r1", model="gemma", node_name="w1",
            gpu_ids=[0, 1], tp_size=2,
        )
        reg.add_replica(r)
        assert reg.get_replica("r1") is not None
        assert reg.count_replicas("gemma") == 1

    def test_valid_state_transition(self):
        reg = ClusterRegistry(_make_cluster_config())
        r = ReplicaState(
            replica_id="r1", model="gemma", node_name="w1",
            gpu_ids=[0, 1], tp_size=2,
        )
        reg.add_replica(r)
        reg.set_replica_status("r1", ReplicaStatus.READY)
        assert reg.get_replica("r1").status == ReplicaStatus.READY

    def test_invalid_transition_raises(self):
        reg = ClusterRegistry(_make_cluster_config())
        r = ReplicaState(
            replica_id="r1", model="gemma", node_name="w1",
            gpu_ids=[0, 1], tp_size=2,
        )
        reg.add_replica(r)
        with pytest.raises(InvalidTransitionError):
            reg.set_replica_status("r1", ReplicaStatus.DRAINING)

    def test_full_lifecycle(self):
        reg = ClusterRegistry(_make_cluster_config())
        reg.allocate_gpus("w1", [0, 1], "r1")
        r = ReplicaState(
            replica_id="r1", model="gemma", node_name="w1",
            gpu_ids=[0, 1], tp_size=2,
        )
        reg.add_replica(r)

        # STARTING → READY
        reg.set_replica_status("r1", ReplicaStatus.READY)
        assert reg.count_ready_replicas("gemma") == 1

        # READY → DRAINING
        reg.set_replica_status("r1", ReplicaStatus.DRAINING)
        assert reg.count_ready_replicas("gemma") == 0

        # DRAINING → STOPPING
        reg.set_replica_status("r1", ReplicaStatus.STOPPING)

        # Remove
        removed = reg.remove_replica("r1")
        assert removed is not None
        assert reg.count_replicas("gemma") == 0
        # GPUs should be freed
        assert len(reg.get_free_gpus("w1")) == 4

    def test_inflight_tracking(self):
        reg = ClusterRegistry(_make_cluster_config())
        r = ReplicaState(
            replica_id="r1", model="gemma", node_name="w1",
            gpu_ids=[0], tp_size=1,
        )
        reg.add_replica(r)
        reg.increment_inflight("r1")
        reg.increment_inflight("r1")
        assert reg.get_replica("r1").inflight_requests == 2
        reg.decrement_inflight("r1")
        assert reg.get_replica("r1").inflight_requests == 1

    def test_duplicate_replica_raises(self):
        reg = ClusterRegistry(_make_cluster_config())
        r = ReplicaState(
            replica_id="r1", model="gemma", node_name="w1",
            gpu_ids=[0], tp_size=1,
        )
        reg.add_replica(r)
        with pytest.raises(RegistryError, match="already exists"):
            reg.add_replica(r)

    def test_get_healthy_replicas(self):
        reg = ClusterRegistry(_make_cluster_config())
        r1 = ReplicaState(
            replica_id="r1", model="gemma", node_name="w1",
            gpu_ids=[0], tp_size=1,
        )
        r2 = ReplicaState(
            replica_id="r2", model="gemma", node_name="w1",
            gpu_ids=[1], tp_size=1, status=ReplicaStatus.READY,
        )
        reg.add_replica(r1)
        reg.add_replica(r2)
        healthy = reg.get_healthy_replicas("gemma")
        assert len(healthy) == 1
        assert healthy[0].replica_id == "r2"


class TestAggregation:
    def test_total_gpus_by_type(self):
        reg = ClusterRegistry(_make_cluster_config())
        reg.set_node_status("w1", NodeStatus.ONLINE)
        reg.set_node_status("w2", NodeStatus.ONLINE)
        summary = reg.total_gpus_by_type()
        assert summary["a40"] == (4, 4)
        assert summary["l40"] == (4, 4)

    def test_snapshot(self):
        reg = ClusterRegistry(_make_cluster_config())
        snap = reg.snapshot()
        assert "nodes" in snap
        assert "replicas" in snap
        assert len(snap["nodes"]) == 2
