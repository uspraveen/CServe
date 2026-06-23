"""Tests for core data models."""

import time

from cserve.common.models import (
    AutoscalePolicy,
    GpuInfo,
    GpuState,
    Job,
    JobEvent,
    ModelConfig,
    NodeState,
    NodeStatus,
    ReplicaState,
    ReplicaStatus,
    RoutingStrategy,
)


class TestReplicaStateMachine:
    def test_starting_can_become_ready(self):
        r = ReplicaState(model="test", node_name="n1", gpu_ids=[0], tp_size=1)
        assert r.status == ReplicaStatus.STARTING
        assert r.can_transition_to(ReplicaStatus.READY)

    def test_starting_can_fail(self):
        r = ReplicaState(model="test", node_name="n1", gpu_ids=[0], tp_size=1)
        assert r.can_transition_to(ReplicaStatus.FAILED)

    def test_starting_cannot_drain(self):
        r = ReplicaState(model="test", node_name="n1", gpu_ids=[0], tp_size=1)
        assert not r.can_transition_to(ReplicaStatus.DRAINING)

    def test_ready_can_drain(self):
        r = ReplicaState(
            model="test", node_name="n1", gpu_ids=[0], tp_size=1,
            status=ReplicaStatus.READY,
        )
        assert r.can_transition_to(ReplicaStatus.DRAINING)

    def test_ready_cannot_stop_directly(self):
        r = ReplicaState(
            model="test", node_name="n1", gpu_ids=[0], tp_size=1,
            status=ReplicaStatus.READY,
        )
        assert not r.can_transition_to(ReplicaStatus.STOPPING)

    def test_draining_can_stop(self):
        r = ReplicaState(
            model="test", node_name="n1", gpu_ids=[0], tp_size=1,
            status=ReplicaStatus.DRAINING,
        )
        assert r.can_transition_to(ReplicaStatus.STOPPING)

    def test_failed_is_terminal(self):
        r = ReplicaState(
            model="test", node_name="n1", gpu_ids=[0], tp_size=1,
            status=ReplicaStatus.FAILED,
        )
        assert r.status.is_terminal()
        for s in ReplicaStatus:
            assert not r.can_transition_to(s)


class TestJob:
    def test_job_defaults(self):
        j = Job(model="gemma3-27b")
        assert j.priority == 50
        assert j.variant == "default"
        assert len(j.job_id) == 32
        assert not j.streaming

    def test_job_expired(self):
        j = Job(model="test", enqueued_at=time.time() - 100, deadline_ms=1000)
        assert j.is_expired()

    def test_job_not_expired(self):
        j = Job(model="test", deadline_ms=30_000)
        assert not j.is_expired()


class TestJobEvent:
    def test_terminal_events(self):
        assert JobEvent.COMPLETED.is_terminal()
        assert JobEvent.FAILED.is_terminal()
        assert JobEvent.TIMEOUT.is_terminal()
        assert JobEvent.CANCELLED.is_terminal()
        assert not JobEvent.ENQUEUED.is_terminal()
        assert not JobEvent.SCHEDULED.is_terminal()
        assert not JobEvent.STARTED.is_terminal()


class TestGpuState:
    def test_gpu_defaults(self):
        g = GpuInfo(index=0)
        assert g.state == GpuState.FREE
        assert g.allocated_replica_id is None


class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig(
            name="test", served_model_name="test", hf_model="org/model",
        )
        assert m.tp == 1
        assert m.routing_strategy == RoutingStrategy.LEAST_OUTSTANDING
        assert m.autoscaling.min_replicas == 1

    def test_autoscale_policy_defaults(self):
        p = AutoscalePolicy()
        assert p.min_replicas == 1
        assert p.max_replicas == 1
        assert not p.allow_scale_to_zero
        assert p.target_inflight == 3.0


class TestNodeState:
    def test_defaults(self):
        n = NodeState(name="n1", host="h1")
        assert n.status == NodeStatus.OFFLINE
        assert n.consecutive_failures == 0
