"""Tests for the scheduler's replica selection strategies."""

from __future__ import annotations

from cserve.common.models import Job, ReplicaState, RoutingStrategy
from cserve.control_plane.scheduler import (
    _select_lor,
    _select_prefix_aware,
    _select_session_affinity,
    _select_weighted_rr,
    select_replica,
)


def _make_replica(replica_id: str, model: str = "test-model", inflight: int = 0) -> ReplicaState:
    return ReplicaState(
        replica_id=replica_id,
        model=model,
        node_name="node1",
        gpu_ids=[0],
        tp_size=1,
        inflight_requests=inflight,
    )


def _make_job(model: str = "test-model", tenant: str = "", payload: bytes = b"") -> Job:
    return Job(model=model, tenant_id=tenant, payload=payload)


class TestLeastOutstanding:
    def test_picks_lowest_inflight(self):
        replicas = [
            _make_replica("r1", inflight=5),
            _make_replica("r2", inflight=1),
            _make_replica("r3", inflight=3),
        ]
        result = _select_lor(replicas)
        assert result.replica_id == "r2"

    def test_single_replica(self):
        replicas = [_make_replica("r1", inflight=10)]
        result = _select_lor(replicas)
        assert result.replica_id == "r1"

    def test_all_equal_returns_first(self):
        replicas = [
            _make_replica("r1", inflight=2),
            _make_replica("r2", inflight=2),
        ]
        result = _select_lor(replicas)
        assert result.replica_id == "r1"


class TestPrefixAware:
    def test_empty_payload_falls_back_to_lor(self):
        replicas = [
            _make_replica("r1", inflight=5),
            _make_replica("r2", inflight=1),
        ]
        job = _make_job(payload=b"")
        result = _select_prefix_aware(replicas, job)
        assert result.replica_id == "r2"

    def test_imbalanced_load_falls_back_to_lor(self):
        replicas = [
            _make_replica("r1", inflight=10),
            _make_replica("r2", inflight=1),
        ]
        job = _make_job(payload=b"hello world prompt text")
        result = _select_prefix_aware(replicas, job)
        assert result.replica_id == "r2"

    def test_consistent_hashing(self):
        replicas = [
            _make_replica("r1", inflight=0),
            _make_replica("r2", inflight=0),
            _make_replica("r3", inflight=0),
        ]
        job = _make_job(payload=b"the same prompt")
        result1 = _select_prefix_aware(replicas, job)
        result2 = _select_prefix_aware(replicas, job)
        assert result1.replica_id == result2.replica_id


class TestSessionAffinity:
    def test_no_tenant_falls_back_to_lor(self):
        replicas = [
            _make_replica("r1", inflight=5),
            _make_replica("r2", inflight=1),
        ]
        job = _make_job(tenant="")
        result = _select_session_affinity(replicas, job)
        assert result.replica_id == "r2"

    def test_same_tenant_same_replica(self):
        replicas = [
            _make_replica("r1"),
            _make_replica("r2"),
            _make_replica("r3"),
        ]
        job = _make_job(tenant="user-42")
        result1 = _select_session_affinity(replicas, job)
        result2 = _select_session_affinity(replicas, job)
        assert result1.replica_id == result2.replica_id

    def test_different_tenants_may_differ(self):
        replicas = [_make_replica(f"r{i}") for i in range(10)]
        job1 = _make_job(tenant="user-a")
        job2 = _make_job(tenant="user-b")
        r1 = _select_session_affinity(replicas, job1)
        r2 = _select_session_affinity(replicas, job2)
        # Not guaranteed to differ, but with 10 replicas, very likely
        assert r1 is not None
        assert r2 is not None


class TestWeightedRoundRobin:
    def test_cycles_through_replicas(self):
        replicas = [
            _make_replica("r1"),
            _make_replica("r2"),
            _make_replica("r3"),
        ]
        seen = set()
        for _ in range(3):
            r = _select_weighted_rr(replicas)
            seen.add(r.replica_id)
        assert len(seen) == 3


class TestSelectReplica:
    def test_lor_strategy(self):
        replicas = [
            _make_replica("r1", inflight=5),
            _make_replica("r2", inflight=1),
        ]
        job = _make_job()
        result = select_replica(replicas, job, RoutingStrategy.LEAST_OUTSTANDING)
        assert result.replica_id == "r2"

    def test_empty_replicas_returns_none(self):
        result = select_replica([], _make_job(), RoutingStrategy.LEAST_OUTSTANDING)
        assert result is None
