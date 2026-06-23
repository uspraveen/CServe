"""Tests for the GPU Memory Guard."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from cserve.common.models import (
    GpuInfo,
    GpuState,
    NodeState,
    NodeStatus,
    ReplicaState,
    ReplicaStatus,
    SafetyConfig,
)
from cserve.control_plane.gpu_guard import GpuMemoryGuard, GuardState


class FakeRegistry:
    def __init__(self) -> None:
        self.nodes: list[NodeState] = []
        self.replicas: dict[str, ReplicaState] = {}

    def get_online_nodes(self) -> list[NodeState]:
        return [n for n in self.nodes if n.status != NodeStatus.OFFLINE]

    def get_replica(self, replica_id: str | None) -> ReplicaState | None:
        if replica_id is None:
            return None
        return self.replicas.get(replica_id)


def _make_safety(**overrides) -> SafetyConfig:
    """Defaults match SafetyConfig; shorten guard timings for unit tests."""
    data = SafetyConfig().model_dump()
    data["guard_mitigation_window_s"] = 10.0
    data["guard_check_interval_s"] = 1.0
    data["guard_consecutive_breaches"] = 2
    data.update(overrides)
    return SafetyConfig(**data)


def _make_node(
    name: str = "node-1",
    gpu_utils: list[float] | None = None,
    gpu_util_pct: list[float] | None = None,
    total_mb: int = 46068,
    replica_id: str | None = None,
) -> NodeState:
    gpus = []
    utils = gpu_utils or [0.5]
    pct_list = gpu_util_pct or []
    for i, util in enumerate(utils):
        used = int(total_mb * util)
        u_pct = pct_list[i] if i < len(pct_list) else 0.0
        gpus.append(GpuInfo(
            index=i, name="A40",
            memory_used_mb=used, memory_total_mb=total_mb,
            utilization_pct=u_pct,
            state=GpuState.ALLOCATED if replica_id else GpuState.FREE,
            allocated_replica_id=replica_id,
        ))
    return NodeState(
        name=name, host=f"{name}.test",
        gpu_type="a40", status=NodeStatus.ONLINE,
        gpus=gpus,
    )


# ─── State transitions ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stays_ok_below_limit():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.50])]
    guard = GpuMemoryGuard(reg, _make_safety())

    await guard.check_all_gpus()
    entry = guard._entries.get(("node-1", 0))
    assert entry is not None
    assert entry.state == GuardState.OK


@pytest.mark.asyncio
async def test_transitions_to_warned_on_breach():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.95])]
    guard = GpuMemoryGuard(reg, _make_safety())

    await guard.check_all_gpus()
    entry = guard._entries[("node-1", 0)]
    assert entry.state == GuardState.WARNED


@pytest.mark.asyncio
async def test_transitions_to_mitigating_after_consecutive():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.95], replica_id="r1")]
    reg.replicas["r1"] = ReplicaState(
        replica_id="r1", model="test-model",
        node_name="node-1", gpu_ids=[0], tp_size=1,
        status=ReplicaStatus.READY,
    )

    paused = []
    guard = GpuMemoryGuard(
        reg, _make_safety(guard_consecutive_breaches=2),
        pause_callback=lambda rid: paused.append(rid),
    )

    # First check → WARNED
    await guard.check_all_gpus()
    assert guard._entries[("node-1", 0)].state == GuardState.WARNED

    # Second check → MITIGATING (consecutive_breaches=2)
    await guard.check_all_gpus()
    assert guard._entries[("node-1", 0)].state == GuardState.MITIGATING
    assert paused == ["r1"]


@pytest.mark.asyncio
async def test_self_heals_during_mitigation():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.95], replica_id="r1")]
    reg.replicas["r1"] = ReplicaState(
        replica_id="r1", model="test-model",
        node_name="node-1", gpu_ids=[0], tp_size=1,
        status=ReplicaStatus.READY,
    )

    resumed = []
    guard = GpuMemoryGuard(
        reg, _make_safety(guard_consecutive_breaches=2),
        pause_callback=lambda rid: None,
        resume_callback=lambda rid: resumed.append(rid),
    )

    # Two checks → MITIGATING
    await guard.check_all_gpus()
    await guard.check_all_gpus()
    assert guard._entries[("node-1", 0)].state == GuardState.MITIGATING

    # Memory drops → self-heal
    reg.nodes[0].gpus[0].memory_used_mb = int(46068 * 0.60)
    await guard.check_all_gpus()
    assert guard._entries[("node-1", 0)].state == GuardState.OK
    assert resumed == ["r1"]


@pytest.mark.asyncio
async def test_recovers_from_warned():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.95])]
    guard = GpuMemoryGuard(reg, _make_safety())

    await guard.check_all_gpus()
    assert guard._entries[("node-1", 0)].state == GuardState.WARNED

    reg.nodes[0].gpus[0].memory_used_mb = int(46068 * 0.50)
    await guard.check_all_gpus()
    assert guard._entries[("node-1", 0)].state == GuardState.OK


@pytest.mark.asyncio
async def test_migration_triggered_after_window():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.95], replica_id="r1")]
    reg.replicas["r1"] = ReplicaState(
        replica_id="r1", model="test-model",
        node_name="node-1", gpu_ids=[0], tp_size=1,
        status=ReplicaStatus.READY,
    )

    migrated = []

    async def mock_migrate(rid):
        migrated.append(rid)

    guard = GpuMemoryGuard(
        reg,
        _make_safety(guard_consecutive_breaches=2, guard_mitigation_window_s=0.1),
        pause_callback=lambda rid: None,
        migrate_callback=mock_migrate,
    )

    # Two checks → MITIGATING
    await guard.check_all_gpus()
    await guard.check_all_gpus()
    assert guard._entries[("node-1", 0)].state == GuardState.MITIGATING

    # Wait for mitigation window to expire
    await asyncio.sleep(0.15)

    # Next check → MIGRATING (triggers migration)
    await guard.check_all_gpus()
    assert guard._entries[("node-1", 0)].state == GuardState.MIGRATING

    # Wait for the background migration task
    await asyncio.sleep(0.1)
    assert migrated == ["r1"]


@pytest.mark.asyncio
async def test_multiple_gpus_independent():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.95, 0.50, 0.93])]
    guard = GpuMemoryGuard(reg, _make_safety())

    await guard.check_all_gpus()

    assert guard._entries[("node-1", 0)].state == GuardState.WARNED
    assert guard._entries[("node-1", 1)].state == GuardState.OK
    assert guard._entries[("node-1", 2)].state == GuardState.WARNED


@pytest.mark.asyncio
async def test_ignores_gpus_with_zero_total():
    reg = FakeRegistry()
    reg.nodes = [NodeState(
        name="empty", host="empty.test",
        gpu_type="a40", status=NodeStatus.ONLINE,
        gpus=[GpuInfo(index=0, memory_used_mb=0, memory_total_mb=0)],
    )]
    guard = GpuMemoryGuard(reg, _make_safety())
    await guard.check_all_gpus()
    assert len(guard._entries) == 0


@pytest.mark.asyncio
async def test_ignores_offline_nodes():
    reg = FakeRegistry()
    reg.nodes = [NodeState(
        name="off", host="off.test",
        gpu_type="a40", status=NodeStatus.OFFLINE,
        gpus=[GpuInfo(index=0, memory_used_mb=40000, memory_total_mb=46068)],
    )]
    guard = GpuMemoryGuard(reg, _make_safety())
    await guard.check_all_gpus()
    assert len(guard._entries) == 0


# ─── Event logging ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_events_recorded():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.95])]
    guard = GpuMemoryGuard(reg, _make_safety())

    await guard.check_all_gpus()
    events = guard.get_recent_events()
    assert len(events) == 1
    assert events[0]["action"] == "threshold_breach"
    assert events[0]["new_state"] == "WARNED"


@pytest.mark.asyncio
async def test_get_all_entries_only_non_ok():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.95, 0.50])]
    guard = GpuMemoryGuard(reg, _make_safety())

    await guard.check_all_gpus()
    entries = guard.get_all_entries()
    assert len(entries) == 1
    assert entries[0]["state"] == "WARNED"
    assert entries[0]["gpu_index"] == 0


@pytest.mark.asyncio
async def test_peak_utilization_tracked():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.95])]
    guard = GpuMemoryGuard(reg, _make_safety())

    await guard.check_all_gpus()
    entry = guard._entries[("node-1", 0)]

    reg.nodes[0].gpus[0].memory_used_mb = int(46068 * 0.50)
    await guard.check_all_gpus()
    assert entry.peak_utilization >= 0.94


@pytest.mark.asyncio
async def test_compute_sustained_breach_without_vram_breach():
    """High GPU-Util for sustain window triggers the same path as VRAM breach."""
    reg = FakeRegistry()
    reg.nodes = [
        _make_node(
            gpu_utils=[0.50],
            gpu_util_pct=[85.0],
            replica_id="r1",
        ),
    ]
    reg.replicas["r1"] = ReplicaState(
        replica_id="r1", model="test-model",
        node_name="node-1", gpu_ids=[0], tp_size=1,
        status=ReplicaStatus.READY,
    )
    t0 = 1_000_000.0
    clock = [t0]

    def fake_time() -> float:
        return clock[0]

    guard = GpuMemoryGuard(
        reg,
        _make_safety(
            gpu_compute_sustain_threshold=0.79,
            gpu_compute_sustain_duration_s=2.0,
            guard_consecutive_breaches=2,
        ),
        pause_callback=lambda _rid: None,
    )

    with patch("cserve.control_plane.gpu_guard.time.time", fake_time):
        await guard.check_all_gpus()
        assert guard._entries[("node-1", 0)].state == GuardState.OK

        clock[0] = t0 + 1.0
        await guard.check_all_gpus()
        assert guard._entries[("node-1", 0)].state == GuardState.OK

        clock[0] = t0 + 2.01
        await guard.check_all_gpus()
        assert guard._entries[("node-1", 0)].state == GuardState.WARNED

        clock[0] = t0 + 2.02
        await guard.check_all_gpus()
        assert guard._entries[("node-1", 0)].state == GuardState.MITIGATING


@pytest.mark.asyncio
async def test_compute_dip_resets_sustain_timer():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.50], gpu_util_pct=[85.0])]
    t0 = 2_000_000.0
    clock = [t0]

    def fake_time() -> float:
        return clock[0]

    guard = GpuMemoryGuard(
        reg,
        _make_safety(
            gpu_compute_sustain_threshold=0.79,
            gpu_compute_sustain_duration_s=2.0,
        ),
    )

    with patch("cserve.control_plane.gpu_guard.time.time", fake_time):
        await guard.check_all_gpus()
        entry = guard._entries[("node-1", 0)]
        assert entry.compute_sustain_started_at == t0
        assert entry.state == GuardState.OK

        clock[0] = t0 + 1.5
        reg.nodes[0].gpus[0].utilization_pct = 50.0
        await guard.check_all_gpus()
        assert entry.compute_sustain_started_at == 0.0
        assert entry.state == GuardState.OK


@pytest.mark.asyncio
async def test_get_compute_pressure_notifications_above_threshold():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.50], gpu_util_pct=[96.0])]
    guard = GpuMemoryGuard(reg, _make_safety())
    await guard.check_all_gpus()
    rows = guard.get_compute_pressure_notifications()
    assert len(rows) == 1
    assert rows[0]["node_name"] == "node-1"
    assert rows[0]["gpu_index"] == 0
    assert rows[0]["compute_util_frac"] >= 0.95
    assert rows[0]["guard_state"] == "OK"
    assert rows[0]["sustain_duration_s"] == 900.0


@pytest.mark.asyncio
async def test_get_compute_pressure_notifications_empty_below_threshold():
    reg = FakeRegistry()
    reg.nodes = [_make_node(gpu_utils=[0.50], gpu_util_pct=[50.0])]
    guard = GpuMemoryGuard(reg, _make_safety())
    await guard.check_all_gpus()
    assert guard.get_compute_pressure_notifications() == []
