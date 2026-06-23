"""GPU Memory Guard — enforces per-GPU memory limits with intelligent mitigation.

State machine per (node, gpu_index) pair:

    OK  ─[breach]→  WARNED  ─[confirmed]→  MITIGATING  ─[timeout]→  MIGRATING
    ↑                  ↓                        ↓                        ↓
    └───[recovered]────┘        [recovered]─────┘        [migration done]┘

A *breach* is either:
  - VRAM used/total ≥ gpu_memory_limit (for guard_consecutive_breaches samples), or
  - GPU compute utilization (nvidia-smi GPU-Util %) ≥ gpu_compute_sustain_threshold
    continuously for gpu_compute_sustain_duration_s (wall clock; default 15 min so
    model load does not count as sustained production overload).

MITIGATING phase (up to guard_mitigation_window_s, default 10 min):
  - Tell vLLM to reduce KV-cache pressure by pausing new request admission
    (set replica to DRAINING temporarily so the gateway stops routing to it).
  - If memory drops below the limit within the window, cancel the migration
    and resume the replica (set it back to READY).

MIGRATING phase:
  - Drain in-flight requests (gateway already stopped routing).
  - Kill the replica on the hot GPU.
  - Find a new placement on a different node/GPU set.
  - Launch a replacement replica.
  - The user never loses service — other replicas of the same model keep
    serving while the migration is in progress.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum

from cserve.common.logging import get_logger
from cserve.common.metrics import HEALTH_INCIDENTS
from cserve.common.models import SafetyConfig

log = get_logger("gpu_guard")


class GuardState(StrEnum):
    OK = "OK"
    WARNED = "WARNED"
    MITIGATING = "MITIGATING"
    MIGRATING = "MIGRATING"


@dataclass
class GpuGuardEntry:
    """Per-GPU tracking state."""
    node_name: str
    gpu_index: int
    replica_id: str | None = None
    state: GuardState = GuardState.OK
    consecutive_breaches: int = 0
    first_breach_at: float = 0.0
    mitigation_started_at: float = 0.0
    peak_utilization: float = 0.0
    last_utilization: float = 0.0
    migration_requested: bool = False
    # Wall-clock sustained compute (GPU-Util %) above threshold
    compute_sustain_started_at: float = 0.0
    last_compute_utilization: float = 0.0


@dataclass
class GuardEvent:
    """Emitted for dashboard and logging."""
    timestamp: float = field(default_factory=time.time)
    node_name: str = ""
    gpu_index: int = 0
    replica_id: str = ""
    model: str = ""
    old_state: str = ""
    new_state: str = ""
    utilization: float = 0.0
    action: str = ""
    details: str = ""


class GpuMemoryGuard:
    """Monitors GPU memory and orchestrates intelligent mitigation + migration."""

    def __init__(
        self,
        registry,
        safety: SafetyConfig,
        migrate_callback=None,
        pause_callback=None,
        resume_callback=None,
    ) -> None:
        self.registry = registry
        self.safety = safety
        self._entries: dict[tuple[str, int], GpuGuardEntry] = {}
        self._recent_events: list[GuardEvent] = []
        self._max_events = 100

        # Callbacks provided by the control plane server
        self._migrate_callback = migrate_callback
        self._pause_callback = pause_callback
        self._resume_callback = resume_callback

    def _replica_guard_exempt(self, replica_id: str) -> bool:
        rep = self.registry.get_replica(replica_id)
        if not rep:
            return False
        cfg = self.registry.get_model_config(rep.model)
        return bool(cfg and cfg.gpu_guard_exempt)

    def reset_per_gpu_tracking_after_cluster_stop(self) -> None:
        """Drop all in-memory guard entries (after full cluster stop).

        The next ``check_all_gpus`` pass rebuilds from fresh nvidia-smi data.
        Without this, **compute sustain** timers and cached GPU-util samples from
        before the stop keep driving dashboard "GPU compute pressure" cards even
        after replicas were torn down and GPUs swept.
        """
        self._entries.clear()

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        return [
            {
                "timestamp": e.timestamp,
                "node_name": e.node_name,
                "gpu_index": e.gpu_index,
                "replica_id": e.replica_id,
                "model": e.model,
                "old_state": e.old_state,
                "new_state": e.new_state,
                "utilization": e.utilization,
                "action": e.action,
                "details": e.details,
            }
            for e in self._recent_events[-limit:]
        ]

    def get_all_entries(self) -> list[dict]:
        return [
            {
                "node_name": e.node_name,
                "gpu_index": e.gpu_index,
                "replica_id": e.replica_id or "",
                "state": e.state.value,
                "consecutive_breaches": e.consecutive_breaches,
                "last_utilization": round(e.last_utilization, 4),
                "peak_utilization": round(e.peak_utilization, 4),
                "mitigation_started_at": e.mitigation_started_at,
                "last_compute_utilization": round(e.last_compute_utilization, 4),
                "compute_sustain_started_at": e.compute_sustain_started_at,
            }
            for e in self._entries.values()
            if e.state != GuardState.OK
        ]

    def get_compute_pressure_notifications(self) -> list[dict]:
        """GPUs at or above the compute sustain threshold (nvidia-smi util).

        Includes OK-state GPUs that are accumulating sustained high utilization,
        so the dashboard can show alerts and a timer before the guard trips.
        """
        now = time.time()
        thresh = self.safety.gpu_compute_sustain_threshold
        sustain_s = self.safety.gpu_compute_sustain_duration_s

        online_keys: set[tuple[str, int]] = set()
        for node in self.registry.get_online_nodes():
            for gpu in node.gpus:
                if gpu.memory_total_mb == 0:
                    continue
                online_keys.add((node.name, gpu.index))

        rows: list[dict] = []
        for key in online_keys:
            entry = self._entries.get(key)
            if entry is None:
                continue
            if entry.last_compute_utilization < thresh:
                continue
            started = entry.compute_sustain_started_at
            elapsed = (now - started) if started > 0.0 else 0.0
            progress = min(1.0, elapsed / sustain_s) if sustain_s > 0 else 0.0
            rows.append({
                "node_name": entry.node_name,
                "gpu_index": entry.gpu_index,
                "replica_id": entry.replica_id or "",
                "guard_state": entry.state.value,
                "compute_util_frac": round(entry.last_compute_utilization, 4),
                "threshold_frac": round(thresh, 4),
                "sustain_duration_s": sustain_s,
                "compute_sustain_started_at": started,
                "sustain_elapsed_s": round(elapsed, 2),
                "sustain_progress": round(progress, 4),
            })
        rows.sort(
            key=lambda r: (-r["sustain_progress"], r["node_name"], r["gpu_index"]),
        )
        return rows

    async def check_all_gpus(self) -> None:
        """Called periodically by the health manager."""
        mem_limit = self.safety.gpu_memory_limit
        compute_thresh = self.safety.gpu_compute_sustain_threshold
        sustain_s = self.safety.gpu_compute_sustain_duration_s
        now = time.time()
        nodes = self.registry.get_online_nodes()

        for node in nodes:
            for gpu in node.gpus:
                if gpu.memory_total_mb == 0:
                    continue
                mem_frac = gpu.memory_used_mb / gpu.memory_total_mb
                comp_frac = max(0.0, min(1.0, gpu.utilization_pct / 100.0))
                key = (node.name, gpu.index)
                entry = self._entries.get(key)

                if entry is None:
                    entry = GpuGuardEntry(
                        node_name=node.name,
                        gpu_index=gpu.index,
                        replica_id=gpu.allocated_replica_id,
                    )
                    self._entries[key] = entry

                entry.last_utilization = mem_frac
                entry.peak_utilization = max(entry.peak_utilization, mem_frac)
                entry.last_compute_utilization = comp_frac
                entry.replica_id = gpu.allocated_replica_id

                if entry.replica_id and self._replica_guard_exempt(entry.replica_id):
                    entry.consecutive_breaches = 0
                    entry.first_breach_at = 0.0
                    entry.mitigation_started_at = 0.0
                    if entry.state != GuardState.OK:
                        await self._transition(
                            entry, GuardState.OK,
                            action="guard_exempt",
                            details="model has gpu_guard_exempt",
                        )
                    continue

                if comp_frac >= compute_thresh:
                    if entry.compute_sustain_started_at == 0.0:
                        entry.compute_sustain_started_at = now
                else:
                    entry.compute_sustain_started_at = 0.0

                compute_sustained = (
                    entry.compute_sustain_started_at > 0.0
                    and (now - entry.compute_sustain_started_at) >= sustain_s
                )

                await self._evaluate(
                    entry, mem_frac, mem_limit, compute_sustained, comp_frac,
                )

    async def _evaluate(
        self,
        entry: GpuGuardEntry,
        mem_frac: float,
        mem_limit: float,
        compute_sustained: bool,
        comp_frac: float,
    ) -> None:
        now = time.time()

        mem_breach = mem_frac >= mem_limit
        effective_breach = mem_breach or compute_sustained

        if entry.state == GuardState.MIGRATING:
            return

        if not effective_breach:
            if entry.state == GuardState.MITIGATING:
                await self._transition(entry, GuardState.OK,
                                       action="self_healed",
                                       details=(
                                           f"Pressure cleared (VRAM {mem_frac:.0%}, "
                                           f"GPU util {comp_frac:.0%})"
                                       ))
                if entry.replica_id and self._resume_callback:
                    try:
                        await self._resume_callback(entry.replica_id)
                    except Exception as e:
                        log.warning("resume after self-heal failed",
                                    replica=entry.replica_id, error=str(e))

            elif entry.state != GuardState.OK:
                await self._transition(entry, GuardState.OK,
                                       action="recovered",
                                       details=f"VRAM {mem_frac:.0%}, GPU util {comp_frac:.0%}")
            entry.consecutive_breaches = 0
            entry.first_breach_at = 0.0
            entry.mitigation_started_at = 0.0
            return

        entry.consecutive_breaches += 1

        if entry.first_breach_at == 0.0:
            entry.first_breach_at = now

        breach_detail = (
            f"GPU compute ≥{self.safety.gpu_compute_sustain_threshold:.0%} "
            f"for ≥{self.safety.gpu_compute_sustain_duration_s:.0f}s "
            f"(now {comp_frac:.0%})"
            if compute_sustained and not mem_breach
            else f"GPU memory at {mem_frac:.0%} (limit {mem_limit:.0%})"
        )

        if entry.state == GuardState.OK:
            await self._transition(entry, GuardState.WARNED,
                                   action="threshold_breach",
                                   details=breach_detail)

        elif entry.state == GuardState.WARNED:
            if entry.consecutive_breaches >= self.safety.guard_consecutive_breaches:
                await self._transition(entry, GuardState.MITIGATING,
                                       action="mitigation_start",
                                       details=f"Confirmed breach ({entry.consecutive_breaches}x), "
                                               f"pausing replica to reduce pressure — {breach_detail}")
                entry.mitigation_started_at = now
                if entry.replica_id and self._pause_callback:
                    try:
                        await self._pause_callback(entry.replica_id)
                    except Exception as e:
                        log.error("pause for mitigation failed",
                                  replica=entry.replica_id, error=str(e))

        elif entry.state == GuardState.MITIGATING:
            elapsed = now - entry.mitigation_started_at
            if elapsed >= self.safety.guard_mitigation_window_s:
                await self._transition(entry, GuardState.MIGRATING,
                                       action="migration_trigger",
                                       details=(
                                           f"Pressure still present after "
                                           f"{elapsed:.0f}s mitigation "
                                           f"(limit {self.safety.guard_mitigation_window_s:.0f}s). "
                                           f"VRAM {mem_frac:.0%}, GPU util {comp_frac:.0%}. "
                                           f"Initiating live migration."
                                       ))
                entry.migration_requested = True
                if entry.replica_id and self._migrate_callback:
                    asyncio.create_task(
                        self._do_migration(entry)
                    )

    async def _do_migration(self, entry: GpuGuardEntry) -> None:
        """Execute the migration and transition back to OK on completion."""
        try:
            if self._migrate_callback and entry.replica_id:
                await self._migrate_callback(entry.replica_id)
            await self._transition(entry, GuardState.OK,
                                   action="migration_complete",
                                   details="Replica migrated to new GPU(s)")
        except Exception as e:
            log.error("migration failed", node=entry.node_name,
                      gpu=entry.gpu_index, replica=entry.replica_id,
                      error=str(e))
            self._record_event(GuardEvent(
                node_name=entry.node_name,
                gpu_index=entry.gpu_index,
                replica_id=entry.replica_id or "",
                old_state=entry.state.value,
                new_state=entry.state.value,
                utilization=entry.last_utilization,
                action="migration_failed",
                details=str(e),
            ))
        finally:
            entry.migration_requested = False
            entry.consecutive_breaches = 0
            entry.first_breach_at = 0.0
            entry.mitigation_started_at = 0.0
            entry.compute_sustain_started_at = 0.0

    async def _transition(
        self, entry: GpuGuardEntry, new_state: GuardState,
        action: str = "", details: str = "",
    ) -> None:
        old = entry.state
        entry.state = new_state

        replica = self.registry.get_replica(entry.replica_id) if entry.replica_id else None
        model = replica.model if replica else ""

        event = GuardEvent(
            node_name=entry.node_name,
            gpu_index=entry.gpu_index,
            replica_id=entry.replica_id or "",
            model=model,
            old_state=old.value,
            new_state=new_state.value,
            utilization=entry.last_utilization,
            action=action,
            details=details,
        )
        self._record_event(event)

        log.info("gpu guard state change",
                 node=entry.node_name, gpu=entry.gpu_index,
                 replica=entry.replica_id, model=model,
                 old_state=old.value, new_state=new_state.value,
                 utilization=f"{entry.last_utilization:.0%}",
                 action=action)

        HEALTH_INCIDENTS.labels(incident_type=f"gpu_guard_{action}").inc()

    def _record_event(self, event: GuardEvent) -> None:
        self._recent_events.append(event)
        if len(self._recent_events) > self._max_events:
            self._recent_events = self._recent_events[-self._max_events:]
