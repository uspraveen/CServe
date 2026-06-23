"""Health Manager — monitors nodes, replicas, and GPU safety.

Three layers of health checks, each on its own interval:
  Layer 1: Node health (every 15s) — can we reach the node agent?
  Layer 2: Replica health (every 10s) — is the vLLM /health endpoint OK?
  Layer 3: GPU Memory Guard (configurable, default 20s) — enforces the
           per-GPU memory limit with an intelligent mitigation → migration
           pipeline.  See gpu_guard.py for the full state machine.

Auto-remediation (Layer 2 — intelligent error handling):
  When a replica fails, the health manager classifies the error and picks
  the cheapest remediation strategy before escalating:

    1. RESTART_IN_PLACE  (OOM, NCCL timeout, segfault, generic crash)
       Kill the vLLM process and relaunch on the SAME GPUs.  Fastest
       recovery — no placement search, no model re-download.  Retried
       up to MAX_IN_PLACE_RETRIES times.

    2. MIGRATE           (repeated in-place failures, import errors,
                          config errors that won't self-heal)
       Release GPUs, find new placement on a different node, relaunch
       there.  The autoscaler's FAILED-replica cleanup also covers this.

    3. GIVE_UP           (max total retries exhausted)
       Mark replica FAILED and let the autoscaler decide whether to
       launch a brand-new replica elsewhere.

  Error classification uses the last 30 lines of vLLM stdout/stderr
  captured by the launcher.  Known patterns:

    OOM           →  "CUDA out of memory", "Free memory on device"
    NCCL          →  "NCCL error", "NCCL watchdog"
    IMPORT        →  "ImportError", "ModuleNotFoundError"
    CONFIG        →  "ValueError:", "invalid argument"
    SEGFAULT      →  exit code -11 or "Segmentation fault"
    UNKNOWN       →  anything else

Other layers:
  - Unreachable node (3 consecutive failures) → mark offline, drain replicas.
  - GPU memory > limit → WARN → MITIGATE (pause) → MIGRATE (live migration).
  - Zombie GPU processes → kill via node agent.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum

import httpx

from cserve.common.logging import get_logger
from cserve.common.metrics import HEALTH_CHECK_DURATION, HEALTH_INCIDENTS
from cserve.common.models import ModelConfig, NodeStatus, ReplicaStatus, SafetyConfig
from cserve.control_plane.gpu_guard import GpuMemoryGuard

log = get_logger("health")

# Node health check runs every 10 s (was 15 s) so an unreachable node is
# detected in ≤ 20 s (2 failures × 10 s) instead of the previous 45 s.
# The launch circuit breaker provides even faster protection — it kicks in
# after the very first ConnectTimeout without waiting for the health cycle.
NODE_CHECK_INTERVAL_S = 10.0
REPLICA_CHECK_INTERVAL_S = 10.0
MAX_CONSECUTIVE_NODE_FAILURES = 2
MAX_CONSECUTIVE_REPLICA_FAILURES = 6
# Transient agent/network blips should not kill a loaded 30B+ model.
MAX_CONSECUTIVE_REPLICA_FAILURES_AGENT_BLIP = 18
# Default CP-side grace when model config is unavailable
STARTUP_GRACE_PERIOD_S = 600.0

# If more than this fraction of READY replicas fail in the same check cycle,
# treat it as a control-plane network issue — NOT a replica failure.
# Avoids mass-killing healthy replicas during transient network blips.
MASS_FAILURE_THRESHOLD = 0.5

MAX_IN_PLACE_RETRIES = 2
MAX_TOTAL_RETRIES = 3
RESTART_BACKOFF_S = 10.0
# Agent vLLM /health probe + HTTP round-trip; must exceed busy vLLM response time.
VLLM_REPLICA_STATUS_TIMEOUT_S = 50.0


class FailureClass(StrEnum):
    OOM = "OOM"
    NCCL = "NCCL"
    IMPORT = "IMPORT"
    CONFIG = "CONFIG"
    SEGFAULT = "SEGFAULT"
    UNKNOWN = "UNKNOWN"


_FAILURE_PATTERNS: list[tuple[FailureClass, re.Pattern]] = [
    (FailureClass.OOM, re.compile(
        r"CUDA out of memory|OutOfMemoryError|Free memory on device.*less than desired",
        re.IGNORECASE,
    )),
    (FailureClass.NCCL, re.compile(
        r"NCCL error|NCCL watchdog|ncclInternalError|ncclSystemError",
        re.IGNORECASE,
    )),
    (FailureClass.IMPORT, re.compile(
        r"ImportError|ModuleNotFoundError|cannot import name",
        re.IGNORECASE,
    )),
    (FailureClass.CONFIG, re.compile(
        r"ValueError:|invalid argument|unrecognized arguments",
        re.IGNORECASE,
    )),
    (FailureClass.SEGFAULT, re.compile(
        r"Segmentation fault|signal 11|SIGSEGV",
        re.IGNORECASE,
    )),
]


def classify_failure(output: str, exit_code: int | None = None) -> FailureClass:
    """Classify a vLLM crash from its stderr output and exit code."""
    if exit_code is not None and exit_code == -11:
        return FailureClass.SEGFAULT
    for cls, pattern in _FAILURE_PATTERNS:
        if pattern.search(output):
            return cls
    return FailureClass.UNKNOWN


def is_restartable_in_place(cls: FailureClass) -> bool:
    """Whether this failure class is worth retrying on the same GPUs."""
    return cls in (
        FailureClass.OOM,
        FailureClass.NCCL,
        FailureClass.SEGFAULT,
        FailureClass.UNKNOWN,
    )


@dataclass
class ReplicaRemediationState:
    """Tracks remediation attempts per replica across restarts."""
    replica_id: str
    model: str
    node_name: str
    gpu_ids: list[int]
    in_place_retries: int = 0
    total_retries: int = 0
    last_failure_class: FailureClass = FailureClass.UNKNOWN
    last_failure_output: str = ""
    last_failure_at: float = 0.0
    last_health_reason: str = ""
    remediation_history: list[dict] = field(default_factory=list)


class HealthManager:
    def __init__(
        self,
        registry,
        db,
        safety_config: SafetyConfig,
        node_agent_client=None,
        gpu_guard: GpuMemoryGuard | None = None,
        restart_callback=None,
        migrate_callback=None,
        models_config: dict[str, ModelConfig] | None = None,
    ) -> None:
        self.registry = registry
        self.db = db
        self.safety = safety_config
        self.node_agent_client = node_agent_client
        self.gpu_guard = gpu_guard
        self._restart_callback = restart_callback
        self._migrate_callback = migrate_callback
        self._models_config = models_config or {}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._http_client: httpx.AsyncClient | None = None

        self._remediation: dict[str, ReplicaRemediationState] = {}
        self._last_health_fail_reason: dict[str, str] = {}
        self._recent_incidents: list[dict] = []
        self._max_incidents = 200

    def _startup_grace_s(self, model_name: str) -> float:
        """How long STARTING replicas may fail /health before remediation."""
        mc = self._models_config.get(model_name)
        if mc is not None:
            return float(mc.autoscaling.replica_startup_timeout_s)
        return STARTUP_GRACE_PERIOD_S

    @property
    def recent_incidents(self) -> list[dict]:
        return list(self._recent_incidents)

    async def start(self) -> None:
        self._running = True
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(8.0, connect=3.0),
        )
        self._tasks = [
            asyncio.create_task(self._node_check_loop()),
            asyncio.create_task(self._replica_check_loop()),
            asyncio.create_task(self._gpu_guard_loop()),
        ]
        log.info("health manager started",
                 gpu_memory_limit=f"{self.safety.gpu_memory_limit:.0%}",
                 gpu_compute_sustain=(
                     f">={self.safety.gpu_compute_sustain_threshold:.0%} "
                     f"for {self.safety.gpu_compute_sustain_duration_s:.0f}s"
                 ),
                 mitigation_window=f"{self.safety.guard_mitigation_window_s}s",
                 max_in_place_retries=MAX_IN_PLACE_RETRIES)

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._http_client:
            await self._http_client.aclose()
        log.info("health manager stopped")

    # ─── Layer 1: Node health ────────────────────────────────────────────

    async def _node_check_loop(self) -> None:
        while self._running:
            try:
                await self._check_all_nodes()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("node check error", error=str(e))
            await asyncio.sleep(NODE_CHECK_INTERVAL_S)

    async def _check_all_nodes(self) -> None:
        nodes = self.registry.get_all_nodes()
        for node in nodes:
            start = time.time()
            try:
                if self.node_agent_client:
                    ok = await self.node_agent_client.ping(node.name)
                else:
                    # Fallback: just check if we've had a recent heartbeat
                    ok = (time.time() - node.last_heartbeat) < NODE_CHECK_INTERVAL_S * 3

                elapsed = time.time() - start
                HEALTH_CHECK_DURATION.labels(check_type="node").observe(elapsed)

                if ok:
                    self.registry.record_node_success(node.name)
                else:
                    failures = self.registry.record_node_failure(node.name)
                    log.warning("node health check failed",
                                node=node.name, consecutive=failures)

                    if failures >= MAX_CONSECUTIVE_NODE_FAILURES:
                        log.error("node marked offline", node=node.name)
                        self.registry.set_node_status(node.name, NodeStatus.OFFLINE)
                        HEALTH_INCIDENTS.labels(incident_type="node_offline").inc()
                        await self.db.log_health_incident(
                            incident_type="node_offline",
                            node_name=node.name,
                            details=f"Failed {failures} consecutive health checks",
                        )
                        # Drain all replicas on this node
                        await self._drain_node_replicas(node.name)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("node check exception", node=node.name, error=str(e))
                self.registry.record_node_failure(node.name)

    async def _drain_node_replicas(self, node_name: str) -> None:
        replicas = self.registry.get_all_replicas()
        for r in replicas:
            if r.node_name == node_name and not r.status.is_terminal():
                try:
                    if r.status.can_accept_requests():
                        self.registry.set_replica_status(r.replica_id, ReplicaStatus.DRAINING)
                    self.registry.set_replica_status(r.replica_id, ReplicaStatus.FAILED)
                    log.warning("replica marked failed due to node offline",
                                replica=r.replica_id, node=node_name)
                except Exception as e:
                    log.error("failed to drain replica", replica=r.replica_id, error=str(e))

    # ─── Layer 2: Replica health ─────────────────────────────────────────

    async def _replica_is_healthy(self, replica) -> tuple[bool, str]:
        """Check replica health via node agent (vLLM is often not reachable from head).

        Returns (healthy, reason) for logging and incidents.
        """
        if self.node_agent_client:
            try:
                st = await self.node_agent_client.get_replica_status(
                    replica.node_name,
                    replica.replica_id,
                    timeout_s=VLLM_REPLICA_STATUS_TIMEOUT_S,
                )
                if st.get("error"):
                    return False, f"agent_error:{st['error']}"
                rep = st.get("replica") or {}
                if rep.get("health_ok"):
                    return True, "agent_vllm_health_ok"
                if rep.get("alive") and not rep.get("health_ok"):
                    cfg = self._models_config.get(replica.model)
                    if cfg and cfg.gpu_guard_exempt and replica.status in (
                        ReplicaStatus.READY,
                        ReplicaStatus.DRAINING,
                    ):
                        return True, (
                            f"agent_vllm_alive_health_slow port={rep.get('port')} "
                            f"pid={rep.get('pid')}"
                        )
                    return False, (
                        f"vllm_not_ready port={rep.get('port')} "
                        f"pid={rep.get('pid')}"
                    )
                if rep:
                    return False, f"agent_replica_not_alive status={rep.get('status')}"
                return False, "agent_empty_replica_payload"
            except Exception as e:
                return False, f"agent_unreachable:{type(e).__name__}:{e}"
        if not replica.http_endpoint:
            return False, "no_http_endpoint"
        try:
            resp = await self._http_client.get(
                f"{replica.http_endpoint}/health",
                timeout=5.0,
            )
            if resp.status_code == 200:
                return True, "head_direct_health_ok"
            return False, f"head_direct_health_status_{resp.status_code}"
        except Exception as e:
            return False, f"head_direct_unreachable:{type(e).__name__}:{e}"

    async def _replica_check_loop(self) -> None:
        while self._running:
            try:
                await self._check_all_replicas()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("replica check error", error=str(e))
            await asyncio.sleep(REPLICA_CHECK_INTERVAL_S)

    async def _check_all_replicas(self) -> None:
        replicas = self.registry.get_all_replicas()
        now = time.time()

        checkable = [
            r for r in replicas
            if not r.status.is_terminal() and r.http_endpoint
        ]
        ready_count = sum(1 for r in checkable if r.status == ReplicaStatus.READY)
        failed_this_cycle = 0

        for replica in checkable:
            grace = self._startup_grace_s(replica.model)
            in_startup = (
                replica.status == ReplicaStatus.STARTING
                and (now - replica.started_at) < grace
            )

            start = time.time()
            healthy, health_reason = await self._replica_is_healthy(replica)

            elapsed = time.time() - start
            HEALTH_CHECK_DURATION.labels(check_type="replica").observe(elapsed)

            if healthy:
                self.registry.update_replica_health(replica.replica_id, True)
                if replica.replica_id in self._remediation:
                    del self._remediation[replica.replica_id]
                cfg = self._models_config.get(replica.model)
                if replica.status == ReplicaStatus.DRAINING and cfg and cfg.gpu_guard_exempt:
                    try:
                        self.registry.set_replica_status(replica.replica_id, ReplicaStatus.READY)
                        log.info(
                            "replica resumed: model is GPU-guard exempt and process is alive",
                            replica=replica.replica_id,
                            model=replica.model,
                        )
                    except Exception:
                        pass
                if replica.status == ReplicaStatus.STARTING:
                    try:
                        self.registry.set_replica_status(replica.replica_id, ReplicaStatus.READY)
                        log.info("replica became ready", replica=replica.replica_id,
                                 model=replica.model, startup_s=f"{now - replica.started_at:.0f}")
                    except Exception:
                        pass
                if replica.replica_id in self._remediation:
                    del self._remediation[replica.replica_id]
                continue

            # Slow /health under load is not a crash (VRAM guard is separate).
            if (
                health_reason.startswith("vllm_not_ready")
                and replica.inflight_requests > 0
            ):
                self.registry.update_replica_health(replica.replica_id, True)
                if replica.replica_id in self._remediation:
                    del self._remediation[replica.replica_id]
                continue

            self.registry.update_replica_health(replica.replica_id, False)
            failed_this_cycle += 1
            self._last_health_fail_reason[replica.replica_id] = health_reason

            if in_startup:
                continue

            failures = replica.consecutive_health_failures + 1
            log.warning("replica health check failed",
                        replica=replica.replica_id, model=replica.model,
                        consecutive=failures, reason=health_reason,
                        check_elapsed_s=f"{elapsed:.2f}")
            await self.db.log_health_incident(
                incident_type="replica_health_check_failed",
                node_name=replica.node_name,
                replica_id=replica.replica_id,
                details=(
                    f"reason={health_reason} consecutive={failures} "
                    f"status={replica.status.value} endpoint={replica.http_endpoint}"
                ),
            )

            threshold = MAX_CONSECUTIVE_REPLICA_FAILURES
            if (
                health_reason.startswith("agent_unreachable:")
                or health_reason.startswith("agent_empty_replica_payload")
                or health_reason.startswith("agent_error:Unknown replica")
            ):
                threshold = MAX_CONSECUTIVE_REPLICA_FAILURES_AGENT_BLIP

            if failures >= threshold:
                # Mass-failure detection: if most READY replicas are unreachable
                # this cycle, it's likely a control-plane network issue, not
                # individual replica crashes. Skip remediation to avoid killing
                # healthy replicas during transient network blips.
                if ready_count > 2 and (failed_this_cycle / ready_count) >= MASS_FAILURE_THRESHOLD:
                    log.warning(
                        "mass failure detected — skipping remediation "
                        "(likely control-plane network issue)",
                        failed=failed_this_cycle, ready=ready_count,
                        threshold=MASS_FAILURE_THRESHOLD,
                        replica=replica.replica_id, model=replica.model)
                    continue

                await self._handle_replica_failure(replica)

    # ─── Layer 2b: Intelligent Remediation ───────────────────────────────

    async def _handle_replica_failure(self, replica) -> None:
        """Classify the failure and apply the cheapest effective remedy."""
        rid = replica.replica_id

        output = ""
        exit_code = None
        if self.node_agent_client:
            try:
                info = await self.node_agent_client.get_replica_status(
                    replica.node_name,
                    rid,
                    timeout_s=VLLM_REPLICA_STATUS_TIMEOUT_S,
                )
                output = info.get("output_tail", "")
                exit_code = info.get("exit_code")
            except Exception:
                pass

        failure_cls = classify_failure(output, exit_code)

        last_reason = self._last_health_fail_reason.get(rid, "")
        if last_reason.startswith("agent_error:Unknown replica"):
            log.warning(
                "skipping remediation: agent restarted, replica not yet re-registered",
                replica=rid,
                model=replica.model,
            )
            self.registry.update_replica_health(rid, True)
            self._last_health_fail_reason.pop(rid, None)
            return

        if (
            last_reason.startswith("vllm_not_ready")
            and failure_cls == FailureClass.UNKNOWN
            and not (output or "").strip()
        ):
            log.warning(
                "skipping remediation: slow vLLM health probe, no crash evidence",
                replica=rid,
                model=replica.model,
                last_health_reason=last_reason,
            )
            self.registry.update_replica_health(rid, True)
            self._last_health_fail_reason.pop(rid, None)
            return

        state = self._remediation.get(rid)
        if not state:
            state = ReplicaRemediationState(
                replica_id=rid,
                model=replica.model,
                node_name=replica.node_name,
                gpu_ids=list(replica.gpu_ids),
            )
            self._remediation[rid] = state

        state.last_health_reason = self._last_health_fail_reason.pop(
            rid, "unknown",
        )
        state.last_failure_class = failure_cls
        state.last_failure_output = output[:500]
        state.last_failure_at = time.time()
        state.total_retries += 1

        log.error("replica failure classified",
                  replica=rid, model=replica.model,
                  failure_class=failure_cls.value,
                  in_place_retries=state.in_place_retries,
                  total_retries=state.total_retries,
                  last_health_reason=state.last_health_reason)

        await self.db.log_health_incident(
            incident_type=f"replica_failure_{failure_cls.value.lower()}",
            node_name=replica.node_name,
            replica_id=rid,
            details=(
                f"failure_class={failure_cls.value} "
                f"last_health_reason={state.last_health_reason} "
                f"output_tail={output[:300]!r}"
            ),
        )

        HEALTH_INCIDENTS.labels(
            incident_type=f"replica_{failure_cls.value.lower()}",
        ).inc()

        can_restart = (
            is_restartable_in_place(failure_cls)
            and state.in_place_retries < MAX_IN_PLACE_RETRIES
        )

        if can_restart:
            await self._restart_in_place(replica, state, failure_cls)
        elif state.total_retries <= MAX_TOTAL_RETRIES:
            await self._migrate_replica(replica, state, failure_cls)
        else:
            await self._give_up(replica, state, failure_cls)

    async def _restart_in_place(self, replica, state: ReplicaRemediationState,
                                failure_cls: FailureClass) -> None:
        """Kill the replica and relaunch on the same GPUs."""
        rid = replica.replica_id
        state.in_place_retries += 1

        action = {
            "timestamp": time.time(),
            "action": "restart_in_place",
            "failure_class": failure_cls.value,
            "attempt": state.in_place_retries,
            "node": replica.node_name,
            "gpus": list(replica.gpu_ids),
        }
        state.remediation_history.append(action)
        self._record_incident(action, replica)

        log.info("attempting in-place restart",
                 replica=rid, model=replica.model,
                 failure_class=failure_cls.value,
                 attempt=state.in_place_retries,
                 max_attempts=MAX_IN_PLACE_RETRIES)

        if self.node_agent_client:
            try:
                await self.node_agent_client.stop_replica(
                    replica.node_name, rid, force=True)
            except Exception as e:
                log.warning("stop before restart failed",
                            replica=rid, error=str(e))

        await asyncio.sleep(RESTART_BACKOFF_S)

        if self._restart_callback:
            try:
                await self._restart_callback(rid, replica.model,
                                             replica.node_name,
                                             list(replica.gpu_ids))
                log.info("in-place restart initiated",
                         replica=rid, model=replica.model)
            except Exception as e:
                log.error("in-place restart failed, will escalate",
                          replica=rid, error=str(e))
                await self._migrate_replica(replica, state, failure_cls)
        else:
            try:
                self.registry.set_replica_status(rid, ReplicaStatus.FAILED)
            except Exception:
                pass

    async def _migrate_replica(self, replica, state: ReplicaRemediationState,
                               failure_cls: FailureClass) -> None:
        """Release current GPUs and relaunch on another node (via control-plane callback)."""
        rid = replica.replica_id
        node_name = replica.node_name

        action = {
            "timestamp": time.time(),
            "action": "migrate",
            "failure_class": failure_cls.value,
            "from_node": node_name,
            "reason": (f"in-place retries exhausted ({state.in_place_retries})"
                       if state.in_place_retries >= MAX_IN_PLACE_RETRIES
                       else f"non-restartable error: {failure_cls.value}"),
        }
        state.remediation_history.append(action)
        self._record_incident(action, replica)

        log.info("escalating to migration",
                 replica=rid, model=replica.model,
                 failure_class=failure_cls.value,
                 from_node=node_name)

        # Next scale-up / replacement launch will deprioritize this node (see ``_launch_replica``).
        self.registry.record_launch_failure_event(
            replica.model,
            node_name,
            f"health_migrate:{failure_cls.value}",
        )

        # Delegate to control plane: drain/stop/remove + launch replacement (often on another node).
        if self._migrate_callback:
            try:
                await self._migrate_callback(rid)
                self._remediation.pop(rid, None)
                await self.db.log_health_incident(
                    incident_type=f"replica_migrate_{failure_cls.value.lower()}",
                    node_name=node_name,
                    replica_id=rid,
                    details=(
                        f"Migrated after {state.in_place_retries} in-place retries "
                        f"(failure_class={failure_cls.value})."
                    ),
                )
                return
            except Exception as e:
                log.error("migrate callback failed",
                          replica=rid, model=replica.model, error=str(e))
                await self.db.log_health_incident(
                    incident_type="replica_migrate_callback_failed",
                    node_name=node_name,
                    replica_id=rid,
                    details=str(e),
                )
                self._remediation.pop(rid, None)
                # Callback may have removed the replica already; clean up if it still exists.
                remnant = self.registry.get_replica(rid)
                if remnant and self.node_agent_client:
                    try:
                        await self.node_agent_client.stop_replica(
                            remnant.node_name, rid, force=True)
                    except Exception:
                        pass
                if self.registry.get_replica(rid):
                    try:
                        self.registry.set_replica_status(rid, ReplicaStatus.FAILED)
                    except Exception:
                        pass
                return

        # No callback (e.g. minimal test harness): best-effort local cleanup.
        if self.node_agent_client:
            try:
                await self.node_agent_client.stop_replica(node_name, rid, force=True)
            except Exception:
                pass

        try:
            self.registry.set_replica_status(rid, ReplicaStatus.FAILED)
        except Exception:
            pass

        await self.db.log_health_incident(
            incident_type=f"replica_migrate_{failure_cls.value.lower()}",
            node_name=node_name,
            replica_id=rid,
            details=(f"No migrate_callback — marked FAILED after {state.in_place_retries} "
                     f"in-place retries. Error: {failure_cls.value}"),
        )

    async def _give_up(self, replica, state: ReplicaRemediationState,
                       failure_cls: FailureClass) -> None:
        """All retries exhausted. Mark FAILED and log."""
        rid = replica.replica_id

        action = {
            "timestamp": time.time(),
            "action": "give_up",
            "failure_class": failure_cls.value,
            "total_retries": state.total_retries,
            "output_snippet": state.last_failure_output[:200],
        }
        state.remediation_history.append(action)
        self._record_incident(action, replica)

        log.error("all remediation attempts exhausted",
                  replica=rid, model=replica.model,
                  failure_class=failure_cls.value,
                  total_retries=state.total_retries)

        self.registry.record_launch_failure_event(
            replica.model,
            replica.node_name,
            f"health_give_up:{failure_cls.value}",
        )

        if self.node_agent_client:
            try:
                await self.node_agent_client.stop_replica(
                    replica.node_name, rid, force=True)
            except Exception:
                pass

        try:
            self.registry.set_replica_status(rid, ReplicaStatus.FAILED)
        except Exception:
            pass

        HEALTH_INCIDENTS.labels(incident_type="replica_give_up").inc()
        await self.db.log_health_incident(
            incident_type="replica_give_up",
            node_name=replica.node_name,
            replica_id=rid,
            details=(f"Gave up after {state.total_retries} total retries. "
                     f"Last error: {failure_cls.value}"),
        )

    def _record_incident(self, action: dict, replica) -> None:
        incident = {
            **action,
            "replica_id": replica.replica_id,
            "model": replica.model,
            "node_name": replica.node_name,
        }
        self._recent_incidents.append(incident)
        if len(self._recent_incidents) > self._max_incidents:
            self._recent_incidents = self._recent_incidents[-self._max_incidents:]

    # ─── Layer 3: GPU Memory Guard ───────────────────────────────────────

    async def _gpu_guard_loop(self) -> None:
        interval = self.safety.guard_check_interval_s
        while self._running:
            try:
                if self.gpu_guard:
                    await self.gpu_guard.check_all_gpus()
                else:
                    await self._legacy_gpu_check()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("gpu guard check error", error=str(e))
            await asyncio.sleep(interval)

    async def _legacy_gpu_check(self) -> None:
        """Fallback GPU check when no guard is configured."""
        nodes = self.registry.get_online_nodes()
        for node in nodes:
            for gpu in node.gpus:
                if gpu.memory_total_mb == 0:
                    continue
                util = gpu.memory_used_mb / gpu.memory_total_mb

                if util >= self.safety.gpu_danger_threshold:
                    log.error("GPU danger threshold exceeded",
                              node=node.name, gpu=gpu.index,
                              utilization=f"{util:.0%}",
                              threshold=f"{self.safety.gpu_danger_threshold:.0%}")
                    HEALTH_INCIDENTS.labels(incident_type="gpu_danger").inc()
                    await self.db.log_health_incident(
                        incident_type="gpu_danger",
                        node_name=node.name,
                        details=(f"GPU {gpu.index} at {util:.0%} "
                                 f"memory utilization"),
                    )
                    if self.node_agent_client:
                        try:
                            await self.node_agent_client.kill_gpu_processes(
                                node.name, [gpu.index], vllm_only=True,
                            )
                        except Exception as e:
                            log.error("failed to kill GPU processes",
                                      node=node.name, gpu=gpu.index,
                                      error=str(e))

                elif util >= self.safety.gpu_warn_threshold:
                    log.warning("GPU above warning threshold",
                                node=node.name, gpu=gpu.index,
                                utilization=f"{util:.0%}")
