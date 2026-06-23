"""Cluster Registry — the single source of truth for cluster state.

The registry owns the canonical view of:
  - Nodes (online/offline, GPU inventory)
  - Replicas (status, placement, inflight counts)
  - Model configs (loaded from YAML, immutable at runtime)

It is an in-memory data structure protected by a threading lock.  All
mutations go through explicit methods that enforce state machine
transitions and emit events to the event log (db.py).

The registry is NOT distributed.  It lives in the control plane process
on the head node.  This is intentional — a single brain avoids consensus
complexity and makes debugging trivial.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from cserve.common.logging import get_logger
from cserve.common.models import (
    AutoscalePolicy,
    ClusterConfig,
    GpuInfo,
    GpuState,
    ModelConfig,
    NodeConfig,
    NodeState,
    NodeStatus,
    ReplicaState,
    ReplicaStatus,
)

log = get_logger("registry")

# ── Circuit breaker constants ─────────────────────────────────────────────────
# Open the circuit after this many consecutive launch failures to a node.
CIRCUIT_OPEN_THRESHOLD: int = 2
# Keep the circuit open for this many seconds before allowing a half-open probe.
CIRCUIT_OPEN_DURATION_S: float = 300.0
# Recent launch failures for UI (model, node, error) — last N events
LAUNCH_FAILURE_EVENT_MAX: int = 50


class RegistryError(Exception):
    pass


class InvalidTransitionError(RegistryError):
    pass


EventCallback = Callable[[str, dict], None]


class ClusterRegistry:
    """Thread-safe in-memory cluster state.

    All reads are lock-free snapshots (we copy on read for safety).
    All writes acquire the lock and validate state transitions.
    """

    def __init__(self, cluster_config: ClusterConfig) -> None:
        self._lock = threading.RLock()
        self._nodes: dict[str, NodeState] = {}
        self._replicas: dict[str, ReplicaState] = {}
        self._models: dict[str, ModelConfig] = {}
        self._event_callbacks: list[EventCallback] = []
        self._recent_launch_failures: deque[dict] = deque(maxlen=LAUNCH_FAILURE_EVENT_MAX)

        self._init_nodes(cluster_config)

    # ─── Initialization ──────────────────────────────────────────────────

    def _init_nodes(self, cfg: ClusterConfig) -> None:
        """Build initial node states from static config."""
        for node_cfg in cfg.nodes:
            gpus = self._build_gpu_list(node_cfg)
            [int(d) for d in node_cfg.cuda_devices.split(",") if d.strip()] if node_cfg.cuda_devices else []
            state = NodeState(
                name=node_cfg.name,
                host=node_cfg.host,
                gpu_type=node_cfg.gpu_type,
                status=NodeStatus.OFFLINE,
                agent_endpoint=f"{node_cfg.host}:{cfg.node_agent.port}",
                gpus=gpus,
                labels=node_cfg.labels,
                schedulable=node_cfg.schedulable,
            )
            self._nodes[node_cfg.name] = state

    @staticmethod
    def _build_gpu_list(node_cfg: NodeConfig) -> list[GpuInfo]:
        if not node_cfg.cuda_devices:
            return []
        indices = [int(d) for d in node_cfg.cuda_devices.split(",") if d.strip()]
        return [
            GpuInfo(index=idx, name=node_cfg.gpu_type.upper() if node_cfg.gpu_type else "")
            for idx in indices
        ]

    # ─── Dynamic node management ─────────────────────────────────────────

    def add_node(self, node_cfg: NodeConfig, agent_port: int) -> None:
        """Hot-add a new node to the registry (no control-plane restart needed)."""
        with self._lock:
            if node_cfg.name in self._nodes:
                log.info("node already registered, skipping", node=node_cfg.name)
                return
            gpus = self._build_gpu_list(node_cfg)
            state = NodeState(
                name=node_cfg.name,
                host=node_cfg.host,
                gpu_type=node_cfg.gpu_type,
                status=NodeStatus.OFFLINE,
                agent_endpoint=f"{node_cfg.host}:{agent_port}",
                gpus=gpus,
                labels=node_cfg.labels,
                schedulable=node_cfg.schedulable,
            )
            self._nodes[node_cfg.name] = state
            log.info("node hot-registered", node=node_cfg.name, gpus=len(gpus))

    def remove_node(self, node_name: str, force: bool = False) -> tuple[bool, str]:
        """Remove a node. Returns (success, reason).

        Only succeeds if the node has no STARTING/READY/DRAINING replicas
        unless force=True (which also removes those replicas from tracking).
        """
        with self._lock:
            if node_name not in self._nodes:
                return False, f"Node '{node_name}' not found"

            active = [
                r for r in self._replicas.values()
                if r.node_name == node_name
                and r.status not in (ReplicaStatus.FAILED,)
            ]
            if active and not force:
                names = ", ".join(r.replica_id for r in active)
                return False, (
                    f"Node '{node_name}' has {len(active)} active replica(s): {names}. "
                    "Drain replicas first or use force=true."
                )

            if force:
                for r in active:
                    del self._replicas[r.replica_id]

            del self._nodes[node_name]
            log.info("node removed", node=node_name, force=force, evicted_replicas=len(active) if force else 0)
            return True, "ok"

    def set_node_schedulable(self, node_name: str, schedulable: bool) -> bool:
        """Update whether a node accepts new replica placements."""
        with self._lock:
            node = self._nodes.get(node_name)
            if node is None:
                return False
            node.schedulable = schedulable
            return True

    def register_event_callback(self, cb: EventCallback) -> None:
        self._event_callbacks.append(cb)

    def _emit(self, event_type: str, data: dict) -> None:
        for cb in self._event_callbacks:
            try:
                cb(event_type, data)
            except Exception:
                log.warning("event callback failed", event_type=event_type)

    # ─── Model config ────────────────────────────────────────────────────

    def load_models(self, models: dict[str, ModelConfig]) -> None:
        with self._lock:
            self._models = dict(models)
        log.info("loaded model configs", count=len(models),
                 models=list(models.keys()))

    def get_model_config(self, model_name: str) -> ModelConfig | None:
        return self._models.get(model_name)

    def get_all_model_configs(self) -> dict[str, ModelConfig]:
        return dict(self._models)

    def get_autoscale_policy(self, model_name: str) -> AutoscalePolicy | None:
        m = self._models.get(model_name)
        return m.autoscaling if m else None

    # ─── Node operations ─────────────────────────────────────────────────

    def get_node(self, name: str) -> NodeState | None:
        return self._nodes.get(name)

    def get_all_nodes(self) -> list[NodeState]:
        return list(self._nodes.values())

    def get_online_nodes(self) -> list[NodeState]:
        return [n for n in self._nodes.values() if n.status != NodeStatus.OFFLINE]

    def get_available_nodes(self) -> list[NodeState]:
        """Return online nodes whose launch circuit breaker is NOT open.

        Use this for placement decisions — it automatically excludes nodes
        that are unreachable from the control plane, even if they are still
        technically ONLINE (e.g. one-way firewall: node can send heartbeats
        but CP cannot reach the node agent port).
        """
        now = time.time()
        available = []
        for n in self._nodes.values():
            if n.status == NodeStatus.OFFLINE:
                continue
            if not n.schedulable:
                continue
            if n.circuit_open_until > 0 and now < n.circuit_open_until:
                continue  # circuit open — skip
            available.append(n)
        return available

    def set_node_status(self, name: str, status: NodeStatus) -> None:
        with self._lock:
            node = self._nodes.get(name)
            if not node:
                raise RegistryError(f"Unknown node: {name}")
            old = node.status
            node.status = status
            if old != status:
                log.info("node status change", node=name,
                         old_status=old.value, new_status=status.value)
                self._emit("node_status_change", {
                    "node": name, "old": old.value, "new": status.value,
                })

    def record_heartbeat(self, name: str) -> None:
        """Record an inbound heartbeat from a node agent.

        Important: this does NOT reset consecutive_failures.  Consecutive
        failures track the control-plane's ability to *reach* the node
        (outbound health-check ping).  A node that sends heartbeats but whose
        port is unreachable from the CP (e.g. blocked by a firewall) should
        still accumulate failures and go OFFLINE — not be kept alive by its
        own heartbeats.
        """
        with self._lock:
            node = self._nodes.get(name)
            if node:
                node.last_heartbeat = time.time()
                if node.status == NodeStatus.OFFLINE:
                    # Only bring back online if the CP can actually reach the
                    # node (consecutive_failures will be reset by a successful
                    # outbound ping in record_node_success below).
                    pass

    def record_node_success(self, name: str) -> None:
        """Record a successful outbound health-check ping to the node.

        Resets the consecutive failure counter, marks the node ONLINE, and
        closes the launch circuit breaker — a reachable node is healthy again.
        """
        with self._lock:
            node = self._nodes.get(name)
            if node:
                node.consecutive_failures = 0
                # Close the circuit: a healthy ping proves the node is reachable.
                if node.circuit_open_until > 0:
                    log.info("circuit closed (health check passed)", node=name)
                node.consecutive_launch_failures = 0
                node.circuit_open_until = 0.0
                if node.status == NodeStatus.OFFLINE:
                    node.status = NodeStatus.ONLINE
                    log.info("node came online", node=name)

    # ── Launch circuit breaker ────────────────────────────────────────────────

    def record_launch_failure_event(self, model: str, node: str, error: str) -> None:
        """Record a launch failure event for UI display (FAILED - retrying)."""
        with self._lock:
            self._recent_launch_failures.append({
                "model": model,
                "node": node,
                "error": error[:200] if error else "",
                "ts": time.time(),
            })

    def get_recent_launch_failures(self, within_s: float = 300.0) -> list[dict]:
        """Return launch failures from the last within_s seconds."""
        cutoff = time.time() - within_s
        return [e for e in self._recent_launch_failures if e["ts"] >= cutoff]

    def record_launch_failure(self, name: str) -> None:
        """Record a failed replica launch attempt on this node.

        After CIRCUIT_OPEN_THRESHOLD consecutive failures the circuit opens
        and the node is excluded from placement for CIRCUIT_OPEN_DURATION_S.
        Callers should call this whenever a launch results in a ConnectTimeout
        or similar network error that indicates the node is unreachable.
        """
        with self._lock:
            node = self._nodes.get(name)
            if not node:
                return
            node.consecutive_launch_failures += 1
            if node.consecutive_launch_failures >= CIRCUIT_OPEN_THRESHOLD:
                node.circuit_open_until = time.time() + CIRCUIT_OPEN_DURATION_S
                log.warning(
                    "circuit opened — node excluded from placement",
                    node=name,
                    failures=node.consecutive_launch_failures,
                    reopen_in_s=CIRCUIT_OPEN_DURATION_S,
                )

    def record_launch_success(self, name: str) -> None:
        """Record a successful replica launch on this node. Closes the circuit."""
        with self._lock:
            node = self._nodes.get(name)
            if node:
                node.consecutive_launch_failures = 0
                node.circuit_open_until = 0.0

    def is_circuit_open(self, name: str) -> bool:
        """Return True if the node's launch circuit is currently open.

        A HALF-OPEN state (timer expired but no success yet) returns False so
        that one probe launch is allowed — if it succeeds the circuit closes,
        if it fails the circuit reopens for another CIRCUIT_OPEN_DURATION_S.
        """
        with self._lock:
            node = self._nodes.get(name)
            if not node:
                return False
            if node.circuit_open_until <= 0:
                return False
            if time.time() < node.circuit_open_until:
                return True
            # Timer expired → half-open: reset so the next launch can probe
            log.info("circuit half-open — allowing probe launch", node=name)
            node.circuit_open_until = 0.0
            node.consecutive_launch_failures = 0
            return False

    def record_node_failure(self, name: str) -> int:
        """Record a failed outbound health-check. Returns the new consecutive failure count."""
        with self._lock:
            node = self._nodes.get(name)
            if not node:
                return 0
            node.consecutive_failures += 1
            return node.consecutive_failures

    def update_gpu_info(self, node_name: str, gpus: list[GpuInfo]) -> None:
        """Update GPU status from a node agent report."""
        with self._lock:
            node = self._nodes.get(node_name)
            if not node:
                return
            # Preserve allocation state — agent reports hardware metrics,
            # but the registry owns the allocation state
            gpu_by_idx = {g.index: g for g in node.gpus}
            for new_gpu in gpus:
                existing = gpu_by_idx.get(new_gpu.index)
                if existing:
                    existing.memory_used_mb = new_gpu.memory_used_mb
                    existing.memory_total_mb = new_gpu.memory_total_mb
                    existing.utilization_pct = new_gpu.utilization_pct
                    existing.temperature_c = new_gpu.temperature_c
                    existing.uuid = new_gpu.uuid or existing.uuid
                    existing.name = new_gpu.name or existing.name

    # ─── GPU allocation ──────────────────────────────────────────────────

    def get_free_gpus(self, node_name: str) -> list[GpuInfo]:
        node = self._nodes.get(node_name)
        if not node:
            return []
        return [g for g in node.gpus if g.state == GpuState.FREE]

    def allocate_gpus(self, node_name: str, gpu_indices: list[int], replica_id: str) -> None:
        """Mark GPUs as allocated. Raises if any are not FREE."""
        with self._lock:
            node = self._nodes.get(node_name)
            if not node:
                raise RegistryError(f"Unknown node: {node_name}")
            for gpu in node.gpus:
                if gpu.index in gpu_indices:
                    if gpu.state != GpuState.FREE:
                        raise RegistryError(
                            f"GPU {gpu.index} on {node_name} is {gpu.state.value}, "
                            f"cannot allocate to {replica_id}"
                        )
                    gpu.state = GpuState.ALLOCATED
                    gpu.allocated_replica_id = replica_id

    def release_gpus(self, node_name: str, gpu_indices: list[int]) -> None:
        """Mark GPUs as free."""
        with self._lock:
            node = self._nodes.get(node_name)
            if not node:
                return
            for gpu in node.gpus:
                if gpu.index in gpu_indices:
                    gpu.state = GpuState.FREE
                    gpu.allocated_replica_id = None

    # ─── Replica operations ──────────────────────────────────────────────

    def add_replica(self, replica: ReplicaState) -> None:
        with self._lock:
            if replica.replica_id in self._replicas:
                raise RegistryError(f"Replica already exists: {replica.replica_id}")
            self._replicas[replica.replica_id] = replica
            log.info("replica added", replica=replica.replica_id,
                     model=replica.model, node=replica.node_name,
                     gpus=replica.gpu_ids, status=replica.status.value)
            self._emit("replica_added", {
                "replica_id": replica.replica_id,
                "model": replica.model,
                "node": replica.node_name,
            })

    def set_replica_status(self, replica_id: str, new_status: ReplicaStatus) -> None:
        """Transition a replica to a new status. Enforces the state machine.

        Idempotent: if the replica is already in new_status the call is a no-op
        (e.g. repeated READY heartbeats from a healthy node agent).
        """
        with self._lock:
            replica = self._replicas.get(replica_id)
            if not replica:
                raise RegistryError(f"Unknown replica: {replica_id}")
            if replica.status == new_status:
                return  # idempotent, nothing to do
            if not replica.can_transition_to(new_status):
                raise InvalidTransitionError(
                    f"Replica {replica_id} cannot transition "
                    f"{replica.status.value} → {new_status.value}"
                )
            old = replica.status
            replica.status = new_status
            log.info("replica status change", replica=replica_id,
                     model=replica.model, old_status=old.value,
                     new_status=new_status.value)
            self._emit("replica_status_change", {
                "replica_id": replica_id,
                "model": replica.model,
                "old": old.value,
                "new": new_status.value,
            })

    def remove_replica(self, replica_id: str) -> ReplicaState | None:
        """Remove a replica from the registry and release its GPUs."""
        with self._lock:
            replica = self._replicas.pop(replica_id, None)
            if replica:
                self.release_gpus(replica.node_name, replica.gpu_ids)
                log.info("replica removed", replica=replica_id,
                         model=replica.model, node=replica.node_name)
                self._emit("replica_removed", {
                    "replica_id": replica_id,
                    "model": replica.model,
                })
            return replica

    def get_replica(self, replica_id: str) -> ReplicaState | None:
        return self._replicas.get(replica_id)

    def get_replicas_for_model(self, model: str, variant: str = "default") -> list[ReplicaState]:
        return [
            r for r in self._replicas.values()
            if r.model == model and r.variant == variant
        ]

    def get_healthy_replicas(self, model: str, variant: str = "default") -> list[ReplicaState]:
        """Get replicas that can accept requests (status=READY)."""
        return [
            r for r in self._replicas.values()
            if r.model == model and r.variant == variant
            and r.status.can_accept_requests()
        ]

    def get_routable_replicas(self, model: str, variant: str = "default") -> list[ReplicaState]:
        """READY replicas not in a gateway cooldown window (after upstream connect failures)."""
        now = time.time()
        return [
            r for r in self._replicas.values()
            if r.model == model and r.variant == variant
            and r.status.can_accept_requests()
            and now >= r.gateway_route_cooldown_until
        ]

    def mark_upstream_connection_failed(
        self, replica_id: str, cooldown_s: float = 20.0,
    ) -> None:
        """Temporarily stop fast-path routing after a true connect failure.

        Gateway connection problems are not health-check evidence.  Do not
        increment health failures here, or transient client-visible timeouts
        cascade into false replica restarts.
        """
        with self._lock:
            replica = self._replicas.get(replica_id)
            if not replica:
                return
            replica.gateway_route_cooldown_until = time.time() + cooldown_s
            log.warning(
                "replica marked upstream-unreachable for fast path",
                replica=replica_id,
                model=replica.model,
                cooldown_s=cooldown_s,
            )

    def get_all_replicas(self) -> list[ReplicaState]:
        return list(self._replicas.values())

    def increment_inflight(self, replica_id: str) -> None:
        # Lock-free: these are only called from the single asyncio event loop
        # (FastAPI request handlers).  No concurrent mutation possible.
        replica = self._replicas.get(replica_id)
        if replica:
            replica.inflight_requests += 1

    def decrement_inflight(self, replica_id: str) -> None:
        replica = self._replicas.get(replica_id)
        if replica and replica.inflight_requests > 0:
            replica.inflight_requests -= 1

    def update_replica_health(self, replica_id: str, healthy: bool) -> None:
        with self._lock:
            replica = self._replicas.get(replica_id)
            if not replica:
                return
            replica.last_health_check = time.time()
            replica.last_health_ok = healthy
            if healthy:
                replica.consecutive_health_failures = 0
            else:
                replica.consecutive_health_failures += 1

    def update_replica_metrics(self, replica_id: str, metrics: dict[str, float]) -> None:
        with self._lock:
            replica = self._replicas.get(replica_id)
            if replica:
                replica.metrics_snapshot = dict(metrics)

    def update_replica_endpoint(self, replica_id: str, endpoint: str, port: int, pid: int) -> None:
        with self._lock:
            replica = self._replicas.get(replica_id)
            if replica:
                replica.http_endpoint = endpoint
                replica.port = port
                replica.pid = pid

    # ─── Aggregate queries ───────────────────────────────────────────────

    def count_replicas(self, model: str, variant: str = "default") -> int:
        return len(self.get_replicas_for_model(model, variant))

    def count_ready_replicas(self, model: str, variant: str = "default") -> int:
        return len(self.get_healthy_replicas(model, variant))

    def total_gpus_by_type(self) -> dict[str, tuple[int, int]]:
        """Returns {gpu_type: (total, free)} across all online nodes."""
        result: dict[str, tuple[int, int]] = {}
        for node in self._nodes.values():
            if node.status == NodeStatus.OFFLINE:
                continue
            gt = node.gpu_type or "unknown"
            total, free = result.get(gt, (0, 0))
            total += len(node.gpus)
            free += sum(1 for g in node.gpus if g.state == GpuState.FREE)
            result[gt] = (total, free)
        return result

    def snapshot(self) -> dict:
        """Return a serializable snapshot of the entire registry for the dashboard."""
        return {
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "replicas": [
                r.model_dump(exclude={"VALID_TRANSITIONS"})
                for r in self._replicas.values()
            ],
            "models": list(self._models.keys()),
            "gpu_summary": self.total_gpus_by_type(),
            "launch_failures": self.get_recent_launch_failures(within_s=300.0),
        }
