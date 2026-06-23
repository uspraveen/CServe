"""Autoscaler — signal-driven per-model scaling decisions.

Runs as an async background task on the head node.  Every cycle (default 5s),
it evaluates each model's autoscaling policy against current metrics and
decides: SCALE_UP, SCALE_DOWN, SCALE_TO_ZERO, or HOLD.

Signal sources (per model):

  1. DemandTracker (in-process)  — fed by the gateway on EVERY request
     including the fast path.  Provides:
       - rps              (requests/sec over the last 30s)
       - avg_inflight     (mean inflight across the window)
       - peak_inflight    (max spike)
       - sustained_above  (has inflight been above threshold for N seconds?)

  2. Registry (in-memory)  — real-time inflight counts per replica,
     replica status, vLLM metrics snapshots (TTFT, cache usage).

  3. Redis queue (only used when fast path overflows)  — queue_depth,
     oldest_job_age.

The key design insight: the fast path never touches Redis, so queue_depth
is often zero.  That's fine — the DemandTracker gives us RPS and inflight
pressure directly from the request hot-path.  The autoscaler trusts the
DemandTracker as its PRIMARY signal and uses Redis queue signals as a
SECONDARY escalation signal.

Spike filtering: a short burst (< sustained_pressure_s) does NOT trigger
scale-up.  Only sustained pressure above the threshold for at least
`sustained_pressure_s` consecutive seconds fires a scale-up.  This
prevents wasting GPU resources on transient spikes.
"""

from __future__ import annotations

import asyncio
import math
import time

from cserve.common.logging import get_logger
from cserve.common.metrics import (
    AUTOSCALER_CURRENT_REPLICAS,
    AUTOSCALER_DECISIONS,
    AUTOSCALER_TARGET_REPLICAS,
)
from cserve.common.models import (
    AutoscaleAction,
    AutoscaleEvent,
    AutoscalePolicy,
    ReplicaStatus,
)
from cserve.control_plane.deployment_precedence import (
    models_by_deploy_priority,
    precedence_blocks_scale_up,
)

log = get_logger("autoscaler")

AUTOSCALE_INTERVAL_S = 5.0

# Consecutive "no placement" launch failures before scrubbing candidate nodes.
PLACEMENT_RECOVERY_THRESHOLD = 3

# How many consecutive seconds inflight must exceed the threshold before
# we consider it "sustained" (vs a transient spike).
SUSTAINED_PRESSURE_S = 10

# RPS per replica above which we consider a model saturated even if
# instantaneous inflight looks fine (fast completions hide load).
RPS_PER_REPLICA_CEILING = 15.0


class ModelScaleState:
    """Tracks per-model scaling state across cycles."""
    __slots__ = (
        "last_scale_up_at", "last_scale_down_at", "last_request_at",
    )

    def __init__(self) -> None:
        self.last_scale_up_at: float = 0.0
        self.last_scale_down_at: float = 0.0
        self.last_request_at: float = time.time()


class Autoscaler:
    def __init__(
        self, registry, queue, db,
        demand_tracker=None,
        launch_callback=None,
        stop_callback=None,
        node_client=None,
        node_cuda_devices: dict[str, list[int]] | None = None,
    ) -> None:
        self.registry = registry
        self.queue = queue
        self.db = db
        self.demand_tracker = demand_tracker
        self._running = False
        self._task: asyncio.Task | None = None
        self._scale_states: dict[str, ModelScaleState] = {}

        self._launch_callback = launch_callback
        self._stop_callback = stop_callback
        self._node_client = node_client
        self._node_cuda_devices = node_cuda_devices or {}
        self._placement_fail_streak: dict[str, int] = {}
        self._paused = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info("autoscaler started", interval_s=AUTOSCALE_INTERVAL_S)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("autoscaler stopped")

    def set_paused(self, paused: bool) -> None:
        """Pause scale up/down evaluation (used during cluster-wide stop)."""
        self._paused = paused
        log.info("autoscaler pause state", paused=paused)

    @property
    def paused(self) -> bool:
        return self._paused

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._evaluate_all_models()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("autoscaler loop error", error=str(e))
            await asyncio.sleep(AUTOSCALE_INTERVAL_S)

    async def _evaluate_all_models(self) -> None:
        if self._paused:
            return
        models = self.registry.get_all_model_configs()
        for model_name, model_cfg in models_by_deploy_priority(models):
            await self._cleanup_failed_replicas(model_name)

            policy = model_cfg.autoscaling
            state = self._scale_states.setdefault(model_name, ModelScaleState())

            signals = await self._gather_signals(model_name, policy)

            action, reasons, target_replicas = self._decide(
                model_name, policy, state, signals
            )

            blocked, wait_for = precedence_blocks_scale_up(
                model_name, self.registry, models,
            )
            if blocked and action == AutoscaleAction.SCALE_UP:
                action = AutoscaleAction.HOLD
                reasons = [
                    f"deployment_precedence: waiting for {wait_for} "
                    f"(min replicas READY before {model_name})",
                ]
                target_replicas = int(signals["current_replicas"])

            current = int(signals["current_replicas"])
            AUTOSCALER_CURRENT_REPLICAS.labels(model=model_name).set(current)
            AUTOSCALER_TARGET_REPLICAS.labels(model=model_name).set(target_replicas)
            AUTOSCALER_DECISIONS.labels(model=model_name, action=action.value).inc()

            event = AutoscaleEvent(
                model=model_name,
                action=action,
                from_replicas=current,
                to_replicas=target_replicas,
                reasons=reasons,
                metrics_snapshot=signals,
            )
            await self.db.log_autoscale_event(event)

            if action == AutoscaleAction.SCALE_UP:
                needed = target_replicas - current
                log.info("scaling up", model=model_name, current=current,
                         target=target_replicas, reasons=reasons)
                state.last_scale_up_at = time.time()
                if self._launch_callback:
                    for _ in range(needed):
                        # Fire-and-forget: each launch runs independently so
                        # a timeout on one node never stalls other launches or
                        # other models.  Errors are caught inside _safe_launch.
                        asyncio.create_task(
                            self._safe_launch(model_name),
                            name=f"launch-{model_name}",
                        )

            elif action in (AutoscaleAction.SCALE_DOWN, AutoscaleAction.SCALE_TO_ZERO):
                excess = current - target_replicas
                log.info("scaling down", model=model_name, current=current,
                         target=target_replicas, reasons=reasons)
                state.last_scale_down_at = time.time()
                replicas = self.registry.get_replicas_for_model(model_name)
                replicas.sort(key=lambda r: r.inflight_requests)
                for r in replicas[:excess]:
                    if self._stop_callback:
                        try:
                            await self._stop_callback(r.replica_id)
                        except Exception as e:
                            log.error("stop failed", replica=r.replica_id, error=str(e))

    async def _safe_launch(self, model_name: str) -> None:
        """Fire-and-forget wrapper around the launch callback.

        Catches all exceptions so a failed launch never bubbles up to and
        cancels the autoscaler loop.  The circuit breaker in the registry
        ensures that if the launch failed because a node was unreachable, that
        node is skipped for future placements automatically.
        """
        try:
            await self._launch_callback(model_name)
            self._placement_fail_streak.pop(model_name, None)
        except Exception as e:
            log.error("launch failed", model=model_name, error=str(e))
            if "no placement available" in str(e).lower():
                streak = self._placement_fail_streak.get(model_name, 0) + 1
                self._placement_fail_streak[model_name] = streak
                if streak >= PLACEMENT_RECOVERY_THRESHOLD:
                    await self._recover_placement_pressure(model_name)
                    self._placement_fail_streak[model_name] = 0

    async def _recover_placement_pressure(self, model_name: str) -> None:
        """Scrub orphan vLLM on nodes that can host this model (placement blocked)."""
        if not self._node_client:
            return
        model_cfg = self.registry.get_model_config(model_name)
        if not model_cfg:
            return

        allowed_nodes = set(model_cfg.nodes_allowed) if model_cfg.nodes_allowed else None
        allowed_types = set(model_cfg.node_types_allowed or [])
        required_type = model_cfg.node_type_required

        for node in self.registry.get_all_nodes():
            if node.status.value == "offline":
                continue
            if allowed_nodes and node.name not in allowed_nodes:
                continue
            if allowed_types and node.gpu_type not in allowed_types:
                continue
            if required_type and node.gpu_type != required_type:
                continue

            gpu_ids = self._node_cuda_devices.get(node.name)
            if not gpu_ids:
                gpu_ids = [g.index for g in node.gpus]
            if not gpu_ids:
                continue

            try:
                result = await self._node_client.kill_gpu_processes(
                    node.name, gpu_ids, vllm_only=True,
                )
                killed = result.get("killed_pids") or []
                if killed:
                    log.warning(
                        "placement recovery: scrubbed vLLM on node",
                        model=model_name,
                        node=node.name,
                        gpus=gpu_ids,
                        killed_pids=killed,
                    )
            except Exception as exc:
                log.warning(
                    "placement recovery: scrub failed",
                    model=model_name,
                    node=node.name,
                    error=str(exc),
                )

    async def _cleanup_failed_replicas(self, model_name: str) -> None:
        """Stop FAILED / stuck DRAINING replicas on the node agent, then remove."""
        replicas = self.registry.get_replicas_for_model(model_name)
        for r in replicas:
            stale = r.status == ReplicaStatus.FAILED or (
                r.status in (ReplicaStatus.DRAINING, ReplicaStatus.STOPPING)
                and r.consecutive_health_failures >= 3
            )
            if not stale:
                continue
            log.info("cleaning up failed replica",
                     model=model_name, replica=r.replica_id,
                     node=r.node_name)
            if self._stop_callback:
                try:
                    await self._stop_callback(r.replica_id)
                    continue
                except Exception as e:
                    log.warning("stop during failed-replica cleanup failed",
                                replica=r.replica_id, error=str(e))
            try:
                self.registry.remove_replica(r.replica_id)
            except Exception as e:
                log.warning("failed to remove replica",
                            replica=r.replica_id, error=str(e))

    async def _gather_signals(self, model_name: str, policy: AutoscalePolicy) -> dict[str, float]:
        """Collect ALL autoscaling signals — registry + demand tracker + Redis."""
        replicas = self.registry.get_replicas_for_model(model_name)
        active_replicas = [
            r for r in replicas
            if r.status not in (
                ReplicaStatus.FAILED,
                ReplicaStatus.STOPPING,
                ReplicaStatus.DRAINING,
            )
        ]
        ready_replicas = [r for r in replicas if r.status == ReplicaStatus.READY]
        current_count = len(active_replicas)
        ready_count = len(ready_replicas)

        # ── Registry: instantaneous inflight ──────────────────────
        if ready_replicas:
            total_inflight = sum(r.inflight_requests for r in ready_replicas)
            avg_inflight = total_inflight / ready_count
        else:
            total_inflight = 0
            avg_inflight = 0.0

        # ── DemandTracker: RPS + sustained pressure ───────────────
        rps = 0.0
        demand_avg_inflight = 0.0
        demand_peak_inflight = 0
        sustained = False
        if self.demand_tracker:
            snap = self.demand_tracker.snapshot(model_name, window_s=30)
            rps = snap.rps
            demand_avg_inflight = snap.avg_inflight
            demand_peak_inflight = snap.peak_inflight
            sustained = self.demand_tracker.sustained_above(
                model_name,
                inflight_threshold=policy.target_inflight * max(ready_count, 1),
                min_duration_s=SUSTAINED_PRESSURE_S,
            )

        # ── Redis queue (secondary, for overflow path) ────────────
        queue_depth = await self.queue.queue_depth(model_name)
        oldest_age_ms = await self.queue.oldest_job_age_ms(model_name) or 0.0

        # ── vLLM metrics (from metrics_collector scrapes) ─────────
        ttft_values = []
        cache_usage_values = []
        for r in ready_replicas:
            m = r.metrics_snapshot
            if "vllm:time_to_first_token_seconds_p95" in m:
                ttft_values.append(m["vllm:time_to_first_token_seconds_p95"] * 1000)
            if "vllm:gpu_cache_usage_perc" in m:
                cache_usage_values.append(m["vllm:gpu_cache_usage_perc"])
        ttft_p95 = max(ttft_values) if ttft_values else 0.0
        avg_cache = (
            sum(cache_usage_values) / len(cache_usage_values)
            if cache_usage_values else 0.0
        )

        # ── Track last request time ───────────────────────────────
        scale_state = self._scale_states.setdefault(model_name, ModelScaleState())
        if rps > 0 or avg_inflight > 0 or queue_depth > 0:
            scale_state.last_request_at = time.time()

        starting_count = sum(
            1 for r in active_replicas if r.status == ReplicaStatus.STARTING
        )

        return {
            "current_replicas": float(current_count),
            "ready_replicas": float(ready_count),
            "starting_replicas": float(starting_count),
            # Fast-path signals (DemandTracker)
            "rps": rps,
            "demand_avg_inflight": demand_avg_inflight,
            "demand_peak_inflight": float(demand_peak_inflight),
            "sustained_pressure": float(sustained),
            # Registry instantaneous
            "avg_inflight_per_replica": avg_inflight,
            "total_inflight": float(total_inflight),
            # Redis queue (overflow path)
            "queue_depth": float(queue_depth),
            "oldest_queue_age_ms": oldest_age_ms,
            # vLLM engine metrics
            "ttft_p95_ms": ttft_p95,
            "avg_gpu_cache_usage": avg_cache,
            # Idle tracking
            "time_since_last_request_s": time.time() - scale_state.last_request_at,
        }

    def _decide(
        self,
        model_name: str,
        policy: AutoscalePolicy,
        state: ModelScaleState,
        signals: dict[str, float],
    ) -> tuple[AutoscaleAction, list[str], int]:
        """Make a scaling decision. Returns (action, reasons, target_replicas)."""
        now = time.time()
        current = int(signals["current_replicas"])
        ready = int(signals["ready_replicas"])
        starting = int(signals.get("starting_replicas", 0))
        rps = signals["rps"]
        sustained = bool(signals["sustained_pressure"])
        avg_inflight = signals["avg_inflight_per_replica"]
        queue_depth = signals["queue_depth"]
        queue_age_ms = signals["oldest_queue_age_ms"]
        ttft_p95 = signals["ttft_p95_ms"]
        avg_cache = signals["avg_gpu_cache_usage"]
        idle_s = signals["time_since_last_request_s"]

        # ── SCALE UP ──────────────────────────────────────────────
        scale_up_reasons: list[str] = []

        # Signal 0: Scale from zero — if there are no replicas but we
        # have demand (DemandTracker RPS > 0 or queue_depth > 0), spin
        # up immediately.  The gateway fires demand_tracker.record_request
        # even when replicas=0, so rps will be > 0.
        if current == 0 and (rps > 0 or queue_depth > 0):
            scale_up_reasons.append(
                f"scale_from_zero: rps={rps:.1f}, queue={int(queue_depth)}"
            )

        # Signal 1: SUSTAINED inflight pressure (spike-filtered)
        if sustained and current > 0:
            scale_up_reasons.append(
                f"sustained_inflight>{policy.target_inflight:.0f} "
                f"for >{SUSTAINED_PRESSURE_S}s"
            )

        # Signal 2: RPS saturation — even if inflight looks low (fast
        # completions), high RPS means the replicas are being hammered
        if ready > 0 and rps > 0:
            rps_per_replica = rps / ready
            if rps_per_replica > RPS_PER_REPLICA_CEILING:
                scale_up_reasons.append(
                    f"rps/replica={rps_per_replica:.1f}"
                    f">{RPS_PER_REPLICA_CEILING}"
                )

        # Signal 3: Redis queue overflow (secondary — means fast path
        # couldn't absorb the load and scheduler kicked in)
        if queue_depth > policy.queue_depth_threshold:
            scale_up_reasons.append(
                f"queue_depth={int(queue_depth)}"
                f">{policy.queue_depth_threshold}"
            )
        if queue_age_ms > policy.max_queue_wait_ms:
            scale_up_reasons.append(
                f"queue_wait={int(queue_age_ms)}ms"
                f">{policy.max_queue_wait_ms}ms"
            )

        # Signal 4: vLLM engine saturation (TTFT degradation, KV cache full)
        if ttft_p95 > policy.ttft_target_ms and current > 0:
            scale_up_reasons.append(
                f"ttft_p95={int(ttft_p95)}ms"
                f">{int(policy.ttft_target_ms)}ms"
            )
        if avg_cache > policy.cache_pressure_threshold and current > 0:
            scale_up_reasons.append(
                f"cache={avg_cache:.0%}"
                f">{policy.cache_pressure_threshold:.0%}"
            )

        # Signal 5: instantaneous inflight check (legacy, non-sustained)
        # Only fires if DemandTracker is unavailable (fallback)
        if not sustained and avg_inflight > policy.target_inflight and current > 0:
            scale_up_reasons.append(
                f"inflight/replica={avg_inflight:.1f}"
                f">{policy.target_inflight}"
            )

        if scale_up_reasons and current < policy.max_replicas:
            if (now - state.last_scale_up_at) > policy.upscale_cooldown_s:
                # Compute step size based on pressure level
                step = self._compute_scale_step(
                    rps, ready, avg_inflight, queue_depth, policy
                )
                target = min(current + step, policy.max_replicas)
                # When scaling from zero, min_replicas=0 should still
                # produce target >= 1 (that's the whole point of scale-up).
                target = max(target, policy.min_replicas, 1)
                return AutoscaleAction.SCALE_UP, scale_up_reasons, target

        # Ensure min_replicas (do not stack launches while one is still starting)
        starting = int(signals.get("starting_replicas", 0))
        if current < policy.min_replicas:
            if starting > 0:
                return (
                    AutoscaleAction.HOLD,
                    [f"waiting_for_startup: {starting} replica(s) still STARTING"],
                    current,
                )
            return (
                AutoscaleAction.SCALE_UP,
                [f"below_min_replicas: {current}<{policy.min_replicas}"],
                policy.min_replicas,
            )

        # ── SCALE DOWN ────────────────────────────────────────────
        # Never scale down while replicas are still loading (STARTING).
        # A 120B model can take >2 min to load; idle_timeout would kill it.
        if (
            starting == 0
            and current > policy.min_replicas
            and queue_depth == 0
            and rps < 1.0
            and avg_inflight < policy.idle_inflight_threshold
            and avg_cache < policy.idle_cache_threshold
            and idle_s > policy.idle_timeout_s
            and (now - state.last_scale_down_at) > policy.downscale_cooldown_s
        ):
            target = max(current - 1, policy.min_replicas)
            return (
                AutoscaleAction.SCALE_DOWN,
                [
                    f"idle for {int(idle_s)}s, rps={rps:.1f}, "
                    f"inflight={avg_inflight:.1f}, cache={avg_cache:.0%}"
                ],
                target,
            )

        # ── SCALE TO ZERO ─────────────────────────────────────────
        if (
            starting == 0
            and policy.allow_scale_to_zero
            and current > 0
            and current <= policy.min_replicas
            and queue_depth == 0
            and rps == 0
            and idle_s > policy.scale_to_zero_after_s
        ):
            return (
                AutoscaleAction.SCALE_TO_ZERO,
                [f"idle for {int(idle_s)}s, rps=0, scaling to zero"],
                0,
            )

        # ── HOLD ──────────────────────────────────────────────────
        return AutoscaleAction.HOLD, [], current

    @staticmethod
    def _compute_scale_step(
        rps: float, ready: int, avg_inflight: float,
        queue_depth: float, policy: AutoscalePolicy,
    ) -> int:
        """Decide how many replicas to add in one step.

        Uses the strongest signal to size the step:
        - RPS-based: how many replicas to bring rps/replica under ceiling
        - Inflight-based: how many to bring avg_inflight under target
        - Queue-based: how many to drain the queue within one cooldown
        """
        candidates = [1]

        if ready > 0 and rps > 0:
            desired = math.ceil(rps / RPS_PER_REPLICA_CEILING)
            candidates.append(max(0, desired - ready))

        if ready > 0 and avg_inflight > policy.target_inflight:
            desired = math.ceil(
                (avg_inflight * ready) / policy.target_inflight
            )
            candidates.append(max(0, desired - ready))

        if queue_depth > 0:
            candidates.append(
                math.ceil(queue_depth / max(policy.target_inflight, 1))
            )

        step = max(1, max(candidates))
        return min(step, policy.max_scale_step)
