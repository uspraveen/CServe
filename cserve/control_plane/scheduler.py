"""Scheduler — assigns pending jobs to healthy replicas.

NOTE: The current gateway does direct fast-path routing (pick a replica
immediately when the request arrives).  This scheduler module handles the
*queue-based* path for when:
  - All replicas are saturated (inflight > threshold) and we need to buffer.
  - Priority ordering matters (critical jobs should jump the queue).
  - Fair-share scheduling across tenants is needed.

The scheduler runs as an async background task, polling Redis queues and
assigning jobs to replicas.  For the common case (replicas available),
the gateway short-circuits this entirely — that's the fast path.

This module also implements the replica selection strategies:
  - Least Outstanding Requests (LOR)
  - Prefix-aware routing
  - Session affinity
  - Weighted round-robin
"""

from __future__ import annotations

import asyncio
import hashlib
import time

from cserve.common.logging import get_logger
from cserve.common.metrics import (
    QUEUE_DEPTH,
    QUEUE_TIME_IN_QUEUE,
    SCHEDULER_JOBS_EXPIRED,
    SCHEDULER_JOBS_SCHEDULED,
    SCHEDULER_LOOP_DURATION,
    SCHEDULER_SCHEDULING_DURATION,
)
from cserve.common.models import Job, JobEvent, JobEventRecord, ReplicaState, RoutingStrategy

log = get_logger("scheduler")


class Scheduler:
    """Background scheduler that drains Redis queues into vLLM replicas."""

    def __init__(self, registry, queue, db) -> None:
        self.registry = registry
        self.queue = queue
        self.db = db
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info("scheduler started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("scheduler stopped")

    async def _run_loop(self) -> None:
        while self._running:
            loop_start = time.time()
            depths: dict[str, int] = {}
            try:
                models = self.registry.get_all_model_configs()
                for model_name in models:
                    await self._process_model_queue(model_name)

                # Update queue depth metrics
                depths = await self.queue.queue_depths_all()
                for model_name, depth in depths.items():
                    QUEUE_DEPTH.labels(model=model_name).set(depth)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("scheduler loop error", error=str(e))

            elapsed = time.time() - loop_start
            SCHEDULER_LOOP_DURATION.observe(elapsed)

            # Adaptive sleep: faster when there's work, slower when idle
            sleep_ms = 10 if any(depths.get(m, 0) > 0 for m in self.registry.get_all_model_configs()) else 100
            await asyncio.sleep(sleep_ms / 1000)

    async def _process_model_queue(self, model_name: str) -> None:
        """Process pending jobs for a single model."""
        # Drain leaked priority-set entries from before ack+zrem fix (age > 60s).
        await self.queue.remove_expired_jobs(model_name)

        depth = await self.queue.queue_depth(model_name)
        if depth == 0:
            return

        replicas = self.registry.get_healthy_replicas(model_name)
        if not replicas:
            return

        model_cfg = self.registry.get_model_config(model_name)
        if not model_cfg:
            return

        # ── vLLM admission control ────────────────────────────────────────────
        # vLLM has its own internal scheduler (max_num_seqs / continuous
        # batching).  If we blindly forward more requests than it can hold in
        # KV cache, vLLM queues them internally (memory pressure) or rejects
        # with 429.  We prevent double-queuing by only dispatching to a replica
        # when its inflight count is below the vLLM capacity limit.
        # We use 90% of max_num_seqs as the dispatch ceiling to leave headroom
        # for vLLM's own prefill/decode batching logic.
        engine_max = model_cfg.engine.max_num_seqs
        dispatch_ceiling = max(1, int(engine_max * 0.9))
        eligible_replicas = [r for r in replicas if r.inflight_requests < dispatch_ceiling]
        if not eligible_replicas:
            # All replicas are at vLLM capacity — let vLLM drain, retry soon
            return

        # Determine how many jobs to dequeue this cycle (bounded by eligible capacity)
        total_headroom = sum(dispatch_ceiling - r.inflight_requests for r in eligible_replicas)
        batch_size = min(depth, total_headroom, 32)

        jobs = await self.queue.dequeue_from_stream(model_name, count=batch_size, block_ms=0)
        for job in jobs:
            sched_start = time.time()

            # Check if client disconnected (cancel-on-disconnect)
            if await self.queue.is_cancelled(job.job_id):
                if job.stream_id:
                    await self.queue.ack_job(
                        model_name, job.stream_id, job.job_id,
                    )
                log.info("job cancelled (client disconnected), skipping dispatch",
                         job_id=job.job_id, model=model_name)
                continue

            # Check deadline
            if job.is_expired():
                SCHEDULER_JOBS_EXPIRED.labels(model=model_name).inc()
                await self.db.log_job_event(JobEventRecord(
                    job_id=job.job_id, event=JobEvent.TIMEOUT,
                    metadata={"model": model_name, "waited_ms": (time.time() - job.enqueued_at) * 1000},
                ))
                if job.stream_id:
                    await self.queue.ack_job(
                        model_name, job.stream_id, job.job_id,
                    )
                continue

            # Select replica — only from admission-eligible ones
            replica = select_replica(eligible_replicas, job, model_cfg.routing_strategy)
            if not replica:
                # All replicas at max capacity — will retry next loop
                continue

            # Assign
            self.registry.increment_inflight(replica.replica_id)
            wait_time = time.time() - job.enqueued_at
            QUEUE_TIME_IN_QUEUE.labels(model=model_name).observe(wait_time)
            SCHEDULER_JOBS_SCHEDULED.labels(model=model_name).inc()

            await self.db.log_job_event(JobEventRecord(
                job_id=job.job_id, event=JobEvent.SCHEDULED,
                replica_id=replica.replica_id, node_name=replica.node_name,
                metadata={"model": model_name, "wait_s": wait_time},
            ))

            # Publish assignment to gateway via callback
            await self.queue.publish_callback(job.job_id, {
                "action": "stream" if job.streaming else "forward",
                "replica_endpoint": replica.http_endpoint,
                "replica_id": replica.replica_id,
                "node_name": replica.node_name,
            })

            # ACK stream + drop from priority set so queue_depth matches reality
            if job.stream_id:
                await self.queue.ack_job(
                    model_name, job.stream_id, job.job_id,
                )

            sched_elapsed = time.time() - sched_start
            SCHEDULER_SCHEDULING_DURATION.observe(sched_elapsed)


# ═══════════════════════════════════════════════════════════════════════════
# Replica selection strategies
# ═══════════════════════════════════════════════════════════════════════════

def select_replica(
    replicas: list[ReplicaState],
    job: Job,
    strategy: RoutingStrategy = RoutingStrategy.LEAST_OUTSTANDING,
) -> ReplicaState | None:
    """Select the best replica for a job using the configured strategy."""
    if not replicas:
        return None

    if strategy == RoutingStrategy.LEAST_OUTSTANDING:
        return _select_lor(replicas)
    elif strategy == RoutingStrategy.PREFIX_AWARE:
        return _select_prefix_aware(replicas, job)
    elif strategy == RoutingStrategy.SESSION_AFFINITY:
        return _select_session_affinity(replicas, job)
    elif strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
        return _select_weighted_rr(replicas)
    else:
        return _select_lor(replicas)


def _select_lor(replicas: list[ReplicaState]) -> ReplicaState:
    """Least Outstanding Requests — pick the replica with fewest in-flight."""
    return min(replicas, key=lambda r: r.inflight_requests)


def _select_prefix_aware(replicas: list[ReplicaState], job: Job) -> ReplicaState:
    """Prefix-aware routing — try to route to a replica that has cached
    the prompt prefix.  Falls back to LOR if no good match.

    Uses a simple hash of the first 256 bytes of the payload as the prefix key.
    """
    if not job.payload:
        return _select_lor(replicas)

    # Check load balance first: if load is very imbalanced, use LOR
    loads = [r.inflight_requests for r in replicas]
    if max(loads) - min(loads) > 3:
        return _select_lor(replicas)

    # Hash the prompt prefix to a replica
    prefix = job.payload[:256]
    prefix_hash = int(hashlib.md5(prefix).hexdigest(), 16)
    idx = prefix_hash % len(replicas)

    # Only use the prefix match if that replica isn't heavily loaded
    candidate = replicas[idx]
    avg_load = sum(loads) / len(loads) if loads else 0
    if candidate.inflight_requests <= avg_load + 2:
        return candidate

    return _select_lor(replicas)


def _select_session_affinity(replicas: list[ReplicaState], job: Job) -> ReplicaState:
    """Session affinity — hash the tenant_id to a consistent replica."""
    if not job.tenant_id:
        return _select_lor(replicas)

    session_hash = int(hashlib.md5(job.tenant_id.encode()).hexdigest(), 16)
    idx = session_hash % len(replicas)
    return replicas[idx]


_rr_counter: dict[str, int] = {}


def _select_weighted_rr(replicas: list[ReplicaState]) -> ReplicaState | None:
    """Weighted round-robin (uniform weights for now)."""
    if not replicas:
        return None
    key = replicas[0].model
    counter = _rr_counter.get(key, 0)
    selected = replicas[counter % len(replicas)]
    _rr_counter[key] = counter + 1
    return selected
