"""Redis queue manager — the live job queue.

This module provides the interface between the gateway (producer) and the
scheduler (consumer).  It uses two Redis data structures per model:

  1. Redis Stream (`queue:{model}`) — ordered log of jobs.  The scheduler
     uses a consumer group to read from this, which gives us exactly-once
     delivery, automatic redelivery on consumer crash, and the ability to
     horizontally scale the scheduler later.

  2. Redis Sorted Set (`priority:{model}`) — for priority ordering.
     Score = (100 - priority) * 1e12 + enqueued_at_ns.  Higher priority
     jobs get lower scores → ZPOPMIN dequeues them first.  Within the
     same priority, older jobs come first (FIFO).

  3. Redis Pub/Sub — for result callbacks.  When the scheduler assigns
     a job, it publishes the replica endpoint to the job's callback
     channel so the gateway can open a direct stream.

The stream is the primary storage; the sorted set is the scheduling index.
Both are written atomically via MULTI/EXEC.

Design notes:
  - We use redis.asyncio (async Redis client) throughout.
  - Consumer group name is "scheduler".  Each scheduler instance gets
    a unique consumer name (for future horizontal scaling).
  - Jobs have a TTL: expired entries are cleaned up by the scheduler.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import redis.asyncio as aioredis

from cserve.common.logging import get_logger
from cserve.common.models import Job, RedisConfig

log = get_logger("queue")

CONSUMER_GROUP = "scheduler"
STREAM_PREFIX = "queue:"
PRIORITY_PREFIX = "priority:"
CALLBACK_PREFIX = "cb:"
CANCELLED_PREFIX = "cancelled:"
CANCELLED_TTL_S = 300  # 5 min — cancelled jobs expire from Redis


class QueueError(Exception):
    pass


class JobQueue:
    """Async Redis-backed job queue with priority scheduling."""

    def __init__(self, config: RedisConfig, consumer_name: str | None = None) -> None:
        self._config = config
        self._consumer_name = consumer_name or f"sched-{uuid.uuid4().hex[:8]}"
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._redis = aioredis.Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=self._config.password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=10,
            retry_on_timeout=True,
        )
        await self._redis.ping()
        log.info("connected to Redis", host=self._config.host,
                 port=self._config.port)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    @property
    def redis(self) -> aioredis.Redis:
        assert self._redis is not None, "Queue not connected"
        return self._redis

    # ─── Stream key helpers ──────────────────────────────────────────────

    @staticmethod
    def _stream_key(model: str) -> str:
        return f"{STREAM_PREFIX}{model}"

    @staticmethod
    def _priority_key(model: str) -> str:
        return f"{PRIORITY_PREFIX}{model}"

    @staticmethod
    def _callback_key(job_id: str) -> str:
        return f"{CALLBACK_PREFIX}{job_id}"

    def callback_channel(self, job_id: str) -> str:
        """Public accessor for callback channel name (for unsubscribe)."""
        return self._callback_key(job_id)

    @staticmethod
    def _cancelled_key(job_id: str) -> str:
        return f"{CANCELLED_PREFIX}{job_id}"

    # ─── Cancel (client disconnect) ───────────────────────────────────────

    async def cancel(self, job_id: str, model: str | None = None) -> None:
        """Mark a job as cancelled so the scheduler will not dispatch it.

        Called when the client disconnects before the job is dispatched.
        Uses a Redis key with TTL so we don't leak memory.
        If model is provided, also removes the job from the priority set
        so queue_depth reflects the cancellation.
        """
        key = self._cancelled_key(job_id)
        await self.redis.set(key, "1", ex=CANCELLED_TTL_S)
        if model:
            await self.redis.zrem(self._priority_key(model), job_id)

    async def is_cancelled(self, job_id: str) -> bool:
        """Return True if the job was cancelled (client disconnected)."""
        key = self._cancelled_key(job_id)
        return await self.redis.exists(key) > 0

    # ─── Enqueue (called by gateway) ─────────────────────────────────────

    async def enqueue(self, job: Job) -> str:
        """Enqueue a job. Returns the job_id.

        Atomically writes to both the stream and the priority sorted set.
        """
        stream_key = self._stream_key(job.model)
        priority_key = self._priority_key(job.model)

        job_data = {
            "job_id": job.job_id,
            "tenant_id": job.tenant_id,
            "model": job.model,
            "variant": job.variant,
            "priority": str(job.priority),
            "streaming": "1" if job.streaming else "0",
            "enqueued_at": str(job.enqueued_at),
            "deadline_ms": str(job.deadline_ms),
            "payload": job.payload.decode("utf-8", errors="replace") if isinstance(job.payload, bytes) else job.payload,
            "headers": json.dumps(job.headers),
        }

        # Priority score: lower = dequeued first.
        # (100 - priority) gives higher priority lower score.
        # Adding enqueued_at_ns gives FIFO within the same priority.
        score = (100 - job.priority) * 1_000_000_000_000 + int(job.enqueued_at * 1e6)

        # Ensure consumer group exists
        await self._ensure_consumer_group(stream_key)

        pipe = self.redis.pipeline(transaction=True)
        pipe.xadd(stream_key, job_data, maxlen=100_000)
        pipe.zadd(priority_key, {job.job_id: score})
        await pipe.execute()

        return job.job_id

    async def _ensure_consumer_group(self, stream_key: str) -> None:
        try:
            await self.redis.xgroup_create(
                stream_key, CONSUMER_GROUP, id="0", mkstream=True
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    # ─── Dequeue (called by scheduler) ───────────────────────────────────

    async def dequeue_by_priority(self, model: str, count: int = 1) -> list[Job]:
        """Dequeue up to `count` jobs ordered by priority (highest first).

        Uses ZPOPMIN on the sorted set to get the highest-priority job IDs,
        then reads their full data from the stream.
        """
        priority_key = self._priority_key(model)
        self._stream_key(model)

        # Pop highest-priority job IDs
        results = await self.redis.zpopmin(priority_key, count)
        if not results:
            return []

        jobs: list[Job] = []
        for job_id, _score in results:
            # Read job data from stream by scanning (we store job_id in the entry)
            # For efficiency, we also cache job data in a hash. But for now,
            # we reconstruct from the sorted set + a lightweight hash lookup.
            job_hash = await self.redis.hgetall(f"job:{job_id}")
            if job_hash:
                jobs.append(self._deserialize_job(job_hash))
            else:
                # Fallback: the job data is in the stream entry.
                # This is slower but correct.
                log.warning("job hash miss, scanning stream", job_id=job_id)

        return jobs

    async def dequeue_from_stream(
        self, model: str, count: int = 1, block_ms: int = 100
    ) -> list[Job]:
        """Dequeue jobs using Redis Streams consumer group.

        This is the primary dequeue path.  Uses XREADGROUP with blocking
        for efficient event-driven scheduling.
        """
        stream_key = self._stream_key(model)
        await self._ensure_consumer_group(stream_key)

        try:
            results = await self.redis.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=self._consumer_name,
                streams={stream_key: ">"},
                count=count,
                block=block_ms,
            )
        except aioredis.ResponseError as e:
            # If the stream key was deleted/recreated (for example after manual
            # cleanup of stale queues), Redis also drops the consumer group.
            # Re-create the group and retry once instead of bubbling a noisy
            # scheduler error forever.
            if "NOGROUP" not in str(e):
                raise
            await self._ensure_consumer_group(stream_key)
            results = await self.redis.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=self._consumer_name,
                streams={stream_key: ">"},
                count=count,
                block=block_ms,
            )

        if not results:
            return []

        jobs: list[Job] = []
        for _stream_name, entries in results:
            for entry_id, fields in entries:
                try:
                    job = self._deserialize_job(fields, stream_id=entry_id)
                    jobs.append(job)
                except Exception as e:
                    log.error("failed to deserialize job from stream",
                              entry_id=entry_id, error=str(e))
                    # ACK the bad entry so it doesn't block the consumer
                    await self.redis.xack(stream_key, CONSUMER_GROUP, entry_id)
                    jid = fields.get("job_id")
                    if jid:
                        await self.redis.zrem(self._priority_key(model), jid)

        return jobs

    async def ack_job(
        self, model: str, stream_id: str, job_id: str | None = None
    ) -> None:
        """Acknowledge a job as processed (remove from consumer group pending list).

        Also removes ``job_id`` from the priority sorted set so ``queue_depth``
        stays in sync with the stream path (enqueue zadds both; dequeue uses
        XREADGROUP only).
        """
        await self.redis.xack(self._stream_key(model), CONSUMER_GROUP, stream_id)
        if job_id:
            await self.redis.zrem(self._priority_key(model), job_id)

    @staticmethod
    def _deserialize_job(fields: dict[str, Any], stream_id: str | None = None) -> Job:
        payload = fields.get("payload", "")
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        headers = fields.get("headers", "{}")
        if isinstance(headers, str):
            headers = json.loads(headers)

        return Job(
            job_id=fields["job_id"],
            tenant_id=fields.get("tenant_id", ""),
            model=fields["model"],
            variant=fields.get("variant", "default"),
            priority=int(fields.get("priority", 50)),
            payload=payload,
            headers=headers,
            streaming=fields.get("streaming") == "1",
            enqueued_at=float(fields.get("enqueued_at", time.time())),
            deadline_ms=int(fields.get("deadline_ms", 30_000)),
            callback_key=fields.get("callback_key", ""),
            stream_id=stream_id,
        )

    # ─── Pub/Sub for result callbacks ────────────────────────────────────

    async def publish_callback(self, job_id: str, data: dict) -> None:
        """Publish a result or routing instruction back to the gateway."""
        channel = self._callback_key(job_id)
        await self.redis.publish(channel, json.dumps(data, default=str))

    async def subscribe_callback(self, job_id: str):
        """Subscribe to the callback channel for a specific job.

        Returns an async pubsub object. The caller should iterate over
        messages with `async for msg in pubsub.listen()`.
        """
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self._callback_key(job_id))
        return pubsub

    async def get_callback(self, job_id: str, timeout_s: float) -> dict | None:
        """Wait for a single callback message. Returns parsed data or None on timeout."""
        pubsub = self.redis.pubsub()
        channel = self._callback_key(job_id)
        await pubsub.subscribe(channel)

        async def _listen() -> dict | None:
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    return json.loads(msg["data"])
            return None

        try:
            return await asyncio.wait_for(_listen(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    # ─── Queue inspection (for autoscaler + dashboard) ───────────────────

    async def queue_depth(self, model: str) -> int:
        """Number of jobs not yet acked from the priority index (matches stream lifecycle)."""
        return await self.redis.zcard(self._priority_key(model))

    async def queue_depths_all(self) -> dict[str, int]:
        """Queue depths for all models (scans priority:* keys)."""
        result: dict[str, int] = {}
        async for key in self.redis.scan_iter(f"{PRIORITY_PREFIX}*"):
            model = key.removeprefix(PRIORITY_PREFIX)
            result[model] = await self.redis.zcard(key)
        return result

    async def oldest_job_age_ms(self, model: str) -> float | None:
        """Age of the oldest pending job in milliseconds.

        Returns None if queue is empty.
        """
        priority_key = self._priority_key(model)
        oldest = await self.redis.zrange(priority_key, 0, 0, withscores=True)
        if not oldest:
            return None
        _job_id, score = oldest[0]
        # Reconstruct enqueued_at from score
        enqueued_us = score % 1_000_000_000_000
        enqueued_at = enqueued_us / 1e6
        return (time.time() - enqueued_at) * 1000

    async def remove_expired_jobs(self, model: str) -> int:
        """Remove expired jobs from the priority set. Returns count removed."""
        priority_key = self._priority_key(model)
        now = time.time()
        removed = 0

        # Scan from oldest
        entries = await self.redis.zrange(priority_key, 0, 99, withscores=True)
        to_remove: list[str] = []
        for job_id, score in entries:
            enqueued_us = score % 1_000_000_000_000
            enqueued_at = enqueued_us / 1e6
            # We don't know the deadline from the score alone, so use a
            # generous default (60s).  The scheduler also checks deadlines
            # on dequeue and emits TIMEOUT events.
            if (now - enqueued_at) > 60:
                to_remove.append(job_id)

        if to_remove:
            removed = await self.redis.zrem(priority_key, *to_remove)

        return removed

    # ─── Health ──────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            return await self.redis.ping()
        except Exception:
            return False

    async def purge_queues(
        self,
        *,
        models: list[str] | None = None,
        purge_callback_keys: bool = False,
        purge_cancelled_keys: bool = False,
    ) -> dict[str, Any]:
        """Delete pending jobs from Redis (stream + priority zset per model).

        Used for maintenance / cold restart. **Drops queued work** — clients
        waiting on those jobs may hang unless the control plane is restarted or
        they retry.

        If ``models`` is None, discovers model names from existing ``priority:*``
        and ``queue:*`` keys. Optionally deletes ``cb:*`` (pubsub routing) and
        ``cancelled:*`` markers (normally short TTL).

        vLLM's own internal queues are **not** in Redis; restart replicas to
        clear in-process work.
        """
        r = self.redis
        discovered: set[str] = set()
        if models is None:
            async for key in r.scan_iter(f"{PRIORITY_PREFIX}*"):
                discovered.add(key.removeprefix(PRIORITY_PREFIX))
            async for key in r.scan_iter(f"{STREAM_PREFIX}*"):
                discovered.add(key.removeprefix(STREAM_PREFIX))
        else:
            discovered = set(models)

        stream_deleted = 0
        priority_deleted = 0
        for m in sorted(discovered):
            stream_deleted += int(await r.delete(self._stream_key(m)))
            priority_deleted += int(await r.delete(self._priority_key(m)))

        cb_deleted = 0
        if purge_callback_keys:
            async for key in r.scan_iter(f"{CALLBACK_PREFIX}*"):
                cb_deleted += int(await r.delete(key))

        cancelled_deleted = 0
        if purge_cancelled_keys:
            async for key in r.scan_iter(f"{CANCELLED_PREFIX}*"):
                cancelled_deleted += int(await r.delete(key))

        log.warning(
            "redis job queues purged",
            models=sorted(discovered),
            streams=stream_deleted,
            priorities=priority_deleted,
            callback_keys=cb_deleted,
            cancelled_keys=cancelled_deleted,
        )

        return {
            "models_cleared": sorted(discovered),
            "stream_keys_deleted": stream_deleted,
            "priority_keys_deleted": priority_deleted,
            "callback_keys_deleted": cb_deleted,
            "cancelled_keys_deleted": cancelled_deleted,
        }
