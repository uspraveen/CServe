"""Rate limiter — per-key token-bucket backed by Redis.

Each API key has a `rate_limit_rpm` (requests per minute).  0 = unlimited.

Implementation: Sliding-window counter using Redis INCR + EXPIRE.
Simpler than a proper token bucket but accurate enough for our use case:
  - Key: `ratelimit:{key_id}:{minute_bucket}`
  - On each request: INCR the counter, EXPIRE to 120s (2 min TTL for safety)
  - If counter > rate_limit_rpm → reject with 429

This adds exactly 1 Redis RTT per request (pipelined INCR+EXPIRE).
If Redis is down, we allow the request (fail-open).
"""

from __future__ import annotations

import time

from cserve.common.logging import get_logger

log = get_logger("rate_limit")


class RateLimiter:
    """Per-key sliding-window rate limiter backed by Redis."""

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    async def check(self, key_id: str, limit_rpm: int) -> tuple[bool, int, int]:
        """Check if a request is allowed.

        Returns:
            (allowed, current_count, limit)
        """
        if limit_rpm <= 0:
            return True, 0, 0

        bucket = int(time.time() // 60)
        redis_key = f"ratelimit:{key_id}:{bucket}"

        try:
            pipe = self._redis.pipeline(transaction=True)
            pipe.incr(redis_key)
            pipe.expire(redis_key, 120)
            results = await pipe.execute()
            current = results[0]
        except Exception as e:
            log.warning("rate limit check failed, allowing request", error=str(e))
            return True, 0, limit_rpm

        if current > limit_rpm:
            return False, current, limit_rpm

        return True, current, limit_rpm
