"""Tests for JobQueue.purge_queues."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cserve.common.models import RedisConfig
from cserve.control_plane.queue import (
    CALLBACK_PREFIX,
    CANCELLED_PREFIX,
    PRIORITY_PREFIX,
    STREAM_PREFIX,
    JobQueue,
)


@pytest.mark.asyncio
async def test_purge_queues_deletes_stream_and_priority():
    q = JobQueue(RedisConfig())
    mock_redis = AsyncMock()
    mock_redis.delete = AsyncMock(return_value=1)

    async def scan_iter(match: str):
        if match == f"{PRIORITY_PREFIX}*":
            yield f"{PRIORITY_PREFIX}m1"
        elif match == f"{STREAM_PREFIX}*":
            yield f"{STREAM_PREFIX}m1"

    mock_redis.scan_iter = MagicMock(side_effect=scan_iter)
    q._redis = mock_redis  # noqa: SLF001

    out = await q.purge_queues(
        models=None,
        purge_callback_keys=False,
        purge_cancelled_keys=False,
    )

    assert set(out["models_cleared"]) == {"m1"}
    assert out["stream_keys_deleted"] == 1
    assert out["priority_keys_deleted"] == 1
    assert out["callback_keys_deleted"] == 0
    mock_redis.delete.assert_any_call(f"{STREAM_PREFIX}m1")
    mock_redis.delete.assert_any_call(f"{PRIORITY_PREFIX}m1")


@pytest.mark.asyncio
async def test_purge_queues_specific_models_only():
    q = JobQueue(RedisConfig())
    mock_redis = AsyncMock()
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.scan_iter = MagicMock()
    q._redis = mock_redis  # noqa: SLF001

    await q.purge_queues(
        models=["alpha"],
        purge_callback_keys=False,
        purge_cancelled_keys=False,
    )

    mock_redis.scan_iter.assert_not_called()
    assert mock_redis.delete.await_count == 2
