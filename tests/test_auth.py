"""Tests for authentication, API key management, rate limiting, and usage tracking."""

from __future__ import annotations

import tempfile

import pytest

from cserve.common.auth import (
    ApiKey,
    AuthenticatedUser,
    KeyRole,
    generate_api_key,
    hash_key,
)
from cserve.control_plane.db import EventLog


class TestKeyGeneration:
    def test_generate_produces_valid_format(self):
        raw, h, kid = generate_api_key()
        assert raw.startswith("csk_")
        assert len(raw) == 4 + 64  # prefix + 32 bytes hex
        assert len(h) == 64  # SHA-256 hex
        assert len(kid) == 12  # 6 bytes hex

    def test_each_key_is_unique(self):
        keys = {generate_api_key()[0] for _ in range(50)}
        assert len(keys) == 50

    def test_hash_is_deterministic(self):
        raw = "csk_abc123"
        assert hash_key(raw) == hash_key(raw)

    def test_hash_matches_generation(self):
        raw, h, _ = generate_api_key()
        assert hash_key(raw) == h


class TestApiKeyModel:
    def test_defaults(self):
        key = ApiKey(key_id="abc", key_hash="def", user_id="u1")
        assert key.role == KeyRole.USER
        assert key.rate_limit_rpm == 0
        assert key.enabled is True
        assert key.total_requests == 0

    def test_admin_role(self):
        key = ApiKey(key_id="abc", key_hash="def", user_id="u1", role=KeyRole.ADMIN)
        assert key.role == KeyRole.ADMIN


class TestAuthenticatedUser:
    def test_construction(self):
        user = AuthenticatedUser(
            key_id="k1", user_id="u1", role=KeyRole.USER, rate_limit_rpm=60,
        )
        assert user.key_id == "k1"
        assert user.rate_limit_rpm == 60


class TestDbApiKeys:
    @pytest.fixture
    async def db(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            log = EventLog(f.name)
            await log.open()
            yield log
            await log.close()

    async def test_create_and_authenticate(self, db):
        raw, api_key = await db.create_api_key(
            user_id="alice", name="test-key", role="user", rate_limit_rpm=100,
        )
        assert raw.startswith("csk_")
        assert api_key.user_id == "alice"
        assert api_key.name == "test-key"
        assert api_key.rate_limit_rpm == 100

        # Authenticate with the raw key
        result = await db.authenticate_key(raw)
        assert result is not None
        assert result.key_id == api_key.key_id
        assert result.user_id == "alice"

    async def test_invalid_key_returns_none(self, db):
        result = await db.authenticate_key("csk_invalid_key_that_doesnt_exist")
        assert result is None

    async def test_disabled_key_returns_none(self, db):
        raw, api_key = await db.create_api_key(user_id="bob", name="temp")
        await db.revoke_api_key(api_key.key_id)
        result = await db.authenticate_key(raw)
        assert result is None

    async def test_list_keys(self, db):
        await db.create_api_key(user_id="alice", name="key-1")
        await db.create_api_key(user_id="alice", name="key-2")
        await db.create_api_key(user_id="bob", name="key-3")

        all_keys = await db.list_api_keys()
        assert len(all_keys) == 3

        alice_keys = await db.list_api_keys(user_id="alice")
        assert len(alice_keys) == 2
        assert all(k["user_id"] == "alice" for k in alice_keys)

    async def test_list_keys_excludes_hash(self, db):
        await db.create_api_key(user_id="alice", name="key-1")
        keys = await db.list_api_keys()
        assert "key_hash" not in keys[0]

    async def test_update_key(self, db):
        _, api_key = await db.create_api_key(user_id="alice", name="old-name")
        await db.update_api_key(api_key.key_id, name="new-name", rate_limit_rpm=200)
        keys = await db.list_api_keys(user_id="alice")
        assert keys[0]["name"] == "new-name"
        assert keys[0]["rate_limit_rpm"] == 200

    async def test_delete_key(self, db):
        _, api_key = await db.create_api_key(user_id="alice", name="temp")
        deleted = await db.delete_api_key(api_key.key_id)
        assert deleted is True
        keys = await db.list_api_keys()
        assert len(keys) == 0

    async def test_key_count(self, db):
        assert await db.get_key_count() == 0
        await db.create_api_key(user_id="alice", name="k1")
        assert await db.get_key_count() == 1
        await db.create_api_key(user_id="bob", name="k2")
        assert await db.get_key_count() == 2

    async def test_list_users(self, db):
        await db.create_api_key(user_id="alice", name="k1")
        await db.create_api_key(user_id="alice", name="k2")
        await db.create_api_key(user_id="bob", name="k3")

        users = await db.list_users()
        assert len(users) == 2
        user_ids = {u["user_id"] for u in users}
        assert user_ids == {"alice", "bob"}

    async def test_increment_key_requests_bumps_counters(self, db):
        _, api_key = await db.create_api_key(user_id="alice", name="k")
        await db.increment_key_requests(api_key.key_id)
        await db.increment_key_requests(api_key.key_id)

        keys = await db.list_api_keys(user_id="alice")
        assert keys[0]["total_requests"] == 2
        assert keys[0]["last_used_at"] > 0


class TestDbUsageLog:
    @pytest.fixture
    async def db(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            log = EventLog(f.name)
            await log.open()
            yield log
            await log.close()

    async def test_log_and_query_usage(self, db):
        await db.log_usage(
            key_id="k1", user_id="alice", model="gemma-27b",
            job_id="j1", replica_id="r1", node_name="node-1",
            gpu_ids="0,1", status_code=200, latency_s=1.5,
            prompt_tokens=100, completion_tokens=200,
        )
        await db.log_usage(
            key_id="k1", user_id="alice", model="gemma-27b",
            job_id="j2", replica_id="r1", latency_s=0.8,
            prompt_tokens=50, completion_tokens=100,
        )
        await db.log_usage(
            key_id="k2", user_id="bob", model="llama-70b",
            job_id="j3", replica_id="r2", latency_s=2.0,
        )

        # All users
        usage = await db.get_usage_by_user(window_s=3600)
        assert len(usage) >= 2

        # Specific user
        alice_usage = await db.get_usage_by_user(user_id="alice", window_s=3600)
        assert len(alice_usage) == 1
        assert alice_usage[0]["requests"] == 2
        assert alice_usage[0]["prompt_tokens"] == 150
        assert alice_usage[0]["completion_tokens"] == 300

    async def test_usage_timeseries(self, db):
        for i in range(5):
            await db.log_usage(
                key_id="k1", user_id="alice", model="gemma-27b",
                job_id=f"j{i}", latency_s=0.5,
            )

        ts = await db.get_usage_timeseries(
            user_id="alice", bucket_s=3600, window_s=7200,
        )
        assert len(ts) >= 1
        assert ts[0]["requests"] == 5
