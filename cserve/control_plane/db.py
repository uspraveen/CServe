"""SQLite event log — the durable record of everything that happens.

This module provides async (aiosqlite) access to the event log that stores:
  - Job lifecycle events (ENQUEUED → ... → COMPLETED/FAILED)
  - Autoscale decisions (with full metrics snapshots)
  - Health incidents (node failures, GPU danger, zombie kills)

The same database powers:
  1. Crash recovery (re-enqueue SCHEDULED-but-not-COMPLETED jobs)
  2. Dashboard historical views
  3. SLO calculations
  4. Autoscaling audit trail

SQLite is chosen over Postgres because:
  - Zero operational burden (no daemon to manage)
  - Single-writer semantics match our single-brain architecture
  - WAL mode gives us concurrent readers with non-blocking writes
  - Good enough for 10K+ events/second on modern SSDs
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import aiosqlite

from cserve.common.auth import ApiKey, generate_api_key, hash_key
from cserve.common.logging import get_logger
from cserve.common.models import AutoscaleEvent, JobEvent, JobEventRecord

log = get_logger("db")

DEFAULT_DB_PATH = "/var/lib/cserve/events.db"

# ═══════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════

_SCHEMA = """
-- Job lifecycle events
CREATE TABLE IF NOT EXISTS job_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT    NOT NULL,
    event       TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    replica_id  TEXT,
    node_name   TEXT,
    gpu_ids     TEXT,
    metadata    TEXT
);

CREATE INDEX IF NOT EXISTS idx_job_events_job_id    ON job_events(job_id);
CREATE INDEX IF NOT EXISTS idx_job_events_timestamp ON job_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_job_events_event     ON job_events(event);
CREATE INDEX IF NOT EXISTS idx_job_events_model     ON job_events(
    json_extract(metadata, '$.model')
);

-- Autoscaling decisions
CREATE TABLE IF NOT EXISTS autoscale_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    model           TEXT    NOT NULL,
    variant         TEXT    NOT NULL,
    action          TEXT    NOT NULL,
    from_replicas   INTEGER NOT NULL,
    to_replicas     INTEGER NOT NULL,
    reasons         TEXT,
    metrics_snapshot TEXT
);

CREATE INDEX IF NOT EXISTS idx_autoscale_timestamp ON autoscale_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_autoscale_model     ON autoscale_events(model);

-- Health incidents
CREATE TABLE IF NOT EXISTS health_incidents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL    NOT NULL,
    node_name   TEXT,
    replica_id  TEXT,
    incident_type TEXT  NOT NULL,
    details     TEXT,
    resolved    INTEGER NOT NULL DEFAULT 0,
    resolved_at REAL
);

CREATE INDEX IF NOT EXISTS idx_health_timestamp ON health_incidents(timestamp);
CREATE INDEX IF NOT EXISTS idx_health_node      ON health_incidents(node_name);

-- API keys (hashed — raw key never stored)
CREATE TABLE IF NOT EXISTS api_keys (
    key_id          TEXT PRIMARY KEY,
    key_hash        TEXT UNIQUE NOT NULL,
    key_prefix      TEXT NOT NULL DEFAULT '',
    user_id         TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    role            TEXT NOT NULL DEFAULT 'user',
    rate_limit_rpm  INTEGER NOT NULL DEFAULT 0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      REAL NOT NULL,
    last_used_at    REAL NOT NULL DEFAULT 0.0,
    total_requests  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user   ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash   ON api_keys(key_hash);

-- Per-request usage log (who used what, how long, which GPUs)
CREATE TABLE IF NOT EXISTS usage_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    key_id          TEXT    NOT NULL,
    user_id         TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    job_id          TEXT    NOT NULL,
    replica_id      TEXT,
    node_name       TEXT,
    gpu_ids         TEXT,
    status_code     INTEGER NOT NULL DEFAULT 200,
    latency_s       REAL    NOT NULL DEFAULT 0.0,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    streaming       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_key       ON usage_log(key_id);
CREATE INDEX IF NOT EXISTS idx_usage_user      ON usage_log(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_model     ON usage_log(model);

-- Dashboard / API tunables (engine, autoscaling, safety) — overlays YAML on startup
CREATE TABLE IF NOT EXISTS ui_runtime_tuning (
    scope        TEXT NOT NULL,
    model_name   TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    updated_at   REAL NOT NULL,
    source       TEXT NOT NULL DEFAULT 'ui',
    PRIMARY KEY (scope, model_name)
);

CREATE INDEX IF NOT EXISTS idx_ui_runtime_tuning_updated ON ui_runtime_tuning(updated_at);
"""


# ═══════════════════════════════════════════════════════════════════════════
# EventLog
# ═══════════════════════════════════════════════════════════════════════════

class EventLog:
    """Async SQLite event log for CServe."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        # WAL mode: concurrent reads, non-blocking writes
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        log.info("event log opened", path=self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ─── Job events ──────────────────────────────────────────────────────

    async def log_job_event(self, record: JobEventRecord) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO job_events (job_id, event, timestamp, replica_id, node_name, gpu_ids, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.job_id,
                record.event.value,
                record.timestamp,
                record.replica_id,
                record.node_name,
                record.gpu_ids,
                json.dumps(record.metadata, default=str) if record.metadata else None,
            ),
        )
        await self._db.commit()

    async def get_job_events(self, job_id: str) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT job_id, event, timestamp, replica_id, node_name, gpu_ids, metadata "
            "FROM job_events WHERE job_id = ? ORDER BY timestamp",
            (job_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "job_id": r[0], "event": r[1], "timestamp": r[2],
                "replica_id": r[3], "node_name": r[4], "gpu_ids": r[5],
                "metadata": json.loads(r[6]) if r[6] else {},
            }
            for r in rows
        ]

    async def get_recent_job_events(self, limit: int = 100, model: str | None = None) -> list[dict]:
        assert self._db is not None
        if model:
            cursor = await self._db.execute(
                "SELECT job_id, event, timestamp, replica_id, node_name, gpu_ids, metadata "
                "FROM job_events WHERE json_extract(metadata, '$.model') = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (model, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT job_id, event, timestamp, replica_id, node_name, gpu_ids, metadata "
                "FROM job_events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [
            {
                "job_id": r[0], "event": r[1], "timestamp": r[2],
                "replica_id": r[3], "node_name": r[4], "gpu_ids": r[5],
                "metadata": json.loads(r[6]) if r[6] else {},
            }
            for r in rows
        ]

    async def get_incomplete_jobs(self) -> list[dict]:
        """Find jobs that were SCHEDULED but never reached a terminal state.

        Used for crash recovery: re-enqueue these jobs on startup.
        """
        assert self._db is not None
        terminal = tuple(e.value for e in JobEvent if e.is_terminal())
        placeholders = ",".join("?" for _ in terminal)
        cursor = await self._db.execute(
            f"SELECT DISTINCT je.job_id, je.metadata FROM job_events je "
            f"WHERE je.event = 'SCHEDULED' "
            f"AND je.job_id NOT IN ("
            f"  SELECT job_id FROM job_events WHERE event IN ({placeholders})"
            f")",
            terminal,
        )
        rows = await cursor.fetchall()
        return [
            {"job_id": r[0], "metadata": json.loads(r[1]) if r[1] else {}}
            for r in rows
        ]

    # ─── Autoscale events ────────────────────────────────────────────────

    async def log_autoscale_event(self, event: AutoscaleEvent) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO autoscale_events "
            "(timestamp, model, variant, action, from_replicas, to_replicas, reasons, metrics_snapshot) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.timestamp,
                event.model,
                event.variant,
                event.action.value,
                event.from_replicas,
                event.to_replicas,
                json.dumps(event.reasons),
                json.dumps(event.metrics_snapshot, default=str),
            ),
        )
        await self._db.commit()

    async def get_autoscale_events(
        self,
        model: str | None = None,
        limit: int = 100,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict]:
        assert self._db is not None
        conditions: list[str] = []
        params: list = []
        if model:
            conditions.append("model = ?")
            params.append(model)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(until)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        cursor = await self._db.execute(
            f"SELECT timestamp, model, variant, action, from_replicas, to_replicas, "
            f"reasons, metrics_snapshot FROM autoscale_events{where} "
            f"ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        )
        rows = await cursor.fetchall()
        return [
            {
                "timestamp": r[0], "model": r[1], "variant": r[2],
                "action": r[3], "from_replicas": r[4], "to_replicas": r[5],
                "reasons": json.loads(r[6]) if r[6] else [],
                "metrics_snapshot": json.loads(r[7]) if r[7] else {},
            }
            for r in rows
        ]

    # ─── Health incidents ────────────────────────────────────────────────

    async def log_health_incident(
        self,
        incident_type: str,
        node_name: str | None = None,
        replica_id: str | None = None,
        details: str | None = None,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO health_incidents (timestamp, node_name, replica_id, incident_type, details) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), node_name, replica_id, incident_type, details),
        )
        await self._db.commit()

    async def get_recent_health_incidents(self, limit: int = 50) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT timestamp, node_name, replica_id, incident_type, details, resolved "
            "FROM health_incidents ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "timestamp": r[0], "node_name": r[1], "replica_id": r[2],
                "incident_type": r[3], "details": r[4], "resolved": bool(r[5]),
            }
            for r in rows
        ]

    async def get_job_events_for_replica(
        self, replica_id: str, limit: int = 100,
    ) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT job_id, event, timestamp, replica_id, node_name, gpu_ids, metadata "
            "FROM job_events WHERE replica_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (replica_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "job_id": r[0], "event": r[1], "timestamp": r[2],
                "replica_id": r[3], "node_name": r[4], "gpu_ids": r[5],
                "metadata": json.loads(r[6]) if r[6] else {},
            }
            for r in rows
        ]

    async def get_health_incidents_for_replicas(
        self, replica_ids: list[str], limit: int = 100,
    ) -> list[dict]:
        assert self._db is not None
        if not replica_ids:
            return []
        placeholders = ",".join("?" for _ in replica_ids)
        cursor = await self._db.execute(
            f"SELECT timestamp, node_name, replica_id, incident_type, details, resolved "
            f"FROM health_incidents WHERE replica_id IN ({placeholders}) "
            f"ORDER BY timestamp DESC LIMIT ?",
            (*replica_ids, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "timestamp": r[0], "node_name": r[1], "replica_id": r[2],
                "incident_type": r[3], "details": r[4], "resolved": bool(r[5]),
            }
            for r in rows
        ]

    # ─── Aggregation queries for dashboard ───────────────────────────────

    async def get_job_latency_stats(
        self, model: str, window_s: float = 300
    ) -> dict:
        """Compute p50/p95/p99 latency for completed jobs in the last window."""
        assert self._db is not None
        since = time.time() - window_s
        cursor = await self._db.execute(
            "SELECT je_start.timestamp AS start_ts, je_end.timestamp AS end_ts "
            "FROM job_events je_start "
            "JOIN job_events je_end ON je_start.job_id = je_end.job_id "
            "WHERE je_start.event = 'ENQUEUED' "
            "AND je_end.event = 'COMPLETED' "
            "AND json_extract(je_start.metadata, '$.model') = ? "
            "AND je_end.timestamp >= ?",
            (model, since),
        )
        rows = await cursor.fetchall()
        if not rows:
            return {"count": 0, "p50": None, "p95": None, "p99": None}
        latencies = sorted(r[1] - r[0] for r in rows)
        n = len(latencies)
        return {
            "count": n,
            "p50": latencies[int(n * 0.5)],
            "p95": latencies[min(int(n * 0.95), n - 1)],
            "p99": latencies[min(int(n * 0.99), n - 1)],
        }

    # ─── API key management ───────────────────────────────────────────────

    async def create_api_key(
        self, user_id: str, name: str = "", role: str = "user",
        rate_limit_rpm: int = 0,
    ) -> tuple[str, ApiKey]:
        """Create a new API key. Returns (raw_key, ApiKey).

        raw_key is only available at creation time — store it securely.
        """
        assert self._db is not None
        raw_key, key_hash, key_id = generate_api_key()
        key_prefix = raw_key[:12]
        now = time.time()
        await self._db.execute(
            "INSERT INTO api_keys "
            "(key_id, key_hash, key_prefix, user_id, name, role, "
            " rate_limit_rpm, enabled, created_at, last_used_at, total_requests) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 0.0, 0)",
            (key_id, key_hash, key_prefix, user_id, name, role,
             rate_limit_rpm, now),
        )
        await self._db.commit()
        api_key = ApiKey(
            key_id=key_id, key_hash=key_hash, key_prefix=key_prefix,
            user_id=user_id, name=name, role=role,
            rate_limit_rpm=rate_limit_rpm, created_at=now,
        )
        log.info("api key created", key_id=key_id, user_id=user_id, name=name)
        return raw_key, api_key

    async def authenticate_key(self, raw_key: str) -> ApiKey | None:
        """Look up a key by its hash. Returns None if not found or disabled."""
        assert self._db is not None
        h = hash_key(raw_key)
        cursor = await self._db.execute(
            "SELECT key_id, key_hash, key_prefix, user_id, name, role, "
            "       rate_limit_rpm, enabled, created_at, last_used_at, total_requests "
            "FROM api_keys WHERE key_hash = ?",
            (h,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        key = ApiKey(
            key_id=row[0], key_hash=row[1], key_prefix=row[2],
            user_id=row[3], name=row[4], role=row[5],
            rate_limit_rpm=row[6], enabled=bool(row[7]),
            created_at=row[8], last_used_at=row[9], total_requests=row[10],
        )
        if not key.enabled:
            return None
        # last_used_at + total_requests updated by gateway.increment_key_requests()
        return key

    async def increment_key_requests(self, key_id: str) -> None:
        """Increment total_requests for a key. Call on every authenticated request
        (including cache hits) so Keys page matches Usage page."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE api_keys SET last_used_at = ?, total_requests = total_requests + 1 "
            "WHERE key_id = ?",
            (time.time(), key_id),
        )
        await self._db.commit()

    async def list_api_keys(self, user_id: str | None = None) -> list[dict]:
        """List all keys (or keys for a specific user). Never returns hashes."""
        assert self._db is not None
        if user_id:
            cursor = await self._db.execute(
                "SELECT key_id, key_prefix, user_id, name, role, rate_limit_rpm, "
                "       enabled, created_at, last_used_at, total_requests "
                "FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT key_id, key_prefix, user_id, name, role, rate_limit_rpm, "
                "       enabled, created_at, last_used_at, total_requests "
                "FROM api_keys ORDER BY created_at DESC",
            )
        rows = await cursor.fetchall()
        return [
            {
                "key_id": r[0], "key_prefix": r[1], "user_id": r[2],
                "name": r[3], "role": r[4], "rate_limit_rpm": r[5],
                "enabled": bool(r[6]), "created_at": r[7],
                "last_used_at": r[8], "total_requests": r[9],
            }
            for r in rows
        ]

    async def revoke_api_key(self, key_id: str) -> bool:
        """Disable an API key (soft delete)."""
        assert self._db is not None
        cursor = await self._db.execute(
            "UPDATE api_keys SET enabled = 0 WHERE key_id = ?", (key_id,),
        )
        await self._db.commit()
        revoked = cursor.rowcount > 0
        if revoked:
            log.info("api key revoked", key_id=key_id)
        return revoked

    async def delete_api_key(self, key_id: str) -> bool:
        """Permanently delete an API key."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM api_keys WHERE key_id = ?", (key_id,),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def update_api_key(
        self, key_id: str, name: str | None = None,
        rate_limit_rpm: int | None = None, enabled: bool | None = None,
    ) -> bool:
        """Update mutable fields of an API key."""
        assert self._db is not None
        updates: list[str] = []
        params: list = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if rate_limit_rpm is not None:
            updates.append("rate_limit_rpm = ?")
            params.append(rate_limit_rpm)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(int(enabled))
        if not updates:
            return False
        params.append(key_id)
        await self._db.execute(
            f"UPDATE api_keys SET {', '.join(updates)} WHERE key_id = ?",
            params,
        )
        await self._db.commit()
        return True

    async def get_key_count(self) -> int:
        """Return total number of API keys (for bootstrap check)."""
        assert self._db is not None
        cursor = await self._db.execute("SELECT COUNT(*) FROM api_keys")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def list_users(self) -> list[dict]:
        """Get unique users with aggregated stats."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT user_id, COUNT(*) as key_count, "
            "       SUM(total_requests) as total_reqs, "
            "       MAX(last_used_at) as last_active, "
            "       MIN(created_at) as first_key_at "
            "FROM api_keys GROUP BY user_id ORDER BY total_reqs DESC",
        )
        rows = await cursor.fetchall()
        return [
            {
                "user_id": r[0], "key_count": r[1], "total_requests": r[2],
                "last_active": r[3], "first_key_at": r[4],
            }
            for r in rows
        ]

    # ─── Usage logging ────────────────────────────────────────────────────

    async def log_usage(
        self, key_id: str, user_id: str, model: str, job_id: str,
        replica_id: str = "", node_name: str = "", gpu_ids: str = "",
        status_code: int = 200, latency_s: float = 0.0,
        prompt_tokens: int = 0, completion_tokens: int = 0,
        streaming: bool = False,
    ) -> None:
        """Log a single request for usage attribution."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO usage_log "
            "(timestamp, key_id, user_id, model, job_id, replica_id, "
            " node_name, gpu_ids, status_code, latency_s, "
            " prompt_tokens, completion_tokens, streaming) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(), key_id, user_id, model, job_id,
                replica_id, node_name, gpu_ids, status_code, latency_s,
                prompt_tokens, completion_tokens, int(streaming),
            ),
        )
        await self._db.commit()

    async def get_usage_by_user(
        self,
        user_id: str | None = None,
        window_s: float = 86400,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> list[dict]:
        """Per-user usage summary over a sliding window or explicit [since_ts, until_ts]."""
        assert self._db is not None
        if since_ts is not None and until_ts is not None:
            since, until = since_ts, until_ts
        else:
            until = time.time()
            since = until - window_s
        if user_id:
            cursor = await self._db.execute(
                "SELECT user_id, model, "
                "       COUNT(*) as requests, "
                "       SUM(latency_s) as total_gpu_time_s, "
                "       AVG(latency_s) as avg_latency_s, "
                "       SUM(prompt_tokens) as prompt_tokens, "
                "       SUM(completion_tokens) as completion_tokens "
                "FROM usage_log WHERE user_id = ? AND timestamp >= ? AND timestamp <= ? "
                "GROUP BY user_id, model ORDER BY requests DESC",
                (user_id, since, until),
            )
        else:
            cursor = await self._db.execute(
                "SELECT user_id, model, "
                "       COUNT(*) as requests, "
                "       SUM(latency_s) as total_gpu_time_s, "
                "       AVG(latency_s) as avg_latency_s, "
                "       SUM(prompt_tokens) as prompt_tokens, "
                "       SUM(completion_tokens) as completion_tokens "
                "FROM usage_log WHERE timestamp >= ? AND timestamp <= ? "
                "GROUP BY user_id, model ORDER BY requests DESC",
                (since, until),
            )
        rows = await cursor.fetchall()
        return [
            {
                "user_id": r[0], "model": r[1], "requests": r[2],
                "total_gpu_time_s": round(r[3] or 0, 3),
                "avg_latency_s": round(r[4] or 0, 4),
                "prompt_tokens": r[5] or 0,
                "completion_tokens": r[6] or 0,
            }
            for r in rows
        ]

    async def get_usage_timeseries(
        self,
        user_id: str | None = None,
        model: str | None = None,
        bucket_s: int = 3600,
        window_s: float = 86400,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> list[dict]:
        """Time-bucketed request counts for usage charts."""
        assert self._db is not None
        if since_ts is not None and until_ts is not None:
            since, until = since_ts, until_ts
        else:
            until = time.time()
            since = until - window_s
        conditions = ["timestamp >= ?", "timestamp <= ?"]
        params: list = [since, until]
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if model:
            conditions.append("model = ?")
            params.append(model)
        where = " AND ".join(conditions)
        cursor = await self._db.execute(
            f"SELECT CAST(timestamp / {bucket_s} AS INTEGER) * {bucket_s} as bucket, "
            f"       COUNT(*) as requests, "
            f"       SUM(latency_s) as gpu_time_s, "
            f"       SUM(prompt_tokens + completion_tokens) as tokens "
            f"FROM usage_log WHERE {where} "
            f"GROUP BY bucket ORDER BY bucket",
            params,
        )
        rows = await cursor.fetchall()
        return [
            {
                "timestamp": r[0], "requests": r[1],
                "gpu_time_s": round(r[2] or 0, 3),
                "tokens": r[3] or 0,
            }
            for r in rows
        ]

    # ─── UI runtime tuning (SQLite overlay for dashboard edits) ────────────

    async def get_all_ui_runtime_tuning(self) -> list[dict]:
        """Return all rows: scope, model_name, payload_json, updated_at, source."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT scope, model_name, payload_json, updated_at, source "
            "FROM ui_runtime_tuning ORDER BY scope, model_name",
        )
        rows = await cursor.fetchall()
        return [
            {
                "scope": r[0],
                "model_name": r[1] if r[1] is not None else "",
                "payload_json": r[2],
                "updated_at": r[3],
                "source": r[4],
            }
            for r in rows
        ]

    async def upsert_ui_runtime_tuning(
        self,
        scope: str,
        model_name: str,
        payload: dict,
        source: str,
        updated_at: float,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO ui_runtime_tuning (scope, model_name, payload_json, updated_at, source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scope, model_name) DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at,
                source = excluded.source
            """,
            (
                scope,
                model_name,
                json.dumps(payload, default=str),
                updated_at,
                source,
            ),
        )
        await self._db.commit()

    async def replace_all_ui_runtime_tuning(
        self,
        rows: list[tuple[str, str, dict, float, str]],
    ) -> None:
        """Replace table contents (used after sync-from-yaml)."""
        assert self._db is not None
        await self._db.execute("DELETE FROM ui_runtime_tuning")
        for scope, model_name, payload, ts, source in rows:
            await self._db.execute(
                "INSERT INTO ui_runtime_tuning (scope, model_name, payload_json, updated_at, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (scope, model_name, json.dumps(payload, default=str), ts, source),
            )
        await self._db.commit()
