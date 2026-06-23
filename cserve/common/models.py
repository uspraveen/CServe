"""Core data models for CServe.

Every component imports from here.  These are the *canonical* representations
of nodes, GPUs, replicas, jobs, and configuration.  Changes here ripple
through the entire system, so be deliberate.

Design decisions:
- Pydantic BaseModel for anything that crosses a serialization boundary
  (config, API responses, DB rows).
- Plain enums (str, Enum) so they serialize to readable JSON/YAML strings
  and can be stored directly in SQLite TEXT columns.
- Immutable-by-default: fields are set at creation. Mutable state (inflight
  counts, status) uses explicit setters or a separate mutable wrapper
  (see registry.py).
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Enums — the state machines are defined HERE, not scattered across modules
# ═══════════════════════════════════════════════════════════════════════════

class GpuState(StrEnum):
    FREE = "FREE"
    ALLOCATED = "ALLOCATED"
    RESERVED = "RESERVED"       # being prepared for a new replica (cleanup in progress)
    FAILED = "FAILED"


class ReplicaStatus(StrEnum):
    STARTING = "STARTING"
    READY = "READY"
    DRAINING = "DRAINING"
    STOPPING = "STOPPING"
    FAILED = "FAILED"

    def can_accept_requests(self) -> bool:
        return self == ReplicaStatus.READY

    def is_terminal(self) -> bool:
        return self in (ReplicaStatus.STOPPING, ReplicaStatus.FAILED)


class NodeStatus(StrEnum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"       # some GPUs failed, but node is reachable


class JobEvent(StrEnum):
    ENQUEUED = "ENQUEUED"
    SCHEDULED = "SCHEDULED"
    STARTED = "STARTED"
    FIRST_TOKEN = "FIRST_TOKEN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"

    def is_terminal(self) -> bool:
        return self in (
            JobEvent.COMPLETED, JobEvent.FAILED,
            JobEvent.TIMEOUT, JobEvent.CANCELLED,
        )


class RoutingStrategy(StrEnum):
    LEAST_OUTSTANDING = "lor"
    PREFIX_AWARE = "prefix"
    SESSION_AFFINITY = "session"
    WEIGHTED_ROUND_ROBIN = "weighted_rr"


class AutoscaleAction(StrEnum):
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    SCALE_TO_ZERO = "SCALE_TO_ZERO"
    HOLD = "HOLD"


# ═══════════════════════════════════════════════════════════════════════════
# GPU
# ═══════════════════════════════════════════════════════════════════════════

class GpuInfo(BaseModel):
    """Snapshot of a single physical GPU on a node."""
    index: int
    uuid: str = ""
    name: str = ""              # e.g. "NVIDIA A40"
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    utilization_pct: float = 0.0
    temperature_c: float = 0.0
    state: GpuState = GpuState.FREE
    allocated_replica_id: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Node
# ═══════════════════════════════════════════════════════════════════════════

class NodeConfig(BaseModel):
    """Static node definition from cluster.yaml."""
    name: str
    host: str
    gpu_count: int = 0
    gpu_type: str = ""          # "a40", "l40", etc.
    cuda_devices: str = ""      # "0,1,2,3" — physical device indices to use
    agent_port: int = 50051
    schedulable: bool = True    # false = exclude from placement (e.g. broken driver)
    labels: dict[str, str] = Field(default_factory=dict)


class HeadConfig(BaseModel):
    """Head node configuration from cluster.yaml."""
    name: str
    host: str
    node_ip: str = ""


class NodeState(BaseModel):
    """Live runtime state of a node, maintained by the registry."""
    name: str
    host: str
    gpu_type: str = ""
    status: NodeStatus = NodeStatus.OFFLINE
    agent_endpoint: str = ""    # "host:50051"
    gpus: list[GpuInfo] = Field(default_factory=list)
    last_heartbeat: float = 0.0
    consecutive_failures: int = 0
    labels: dict[str, str] = Field(default_factory=dict)
    schedulable: bool = True

    # ── Launch circuit breaker ────────────────────────────────────────────────
    # After CIRCUIT_OPEN_THRESHOLD consecutive launch failures the circuit
    # opens and the node is excluded from placement until it self-heals.
    #   circuit_open_until == 0.0  → circuit CLOSED (normal)
    #   circuit_open_until  > now  → circuit OPEN   (skip placement)
    #   circuit_open_until <= now  → circuit HALF-OPEN (one probe allowed)
    consecutive_launch_failures: int = 0
    circuit_open_until: float = 0.0   # unix timestamp; 0.0 = closed


# ═══════════════════════════════════════════════════════════════════════════
# Replica
# ═══════════════════════════════════════════════════════════════════════════

class ReplicaState(BaseModel):
    """Live runtime state of a single vLLM replica."""
    replica_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model: str
    variant: str = "default"
    node_name: str
    gpu_ids: list[int]          # physical GPU indices on the node
    tp_size: int
    http_endpoint: str = ""     # "http://node_ip:port"
    port: int = 0
    pid: int = 0
    status: ReplicaStatus = ReplicaStatus.STARTING
    inflight_requests: int = 0
    started_at: float = Field(default_factory=time.time)
    last_health_check: float = 0.0
    last_health_ok: bool = False
    consecutive_health_failures: int = 0
    metrics_snapshot: dict[str, float] = Field(default_factory=dict)
    # Gateway fast-path: skip routing here until this time (after connect failures).
    gateway_route_cooldown_until: float = 0.0

    # Valid state transitions — enforced by registry.py
    VALID_TRANSITIONS: dict[ReplicaStatus, set[ReplicaStatus]] = {
        ReplicaStatus.STARTING: {ReplicaStatus.READY, ReplicaStatus.FAILED},
        ReplicaStatus.READY: {ReplicaStatus.DRAINING, ReplicaStatus.FAILED},
        ReplicaStatus.DRAINING: {ReplicaStatus.STOPPING, ReplicaStatus.FAILED},
        ReplicaStatus.STOPPING: {ReplicaStatus.FAILED},  # removed from registry when done
        ReplicaStatus.FAILED: set(),  # terminal
    }

    def can_transition_to(self, target: ReplicaStatus) -> bool:
        return target in self.VALID_TRANSITIONS.get(self.status, set())


# ═══════════════════════════════════════════════════════════════════════════
# Job
# ═══════════════════════════════════════════════════════════════════════════

class Job(BaseModel):
    """A single inference request tracked through the system."""
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tenant_id: str = ""
    model: str
    variant: str = "default"
    priority: int = 50          # 0=background, 50=normal, 100=critical
    payload: bytes = b""        # original HTTP request body
    headers: dict[str, str] = Field(default_factory=dict)
    streaming: bool = False
    enqueued_at: float = Field(default_factory=time.time)
    deadline_ms: int = 30_000   # max acceptable wait before 408/503
    callback_key: str = ""      # Redis Pub/Sub channel for result notification
    stream_id: str | None = None  # Redis Stream entry ID (for ack when cancelled)

    def is_expired(self) -> bool:
        return (time.time() - self.enqueued_at) * 1000 > self.deadline_ms


class JobEventRecord(BaseModel):
    """A single lifecycle event for a job, stored in SQLite."""
    job_id: str
    event: JobEvent
    timestamp: float = Field(default_factory=time.time)
    replica_id: str | None = None
    node_name: str | None = None
    gpu_ids: str | None = None  # comma-separated
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Model & Engine Configuration
# ═══════════════════════════════════════════════════════════════════════════

class EngineConfig(BaseModel):
    """vLLM engine args for a model."""
    max_model_len: int = 81920
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.70
    dtype: str = "auto"
    trust_remote_code: bool = True
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    disable_custom_all_reduce: bool = False
    reasoning_parser: str | None = None
    runner: str | None = None       # e.g. "pooling" for embeddings
    convert: str | None = None      # e.g. "embed" for embeddings
    kv_cache_dtype: str | None = None  # e.g. "fp8"


class AutoscalePolicy(BaseModel):
    """Per-model autoscaling configuration."""
    min_replicas: int = 1
    max_replicas: int = 1
    allow_scale_to_zero: bool = False

    # Queue overflow protection — return HTTP 429 when this is exceeded.
    # Prevents Redis memory bloat under heavy load; clients should back-off
    # and retry.  Set to 0 to disable the check.
    max_queue_depth: int = 200

    # Scale-up triggers (ANY triggers scale-up)
    queue_depth_threshold: int = 5
    max_queue_wait_ms: int = 2000
    target_inflight: float = 3.0
    ttft_target_ms: float = 1000.0
    cache_pressure_threshold: float = 0.85

    # Scale-up behavior
    upscale_cooldown_s: float = 30.0
    max_scale_step: int = 4

    # Scale-down triggers (ALL must be true)
    idle_inflight_threshold: float = 0.5
    idle_cache_threshold: float = 0.3
    idle_timeout_s: float = 120.0
    downscale_cooldown_s: float = 120.0
    drain_timeout_s: float = 60.0

    # Scale-to-zero
    scale_to_zero_after_s: float = 600.0

    # Max seconds to wait for vLLM /health during startup (node agent + CP grace).
    # Large models (e.g. 120B TP4) may need 15–20+ minutes on first load.
    replica_startup_timeout_s: float = Field(default=600.0, ge=60.0, le=7200.0)


class ModelConfig(BaseModel):
    """Full configuration for a single model (parsed from models.yaml)."""
    name: str                   # key in models.yaml (internal name)
    served_model_name: str
    hf_model: str
    tp: int = 1
    node_type_required: str | None = None
    node_types_allowed: list[str] = Field(default_factory=list)
    routing_strategy: RoutingStrategy = RoutingStrategy.LEAST_OUTSTANDING
    hf_token: str | None = None
    engine: EngineConfig = Field(default_factory=EngineConfig)
    autoscaling: AutoscalePolicy = Field(default_factory=AutoscalePolicy)
    # Lower value = deploy earlier (gemma4 before gpt-oss before gemma3/qwen).
    deploy_priority: int = 50
    # If set, placement is limited to these node names (excludes broken nodes).
    nodes_allowed: list[str] = Field(default_factory=list)
    # Skip cluster-wide VRAM/compute GPU guard for this model (e.g. 120B needs high util).
    gpu_guard_exempt: bool = False


class GlobalConfig(BaseModel):
    """Global settings from models.yaml."""
    env: dict[str, str] = Field(default_factory=dict)
    defaults: EngineConfig = Field(default_factory=EngineConfig)


# ═══════════════════════════════════════════════════════════════════════════
# Cluster Configuration (top-level parsed from cluster.yaml)
# ═══════════════════════════════════════════════════════════════════════════

class RedisConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    password: str | None = None


class SafetyConfig(BaseModel):
    # VRAM fraction (used/total) — triggers the same guard pipeline as before.
    gpu_memory_limit: float = 0.79
    gpu_warn_threshold: float = 0.70
    gpu_danger_threshold: float = 0.79
    gpu_danger_consecutive_checks: int = 2
    # GPU compute (nvidia-smi GPU-Util % / 100) sustained above threshold → same pipeline.
    gpu_compute_sustain_threshold: float = 0.99
    # Long default so vLLM model load (often minutes of pegged GPU-Util) does not
    # trip mitigation; tune down only if you want faster compute-pressure response.
    gpu_compute_sustain_duration_s: float = 900.0
    # GPU memory guard: intelligent mitigation before forced migration
    guard_mitigation_window_s: float = 600.0  # 10 min to self-heal
    guard_check_interval_s: float = 20.0
    guard_consecutive_breaches: int = 2  # confirm before starting mitigation


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8002


class NodeAgentConfig(BaseModel):
    port: int = 50051
    heartbeat_interval_s: int = 10


class SshConfig(BaseModel):
    """SSH credentials and deployment paths for node agent management."""
    username: str = "praveen"
    key_path: str = "~/.ssh/id_ed25519"
    password: str | None = None          # optional — used when key auth is unavailable
    port: int = 22
    timeout_s: float = 30.0
    cserve_src: str = "~/CServe"
    python_path: str = "~/miniconda3/bin/python3"
    pip_path: str = "~/miniconda3/bin/pip"


class ClusterConfig(BaseModel):
    """Top-level configuration parsed from cluster.yaml."""
    head: HeadConfig
    nodes: list[NodeConfig] = Field(default_factory=list)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    node_agent: NodeAgentConfig = Field(default_factory=NodeAgentConfig)
    ssh: SshConfig = Field(default_factory=SshConfig)


# ═══════════════════════════════════════════════════════════════════════════
# Autoscale Event (for logging decisions)
# ═══════════════════════════════════════════════════════════════════════════

class AutoscaleEvent(BaseModel):
    """Logged every time the autoscaler makes a decision (including HOLD)."""
    timestamp: float = Field(default_factory=time.time)
    model: str
    variant: str = "default"
    action: AutoscaleAction
    from_replicas: int
    to_replicas: int
    reasons: list[str] = Field(default_factory=list)
    metrics_snapshot: dict[str, float] = Field(default_factory=dict)
