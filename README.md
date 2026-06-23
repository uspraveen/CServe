# CServe

**LLM Inference Cluster Orchestration — Ray-free, vLLM-native, autoscaling-first.**

CServe is a production-grade orchestration system for serving LLMs and multimodal models on bare-metal GPU clusters. It replaces Ray Serve with purpose-built components that give operators full control over scheduling, autoscaling, GPU lifecycle, and request flow — while using **vLLM** as the inference engine.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Component Deep-Dive](#3-component-deep-dive)
   - [3.1 API Gateway](#31-api-gateway)
   - [3.2 Queue Layer (Redis + SQLite)](#32-queue-layer-redis--sqlite)
   - [3.3 Scheduler](#33-scheduler)
   - [3.4 Autoscaler](#34-autoscaler)
   - [3.5 Cluster Registry](#35-cluster-registry)
   - [3.6 Health Manager](#36-health-manager)
   - [3.7 Node Agent](#37-node-agent)
   - [3.8 Dashboard & Observability](#38-dashboard--observability)
4. [Request Lifecycle](#4-request-lifecycle)
5. [Latency Optimization Strategy](#5-latency-optimization-strategy)
6. [Autoscaling Deep-Dive](#6-autoscaling-deep-dive)
7. [GPU Management & Safety](#7-gpu-management--safety)
8. [Configuration Model](#8-configuration-model)
9. [Wire Protocols](#9-wire-protocols)
10. [Project Layout](#10-project-layout)
11. [Design Principles](#11-design-principles)

---

## 1. Architecture Overview

CServe is split into three planes:

| Plane | Runs On | Purpose |
|-------|---------|---------|
| **Control Plane** | Head node (no GPUs) | API gateway, job queue, scheduler, autoscaler, cluster registry, health manager |
| **Data Plane** | Each GPU worker node | Node agent daemon, vLLM server processes |
| **Observability Plane** | Head node + browser | Dashboard UI, Prometheus metrics, structured event log |

**Key departures from Ray Serve:**
- No distributed actor framework. The head node is the single brain; workers run a thin agent.
- No opaque GCS or Raylet. Cluster state lives in an in-memory registry backed by SQLite.
- No 12-port bootstrap. Workers expose one HTTP port; the head exposes one HTTP port.
- No implicit health checks. GPU health, process health, and queue health are explicit subsystems.

---

## 2. Architecture Diagram

```
                                                                                                  
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │                              CONTROL PLANE  (Head Node)                                      │
  │                                                                                              │
  │  ┌─────────────────────┐      ┌─────────────────────────────────────────────────────────┐    │
  │  │   API Gateway        │      │                    Scheduler                            │    │
  │  │   (FastAPI)          │      │                                                         │    │
  │  │                      │      │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │    │
  │  │  • OpenAI-compat API │      │  │ Job Picker   │  │ Replica      │  │ Backpressure │  │    │
  │  │  • Auth / Rate Limit │      │  │ (priority +  │  │ Selector     │  │ Controller   │  │    │
  │  │  • Request → Job     │      │  │  fair-share) │  │ (LOR/prefix) │  │              │  │    │
  │  │  • Streaming proxy   │      │  └──────┬───────┘  └──────┬───────┘  └──────────────┘  │    │
  │  └──────────┬───────────┘      └─────────┼─────────────────┼────────────────────────────┘    │
  │             │                             │                 │                                  │
  │             ▼                             ▼                 ▼                                  │
  │  ┌─────────────────────┐      ┌─────────────────────────────────────────────────────────┐    │
  │  │   Redis              │      │                  Cluster Registry                       │    │
  │  │                      │      │                                                         │    │
  │  │  • Stream per model  │◄────►│  Nodes → GPUs → Replicas (in-memory + SQLite-backed)   │    │
  │  │  • Sorted set for    │      │  Per-model config, SLAs, autoscaling state              │    │
  │  │    priority queuing  │      │  Replica endpoints, health status, metrics cache        │    │
  │  │  • Pub/Sub for       │      │                                                         │    │
  │  │    real-time events  │      └────────────────────────────┬────────────────────────────┘    │
  │  └─────────────────────┘                                    │                                  │
  │                                                             │                                  │
  │  ┌─────────────────────┐      ┌─────────────────────────────┴────────────────────────────┐   │
  │  │   SQLite DB          │      │                   Autoscaler                             │   │
  │  │                      │      │                                                          │   │
  │  │  • Job event log     │      │  Per-model scaling loop:                                 │   │
  │  │    (full lifecycle)  │      │    • queue_depth + time_in_queue                         │   │
  │  │  • Autoscale events  │      │    • inflight_per_replica (from vLLM metrics)            │   │
  │  │  • Node/GPU history  │      │    • TTFT / ITL / e2e latency percentiles                │   │
  │  │  • SLO reports       │      │    • gpu_cache_usage_perc (from vLLM metrics)            │   │
  │  └─────────────────────┘      │    • gpu_memory_used (from node agent)                   │   │
  │                                │  Decisions: SCALE_UP / SCALE_DOWN / HOLD                  │   │
  │  ┌─────────────────────┐      │  Cooldown windows, burst tolerance, drain-before-kill     │   │
  │  │   Health Manager     │      └──────────────────────────────────────────────────────────┘   │
  │  │                      │                                                                     │
  │  │  • Node reachability │      ┌──────────────────────────────────────────────────────────┐   │
  │  │  • Replica /health   │      │              Metrics Collector & Exporter                 │   │
  │  │  • GPU safety checks │      │                                                          │   │
  │  │  • Zombie detection  │      │  Scrapes vLLM /metrics + node agent GPU stats            │   │
  │  │  • Auto-remediation  │      │  Exports unified Prometheus endpoint on head node        │   │
  │  └─────────────────────┘      │  Feeds autoscaler, health manager, and dashboard          │   │
  │                                └──────────────────────────────────────────────────────────┘   │
  └───────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                  │
                              gRPC (control) + HTTP (vLLM proxy)
                                                  │
       ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
       │                                          │                                          │
       ▼                                          ▼                                          ▼
  ┌──────────────────────┐  ┌──────────────────────────────────┐  ┌──────────────────────────┐
  │  DATA PLANE          │  │  DATA PLANE                      │  │  DATA PLANE              │
  │  cosmos-7 (8× A40)   │  │  cosmos-10 (4× L40)             │  │  cosmos-N ...            │
  │                      │  │                                  │  │                          │
  │  ┌────────────────┐  │  │  ┌────────────────┐             │  │  ┌────────────────┐      │
  │  │  Node Agent     │  │  │  │  Node Agent     │             │  │  │  Node Agent     │      │
  │  │  (gRPC server)  │  │  │  │  (gRPC server)  │             │  │  │  (gRPC server)  │      │
  │  │                │  │  │  │                │             │  │  │                │      │
  │  │  • Launch/kill  │  │  │  │  • Launch/kill  │             │  │  │  • Launch/kill  │      │
  │  │    vLLM procs   │  │  │  │    vLLM procs   │             │  │  │    vLLM procs   │      │
  │  │  • GPU monitor  │  │  │  │  • GPU monitor  │             │  │  │  • GPU monitor  │      │
  │  │  • Process      │  │  │  │  • Process      │             │  │  │  • Process      │      │
  │  │    lifecycle    │  │  │  │    lifecycle    │             │  │  │    lifecycle    │      │
  │  └───────┬────────┘  │  │  └───────┬────────┘             │  │  └───────┬────────┘      │
  │          │            │  │          │                      │  │          │                │
  │    ┌─────┴─────┐      │  │    ┌─────┴─────┐               │  │    ┌─────┴─────┐         │
  │    │           │      │  │    │           │               │  │    │           │         │
  │  ┌─┴──┐     ┌─┴──┐   │  │  ┌─┴──┐     ┌─┴──┐            │  │  ┌─┴──┐     ┌─┴──┐      │
  │  │vLLM│     │vLLM│   │  │  │vLLM│     │vLLM│            │  │  │vLLM│     │vLLM│      │
  │  │:8100│    │:8101│   │  │  │:8100│    │:8101│            │  │  │:8100│    │:8101│      │
  │  │GPU  │    │GPU  │   │  │  │GPU  │    │GPU  │            │  │  │GPU  │    │GPU  │      │
  │  │0,1  │    │2,3,4│   │  │  │0,1  │    │2,3  │            │  │  │0,1  │    │2,3  │      │
  │  │,5   │    │,6,7 │   │  │  │     │    │     │            │  │  │     │    │     │      │
  │  └─────┘    └─────┘   │  │  └─────┘    └─────┘            │  │  └─────┘    └─────┘      │
  └────────────────────────┘  └────────────────────────────────┘  └──────────────────────────┘


  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │                            OBSERVABILITY PLANE                                               │
  │                                                                                              │
  │  ┌───────────────────────────┐  ┌────────────────────┐  ┌─────────────────────────────────┐ │
  │  │  Dashboard UI (React)     │  │  Prometheus         │  │  Structured Event Log           │ │
  │  │                           │  │                     │  │  (SQLite)                       │ │
  │  │  • Cluster topology       │  │  • vLLM metrics     │  │                                 │ │
  │  │  • GPU heatmap            │  │    (per replica)    │  │  • Job lifecycle events         │ │
  │  │  • Request flow traces    │  │  • Node agent GPU   │  │  • Autoscale decisions          │ │
  │  │  • Autoscale timeline     │  │    stats            │  │  • Health incidents             │ │
  │  │  • Queue depth + latency  │  │  • Gateway QPS +    │  │  • Node/replica state changes   │ │
  │  │  • Model detail pages     │  │    latency          │  │                                 │ │
  │  │  • Live WebSocket feed    │  │  • Scheduler stats  │  │  Queryable via dashboard API    │ │
  │  └───────────────────────────┘  └────────────────────┘  └─────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Deep-Dive

### 3.1 API Gateway

The public-facing HTTP server. Clients talk to this and nothing else.

**Dual routing architecture — fast path vs queue path:**

1. **Fast path** (default, no Redis): The gateway picks the best replica directly using Least Outstanding Requests (LOR) and proxies the request to vLLM immediately. This adds < 1ms of overhead. Every fast-path request is recorded in the in-process `DemandTracker` so the autoscaler can see real-time load.

2. **Queue path** (overflow): When all replicas are saturated (inflight >= 90% of vLLM's `max_num_seqs`), requests fall through to the Redis queue. The scheduler drains the queue and assigns jobs to replicas as capacity opens.

3. **Backpressure (429)**: Each model has a `max_queue_depth` cap (default 200). When the queue hits this limit, the gateway returns HTTP 429 with `Retry-After` and `X-CServe-Queue-Depth` headers. This prevents Redis memory bloat under extreme load.

**Authentication & rate limiting:**

- Every request must include `Authorization: Bearer csk_...`
- API keys are hashed (SHA-256) and stored in SQLite — raw keys are only shown once at creation time.
- Per-key rate limiting via Redis sliding-window counters (requests/minute).
- Per-request usage logging: user, model, latency, token counts, GPU attribution.
- Admin API (`/admin/keys`, `/admin/users`, `/admin/usage`) for key CRUD and usage analytics.
- Bootstrap: first startup auto-creates an admin key and prints it to console.

**Responsibilities:**
- Expose an **OpenAI-compatible HTTP API** (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`).
- Authenticate every request via API key (Bearer token).
- Enforce per-key rate limits (token-bucket in Redis, fail-open if Redis is down).
- For **non-streaming** requests: select replica, forward, return full response.
- For **streaming** requests: select replica, **open a direct HTTP/SSE stream** to the vLLM replica and proxy tokens back to the client in real-time. The gateway does NOT buffer the full response.
- Log per-request usage (user, model, latency, tokens) to SQLite for attribution/billing.
- Feed the `DemandTracker` on every request (RPS + inflight pressure for autoscaling).
- Return structured errors with `Retry-After` headers on overload (429) or cold-start (503).

**Technology:** FastAPI (async, uvicorn), with `httpx.AsyncClient` for replica proxying.

**Key metrics exported:**
- `cserve_gateway_requests_total` (counter, labels: model, status_code, tenant)
- `cserve_gateway_request_duration_seconds` (histogram, labels: model, streaming)
- `cserve_gateway_active_streams` (gauge, labels: model)
- `cserve_gateway_queue_submit_duration_seconds` (histogram)

---

### 3.2 Queue Layer (Redis + SQLite)

Every request becomes a **Job** that flows through a two-tier queue.

#### Redis (Live Queue) — speed

- **One Redis Stream per model** (e.g. `queue:gemma3-27b`, `queue:gpt-oss-120b`).
- Each stream entry contains:
  ```
  job_id:         UUID
  tenant_id:      string
  model:          string
  variant:        string (default: "default")
  priority:       int (0=background, 50=normal, 100=critical)
  payload:        bytes (the original HTTP request body)
  headers:        bytes (original HTTP headers, serialized)
  enqueued_at:    float (unix timestamp, nanosecond precision)
  deadline_ms:    int (max acceptable wait time before 408/503)
  streaming:      bool
  callback_key:   string (Redis Pub/Sub channel for result notification)
  ```
- **Priority ordering** via Redis Sorted Sets: a parallel sorted set `priority:gemma3-27b` with `score = (100 - priority) * 1e12 + enqueued_at` so higher-priority jobs are dequeued first, with FIFO within the same priority.
- **Consumer groups** for the scheduler (allows future horizontal scaling of the scheduler itself).
- **Pub/Sub channels** for real-time result delivery back to the gateway.
- **TTL on completed entries**: auto-expire after 5 minutes.

#### SQLite (Durable Event Log) — durability and analytics

Every job state transition is appended to a `job_events` table:

```sql
CREATE TABLE job_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    event       TEXT NOT NULL,  -- ENQUEUED, SCHEDULED, STARTED, FIRST_TOKEN,
                                -- COMPLETED, FAILED, TIMEOUT, CANCELLED
    timestamp   REAL NOT NULL,  -- unix epoch with microseconds
    replica_id  TEXT,
    node_name   TEXT,
    gpu_ids     TEXT,           -- comma-separated GPU indices
    metadata    TEXT            -- JSON blob (latency, tokens, error message, etc.)
);
CREATE INDEX idx_job_events_job_id ON job_events(job_id);
CREATE INDEX idx_job_events_timestamp ON job_events(timestamp);
CREATE INDEX idx_job_events_event ON job_events(event);
```

Additional tables for autoscaling events, node state changes, and health incidents.

**Why both?**
- Redis: sub-millisecond enqueue/dequeue. The scheduler polls Redis, not SQLite.
- SQLite: crash recovery (re-enqueue `SCHEDULED` but not `COMPLETED` jobs), dashboard history, SLO calculations.

---

### 3.3 Scheduler

The scheduler handles the **queue path** only — requests that overflow when the gateway's fast path finds all replicas saturated. Under normal load, the scheduler is idle because the gateway routes directly via LOR.

**Scheduling algorithm (runs every 50ms):**

```
for each model with pending jobs in Redis:
  replicas = registry.get_healthy_replicas(model)
  
  # vLLM admission control — only dispatch to replicas with headroom
  dispatch_ceiling = max_num_seqs * 0.9
  eligible = [r for r in replicas if r.inflight < dispatch_ceiling]
  
  if not eligible:
    # All replicas at vLLM capacity — wait for drain, retry next cycle
    continue
  
  total_headroom = sum(ceiling - r.inflight for r in eligible)
  batch_size = min(queue_depth, total_headroom, 32)
  
  jobs = dequeue_batch(model, batch_size)
  
  for job in jobs:
    replica = min(eligible, key=lambda r: r.inflight)  # LOR
    registry.increment_inflight(replica.id)
    forward_to_vllm(job, replica)
```

This prevents double-queuing (Redis queue → vLLM internal queue), which would cause KV cache memory pressure.

#### Replica Selection Strategies

The scheduler supports pluggable replica selection with these built-in strategies:

1. **Least Outstanding Requests (LOR)** — default. Pick the replica with fewest in-flight requests. Simple, effective, avoids overloading hot replicas.

2. **Prefix-Aware Routing** — for workloads with shared system prompts. Maintain a per-replica prefix hash. Route jobs whose prompt prefix matches a replica's cached prefix, gaining **up to 164x improvement in TTFT** (per vLLM router benchmarks). Fall back to LOR when no prefix match or when load is imbalanced.

3. **Session Affinity** — for multi-turn conversations. Hash the `session_id` to a consistent replica, maximizing KV cache reuse across turns.

4. **Weighted Round-Robin** — for heterogeneous GPU types (e.g. A40 vs L40 with different throughput).

Strategy is configurable per model in `models.yaml`.

#### Fair-Share Scheduling

To prevent a single tenant from monopolizing capacity:
- Maintain per-tenant request counters in Redis.
- When dequeuing, interleave across tenants using deficit round-robin.
- Configurable per-tenant concurrency limits.

---

### 3.4 Autoscaler

An independent loop (runs every 5 seconds) that makes per-model scaling decisions.

#### The DemandTracker Problem & Solution

The fast path bypasses Redis entirely — requests go from the gateway straight to vLLM. That means `queue_depth` is often zero even under heavy load. A naive autoscaler that only watches queue signals would never scale up.

**Solution: `DemandTracker`** (`cserve/common/demand.py`) — an in-process, O(1) sliding-window tracker. The gateway calls `record_request(model, inflight)` on every single request (< 1μs overhead). The autoscaler reads a 30-second snapshot every cycle.

**Spike filtering:** A short burst (< 10 seconds) does NOT trigger scale-up. Only sustained pressure above the threshold for 10+ consecutive seconds fires. This prevents wasting GPUs on transient spikes.

#### Input Signals (Priority Order)

| # | Signal | Source | What it tells us |
|---|--------|--------|------------------|
| 1 | `sustained_pressure` | DemandTracker | Inflight above threshold for 10+ consecutive seconds |
| 2 | `rps / replica > 15` | DemandTracker | High throughput even if inflight looks low (fast completions) |
| 3 | `queue_depth > threshold` | Redis | Overflow path — fast path couldn't absorb load |
| 4 | `queue_wait > max_ms` | Redis | Jobs waiting too long in overflow queue |
| 5 | `ttft_p95 > target` | vLLM metrics | Engine saturation, KV cache contention |
| 6 | `cache_usage > threshold` | vLLM metrics | KV cache filling up, latency will degrade |
| 7 | `inflight/replica > target` | Registry | Fallback instantaneous check |

#### Decision Logic

**SCALE UP** — ANY signal triggers (subject to cooldown):
- `sustained_pressure` for > 10 seconds (spike-filtered)
- `rps_per_replica > 15` (catches fast-completion saturation invisible to inflight)
- `queue_depth > threshold` (overflow path activated)
- `ttft_p95 > target` or `cache_usage > threshold` (vLLM engine saturation)
- Always capped at `max_replicas`. Step size computed from strongest signal.

**SCALE DOWN** — ALL conditions must be true:
- `current > min_replicas`
- `queue_depth == 0`
- `rps < 1.0` (prevents scaling down a model still serving fast requests)
- `avg_inflight < idle_inflight_threshold`
- `avg_cache < idle_cache_threshold`
- `idle_timeout_s` seconds since last request (default 120s — filters transient idle)
- `downscale_cooldown_s` since last scale-down (default 120s)
- Never goes below `min_replicas`.

**SCALE TO ZERO** — only if `allow_scale_to_zero: true`:
- `rps == 0` and `queue_depth == 0`
- Idle for `scale_to_zero_after_s` (default 600s)

**HOLD** — no action needed.

#### Scale-Up Step Sizing

Uses the strongest signal to size the step:
- **RPS-based:** `ceil(rps / 15) - current_ready` (how many replicas to bring rps/replica under 15)
- **Inflight-based:** `ceil(avg_inflight * ready / target_inflight) - ready`
- **Queue-based:** `ceil(queue_depth / target_inflight)`
- Step is capped at `max_scale_step` (default 4).

#### Drain Before Kill

When scaling down:
1. Mark replica as `DRAINING` in the registry.
2. Scheduler stops assigning new jobs to it.
3. Wait for in-flight requests to complete (up to `drain_timeout_s`).
4. Send `STOP` command to node agent.
5. Node agent terminates vLLM process and frees GPUs.

---

### 3.5 Cluster Registry

The single source of truth for cluster state. In-memory for speed, persisted to SQLite for recovery.

#### Data Model

```
Registry
├── Nodes
│   ├── node_name: str
│   ├── host: str
│   ├── gpu_type: str ("a40", "l40")
│   ├── total_gpus: int
│   ├── gpu_states: list[GpuState]  # per-GPU: FREE, ALLOCATED, FAILED
│   ├── agent_endpoint: str         # "host:50051" (gRPC)
│   ├── status: ONLINE | OFFLINE | DEGRADED
│   ├── last_heartbeat: float
│   └── labels: dict                # arbitrary key-value for placement rules
│
├── Replicas
│   ├── replica_id: str (UUID)
│   ├── model: str
│   ├── variant: str
│   ├── node_name: str
│   ├── gpu_ids: list[int]          # physical GPU indices on that node
│   ├── tp_size: int
│   ├── http_endpoint: str          # "http://node_ip:port"
│   ├── status: STARTING | READY | DRAINING | STOPPING | FAILED
│   ├── inflight_requests: int      # updated by scheduler
│   ├── metrics_snapshot: dict      # last scraped vLLM metrics
│   ├── started_at: float
│   └── last_health_check: float
│
└── Models
    ├── model_name: str
    ├── served_model_name: str
    ├── hf_model: str
    ├── tp_size: int
    ├── node_type_required: str
    ├── node_types_allowed: list[str]
    ├── engine_config: dict          # vLLM engine args
    ├── autoscale_policy: AutoscalePolicy
    ├── routing_strategy: str        # "lor", "prefix", "session", "weighted_rr"
    └── current_replicas: list[replica_id]
```

#### State Transitions

Replicas follow a strict state machine:
```
                   ┌──────────────┐
          launch   │              │  vLLM /health returns 200
    ───────────►   │  STARTING    ├──────────────────────────────►  READY
                   │              │                                    │
                   └──────┬───────┘                                   │
                          │ timeout / crash                           │
                          ▼                                           │
                   ┌──────────────┐    autoscaler drain          ┌────┴─────┐
                   │   FAILED     │◄─────────────────────────────│ DRAINING │
                   └──────────────┘    in-flight done             └────┬─────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  STOPPING    │
                                                               └──────┬───────┘
                                                                      │ process killed
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  (removed)   │
                                                               └──────────────┘
```

---

### 3.6 Health Manager

Runs as a background task on the head node, checking three layers of health.

#### Layer 1: Node Health (every 15s)
- gRPC `Ping()` to each node agent.
- If 3 consecutive failures → mark node `OFFLINE`, re-place its replicas elsewhere.

#### Layer 2: Replica Health (every 10s)
- HTTP `GET /health` to each vLLM replica endpoint (via node agent proxy or direct).
- Cross-check with vLLM `vllm:num_requests_running` (if it's stuck at a high number for too long, the replica may be deadlocked).
- If replica health fails:
  - Mark `FAILED`, scheduler stops routing.
  - Tell node agent to kill and restart the process.
  - If restart fails twice → mark GPU as potentially bad, alert operator.

#### Layer 3: GPU Memory Guard (configurable interval, default 20s)
- **Hard limit**: `gpu_memory_limit` (default **79%**) — the maximum acceptable GPU memory utilization.
- **Graduated response** (see [Section 7](#7-gpu-management--safety) for the full state machine):
  1. **WARNED**: Memory exceeds limit. Guard waits for confirmation.
  2. **MITIGATING**: Confirmed breach. Replica is paused (DRAINING) to let KV-cache drain naturally.
  3. **Self-heal**: If memory drops during mitigation, replica is automatically resumed.
  4. **MIGRATING**: Mitigation window expired (~10 min). Replica is killed and relaunched on different GPUs.
- Zombie detection: node agent scans `nvidia-smi` for GPU-holding processes that are NOT children of a managed vLLM server. Kill them.

#### Auto-Remediation Actions
| Condition | Action |
|-----------|--------|
| Replica /health fails | Kill and restart on same node/GPUs |
| Restart fails twice | Mark GPUs as suspect, try different GPUs or node |
| Node unreachable | Drain replicas, attempt restart via SSH fallback |
| GPU memory > 79% (sustained) | Pause → wait 10 min → live-migrate to new GPUs |
| Orphaned GPU process | Kill via node agent |

---

### 3.7 Node Agent

A lightweight daemon running on every GPU worker node. This is the only CServe process on workers.

**Interface:** Dual transport — switchable at startup via `--transport`:

- `--transport http` (default): JSON-over-HTTP via FastAPI/uvicorn. Simple, debuggable.
- `--transport grpc`: Binary gRPC via compiled protobuf stubs (`cserve/generated/`). 2-3x faster serialization, HTTP/2 multiplexing.

Both transports implement the same RPC contract defined in `proto/node_agent.proto`. The control plane has matching clients: `NodeAgentClient` (HTTP) and `GrpcNodeAgentClient` (gRPC).

**Endpoints** (mirror the protobuf RPCs):

**`LaunchReplica` flow:**

1. Receive: model config, GPU IDs, vLLM engine args, port assignment.
2. Pre-flight checks:
   - Verify requested GPUs are not in use (via `nvidia-smi`).
   - If GPUs are dirty, run cleanup (SIGTERM → wait → SIGKILL → verify via `nvidia-smi`).
   - Check free GPU memory meets the `gpu_memory_utilization` requirement.
3. Build vLLM command:
   ```
   python -m vllm.entrypoints.openai.api_server \
     --model <hf_model> \
     --served-model-name <name> \
     --host 0.0.0.0 \
     --port <assigned_port> \
     --tensor-parallel-size <tp> \
     --gpu-memory-utilization <util> \
     --max-model-len <len> \
     --trust-remote-code \
     ...
   ```
4. Set `CUDA_VISIBLE_DEVICES`, start subprocess in a new process group.
5. Wait for `/health` to return 200 (poll every 2s, timeout 15 minutes for large models).
6. Report back to control plane: `READY`, endpoint URL, PID.

**Process supervision:**
- Node agent monitors each vLLM subprocess. If it crashes (non-zero exit), report `FAILED` to control plane immediately via gRPC callback or next heartbeat.
- Logs are written to `/var/log/cserve/replicas/<model>__<replica_id>.log`.

**Metrics reporting (every 10s):**
- Scrape `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv`.
- Scrape each vLLM replica's `/metrics` endpoint (Prometheus format).
- Bundle into `NodeStatusResponse` and send to control plane.

---

### 3.8 Dashboard & Observability

#### Dashboard Backend (part of control plane)

- FastAPI routes under `/dashboard/api/...` (read-only REST endpoints).
- WebSocket endpoint `/dashboard/ws` for real-time push of full cluster snapshots at 1Hz:
  - Nodes, GPUs, replicas, models, queue depths, autoscale events, health incidents, recent jobs.

#### Dashboard Frontend (React + TypeScript + Tailwind CSS)

Production dashboard in `dashboard-ui/`. Built with Vite, outputs to `cserve/dashboard/static/dist/`. A legacy single-file HTML version is also available at `cserve/dashboard/static/index.html`.

**Page: Topology** (`/dashboard/`)
- KPI bar: nodes, GPUs total/free, replicas, inflight, queue depth
- Per-model queue cards with depth progress bars (green → amber → red)
- Node cards: GPU grid (color-coded memory utilization chips), replica list with status badges and inflight counts
- Live request feed: job ID, event type, model, timestamp

**Page: Autoscaling** (`/dashboard/autoscaling`)
- Summary KPIs: scale-up/down/zero counts, active models
- Per-model scale cards: replica count, recent event timeline with action badges
- Full scale event log table: time, model, action, from→to, triggering reasons

**Page: Usage** (`/dashboard/usage`)
- Time-window selector (1h / 6h / 24h / 7d)
- KPI bar: total requests, GPU time, tokens, active users
- Per-user per-model breakdown table: requests, GPU time, avg latency, prompt/completion tokens

**Page: Users & Keys** (`/dashboard/users`)
- User table: user ID, key count, total requests, last active
- API key table: prefix, user, name, role badge, rate limit, request count, status, revoke button
- Create key form: user ID, name, role (user/admin), rate limit
- New key reveal panel (shown once, never again)

---

## 4. Request Lifecycle

### Fast Path (default — no Redis, no Scheduler)

```
 Client          Gateway          DemandTracker     Registry              vLLM
   │                │                  │                │                   │
   │  POST /v1/...  │                  │                │                   │
   ├───────────────►│                  │                │                   │
   │                │  validate model  │                │                   │
   │                │  check 429 cap   │                │                   │
   │                │                  │                │                   │
   │                │  get_healthy_replicas(model)      │                   │
   │                ├─────────────────────────────────►│                   │
   │                │                  │                │  [R1, R2, R3]     │
   │                │◄─────────────────────────────────┤                   │
   │                │                  │                │                   │
   │                │  LOR: pick min(inflight) → R1    │                   │
   │                │  increment_inflight(R1)          │                   │
   │                │  record_request(model, inflight)  │                   │
   │                ├──────────────►│  │                │                   │
   │                │               │[1μs in-process]  │                   │
   │                │                  │                │                   │
   │                │  open HTTP/SSE stream to R1       │                   │
   │                ├──────────────────────────────────────────────────────►│
   │                │                  │                │                   │
   │  SSE: tokens   │                  │                │       inference   │
   │◄───────────────┤                  │                │                   │
   │  SSE: [DONE]   │                  │                │                   │
   │◄───────────────┤                  │                │                   │
   │                │  decrement_inflight(R1)           │                   │
   │                │  SQLite: COMPLETED                │                   │
```

### Queue Path (overflow — replicas saturated)

When all replicas are at 90% of vLLM's `max_num_seqs`, the gateway returns 503. The Redis queue + Scheduler path handles buffered requests.

**Latency breakdown** (warm replica, fast path):
- Gateway overhead: < 1ms (validate, LOR select, DemandTracker record)
- Gateway → vLLM network hop: < 2ms (same datacenter)
- vLLM prefill: model-dependent (100ms–2s depending on prompt length and model size)
- **Total overhead added by CServe: < 3ms**

---

## 5. Latency Optimization Strategy

### 5.1 Minimize CServe Overhead

| Optimization | Technique |
|-------------|-----------|
| Fast path | Gateway routes directly to vLLM via LOR — **bypasses Redis entirely**. < 1ms overhead. |
| Queue fallback | Redis Streams only activated when all replicas are saturated. Overflow, not default. |
| Proxy overhead | For streaming: gateway opens a **direct** `httpx` stream to the vLLM replica. No intermediate buffering. Token bytes flow straight through. |
| Demand tracking | `DemandTracker` records per-request signals in < 1μs (in-process dict write, no locks on hot path reads). |
| Connection reuse | Gateway maintains a pool of persistent HTTP connections to each vLLM replica (`httpx.AsyncClient` with keep-alive). No TCP handshake per request. |
| Admission control | Scheduler only dispatches to replicas below 90% of `max_num_seqs`, preventing vLLM internal queuing and memory pressure. |

### 5.2 Maximize vLLM Performance

These are vLLM-level optimizations that CServe enables and manages:

| Optimization | How CServe enables it |
|-------------|----------------------|
| **Prefix caching** | Prefix-aware routing sends requests with similar prompts to the same replica, maximizing KV cache hits. Up to 164x TTFT improvement. |
| **Session affinity** | Consistent hashing on `session_id` keeps multi-turn conversations on the same replica. KV cache from prior turns is reused. |
| **Right-sized TP** | Per-model tensor parallelism configured in YAML. CServe places replicas on the correct number of GPUs. |
| **GPU memory tuning** | Per-model `gpu_memory_utilization` prevents OOM while maximizing cache space. |
| **Continuous batching** | vLLM handles this natively. CServe's job is to keep replicas fed with requests at the right rate. |
| **KV cache monitoring** | Autoscaler watches `vllm:gpu_cache_usage_perc`. When cache fills up, it scales out before latency degrades. |

### 5.3 Future: Disaggregated Prefill/Decode

vLLM supports separating prefill (compute-heavy) and decode (memory-bound) into different instances. CServe's architecture is ready for this:
- The scheduler can route prefill jobs to prefill-optimized replicas and decode jobs to decode-optimized replicas.
- The node agent can launch vLLM in `--kv-connector` mode with NIXL for high-performance KV cache transfer between nodes.
- This is an optional advanced configuration, not a requirement for MVP.

---

## 6. Autoscaling Deep-Dive

### Policy Configuration (per model in `models.yaml`)

```yaml
autoscaling:
  min_replicas: 1                    # NEVER scale below this (hard floor)
  max_replicas: 8                    # NEVER scale above this (hard ceiling)
  allow_scale_to_zero: false         # if true, can kill last replica after long idle
  
  # Queue overflow protection (gateway returns 429 when exceeded)
  max_queue_depth: 200               # prevents Redis memory bloat; 0 disables

  # Scale-up triggers (ANY of these triggers scale-up)
  queue_depth_threshold: 5           # pending jobs in Redis overflow queue
  max_queue_wait_ms: 2000            # oldest job waiting longer than this
  target_inflight: 3                 # avg in-flight requests per replica
  ttft_target_ms: 1000               # p95 TTFT target
  cache_pressure_threshold: 0.85     # avg gpu_cache_usage_perc
  
  # Scale-up behavior
  upscale_cooldown_s: 30             # minimum time between scale-ups
  max_scale_step: 4                  # max replicas to add at once
  
  # Scale-down triggers (ALL must be true simultaneously)
  idle_inflight_threshold: 0.5       # avg inflight/replica below this
  idle_cache_threshold: 0.3          # avg cache usage below this
  idle_timeout_s: 120                # no requests for this long (filters transient idle)
  downscale_cooldown_s: 120          # minimum time between scale-downs
  drain_timeout_s: 60                # wait this long for in-flight to finish before kill
  
  # Scale-to-zero (requires allow_scale_to_zero: true)
  scale_to_zero_after_s: 600         # idle time (rps=0) before killing last replica
```

**Min/max replica guarantees:**
- `min_replicas` is enforced as a hard floor. If the current count drops below it (e.g. due to failures), the autoscaler immediately scales back up, bypassing cooldown.
- `max_replicas` is enforced as a hard ceiling. No scale-up decision can exceed it.
- Scale-down will never reduce below `min_replicas`.
- Scale-to-zero can only fire when `allow_scale_to_zero: true` AND idle for `scale_to_zero_after_s`.

### Autoscale Event Log (SQLite)

```sql
CREATE TABLE autoscale_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    model           TEXT NOT NULL,
    variant         TEXT NOT NULL,
    action          TEXT NOT NULL,    -- SCALE_UP, SCALE_DOWN, SCALE_TO_ZERO, HOLD
    from_replicas   INTEGER NOT NULL,
    to_replicas     INTEGER NOT NULL,
    reasons         TEXT,             -- JSON array of reason strings
    metrics_snapshot TEXT             -- JSON of all input signals at decision time
);
```

Every decision (including HOLD) is logged so operators can see exactly why the system did or didn't scale.

---

## 7. GPU Management & Safety

### GPU State Tracking

Each GPU on each node has an explicit state:

| State | Meaning |
|-------|---------|
| `FREE` | No vLLM process, memory is available |
| `ALLOCATED` | Assigned to a replica, in use |
| `RESERVED` | Being prepared for a new replica (cleanup in progress) |
| `FAILED` | Hardware error or repeated process crashes |

### GPU Memory Guard (intelligent mitigation + live migration)

CServe enforces a strict per-GPU memory utilization limit (default **79%**).
When a GPU exceeds this limit, the guard follows a graduated response:

```
OK ─[breach]→ WARNED ─[confirmed]→ MITIGATING ─[timeout]→ MIGRATING
↑                ↓                      ↓                      ↓
└──[recovered]───┘      [recovered]─────┘      [migration done]┘
```

| Phase | What happens | Duration |
|-------|-------------|----------|
| **WARNED** | Memory exceeds limit on first check. Guard waits for confirmation. | 1 check cycle |
| **MITIGATING** | Confirmed breach. Replica is paused (set to DRAINING so gateway stops routing). GPU pressure drops as in-flight requests complete. | Up to 10 min (configurable) |
| **Self-heal** | If memory drops below the limit during mitigation, the replica is automatically resumed — no migration needed. | — |
| **MIGRATING** | Mitigation timeout. Guard kills the replica and launches a replacement on a different node/GPU set. Other replicas of the same model keep serving. | Until replacement is healthy |

**Key design choices:**
- The 10-minute mitigation window avoids unnecessary migrations caused by temporary spikes (e.g., a long-context request that temporarily inflates KV-cache).
- During mitigation, the gateway stops sending new requests to the affected replica, giving vLLM time to release cache naturally.
- During migration, the model remains available through its other replicas — users never lose service, just temporarily reduced capacity.
- All guard state transitions are visible in the dashboard's GPU Memory Guard panel and logged to SQLite for audit.

Configuration in `cluster.yaml`:
```yaml
safety:
  gpu_memory_limit: 0.79           # hard limit per GPU
  guard_mitigation_window_s: 600   # 10 min self-heal window
  guard_check_interval_s: 20       # check frequency
  guard_consecutive_breaches: 2    # confirm before mitigation
```

### Cleanup Protocol (node agent)

When freeing GPUs after a replica is stopped:

1. **SIGTERM** the vLLM process group (allows graceful shutdown).
2. Wait up to 15 seconds for in-flight requests to drain.
3. **SIGKILL** the process group if still alive.
4. **nvidia-smi scan**: find any orphaned processes on those GPU indices:
   - Match by process name (`vllm`, `VLLM::Worker_TP`, `VLLM::EngineCore`).
   - Match by command line (`vllm.entrypoints`).
   - SIGTERM → wait 5s → SIGKILL any found.
5. **Verify**: poll `nvidia-smi` until GPU memory is below 100 MiB, or timeout.
6. Report GPU state as `FREE`.

This is the exact logic from the existing cosmos-gpu-cluster, promoted to a first-class node agent operation.

---

## 8. Configuration Model

### `cluster.yaml` — Node Topology

```yaml
cluster:
  head:
    name: cosmos-9
    host: cosmos-9.ddns.ualr.edu
    node_ip: 144.167.35.146

  nodes:
    - name: cosmos-7
      host: 144.167.35.154
      gpu_count: 8
      gpu_type: a40
      cuda_devices: "0,1,2,3,4,5,6,7"
      labels:
        rack: "A"

    - name: cosmos-10
      host: cosmos-10.ddns.ualr.edu
      gpu_count: 4
      gpu_type: l40
      cuda_devices: "0,1,2,3"

node_agent:
  port: 50051
  heartbeat_interval_s: 10

gateway:
  host: "0.0.0.0"
  port: 8002

redis:
  host: "127.0.0.1"
  port: 6379

safety:
  gpu_memory_limit: 0.79
  gpu_warn_threshold: 0.70
  gpu_danger_threshold: 0.88
  gpu_danger_consecutive_checks: 2
  guard_mitigation_window_s: 600
  guard_check_interval_s: 20
  guard_consecutive_breaches: 2
```

### `models.yaml` — Model Definitions

```yaml
global:
  env:
    HF_TOKEN: "hf_..."
    TORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
  defaults:
    max_model_len: 81920
    gpu_memory_utilization: 0.70
    dtype: auto
    trust_remote_code: true

models:
  gemma3-27b:
    served_model_name: "gemma3-27b"
    hf_model: "google/gemma-3-27b-it"
    tp: 2
    node_type_required: "l40"
    node_types_allowed: ["a40", "l40"]
    routing_strategy: "lor"        # least outstanding requests
    engine:
      max_model_len: 81920
      max_num_seqs: 8
    autoscaling:
      min_replicas: 5
      max_replicas: 6
      target_inflight: 2
      # ... full policy ...
```

---

## 9. Wire Protocols

| Path | Protocol | Why |
|------|----------|-----|
| Client → Gateway | HTTP/1.1 + SSE (streaming) | OpenAI SDK compatibility. Bearer auth on every request. |
| Gateway → vLLM replica | HTTP/1.1 + SSE | vLLM's native protocol, direct stream passthrough |
| Control Plane → Node Agent | JSON-over-HTTP **or** gRPC | Switchable at deploy time: `--transport http` (default) / `--transport grpc` |
| Node Agent → Control Plane | HTTP POST (heartbeats, status) | Periodic heartbeat with GPU metrics and replica status reports. |
| Gateway ↔ Redis | Redis protocol (RESP3) | Sub-ms queue ops (overflow path) + per-key rate limit counters. |
| Gateway ↔ DemandTracker | In-process function call | Zero overhead. Sliding-window RPS + inflight tracking for autoscaler. |
| Gateway ↔ SQLite | aiosqlite | Usage logging, API key auth, event log. |
| Dashboard → Control Plane | HTTP + WebSocket | REST for history/CRUD, WS for 1Hz live cluster snapshots. |
| vLLM replicas → Prometheus | HTTP `/metrics` | Standard Prometheus scraping, parsed by MetricsCollector. |

---

## 10. Project Layout

```
CServe/
├── README.md                        # This document
├── pyproject.toml                   # Python project metadata, deps, entry points
│
├── proto/                           # API contract definitions
│   └── node_agent.proto             # NodeAgent gRPC service (the canonical API contract)
│
├── cserve/                          # Main Python package
│   ├── __init__.py
│   │
│   ├── common/                      # Shared utilities (no control-plane imports)
│   │   ├── auth.py                  # API key generation, hashing, models (ApiKey, AuthenticatedUser)
│   │   ├── config.py                # YAML config loader + validation
│   │   ├── demand.py                # DemandTracker: sliding-window RPS + inflight
│   │   ├── logging.py               # Structured logging (structlog)
│   │   ├── metrics.py               # All Prometheus metric definitions
│   │   ├── models.py                # Pydantic models: Node, Replica, Job, Config, etc.
│   │   └── rate_limit.py            # Per-key token-bucket rate limiter (Redis sliding window)
│   │
│   ├── generated/                   # Proto-compiled gRPC stubs (auto-generated)
│   │   ├── __init__.py
│   │   ├── node_agent_pb2.py        # Protobuf message classes
│   │   ├── node_agent_pb2.pyi       # Type stubs
│   │   └── node_agent_pb2_grpc.py   # gRPC servicer/stub classes
│   │
│   ├── control_plane/               # Head node services
│   │   ├── autoscaler.py            # Per-model scaling decisions (DemandTracker + Redis)
│   │   ├── db.py                    # SQLite event log + API key CRUD + usage tracking
│   │   ├── gateway.py               # FastAPI: OpenAI-compatible API + auth + LOR + proxy
│   │   ├── grpc_client.py           # gRPC client for node agents (binary transport)
│   │   ├── gpu_guard.py             # GPU memory guard: threshold FSM, mitigation, live migration
│   │   ├── health.py                # 3-layer health: node, replica, GPU memory guard
│   │   ├── metrics_collector.py     # Scrapes vLLM /metrics, updates registry
│   │   ├── node_client.py           # HTTP client for node agents (JSON transport)
│   │   ├── orchestrator.py          # Replica lifecycle: launch/stop with retries + rollback
│   │   ├── placement.py             # GPU placement algorithm (contiguous, TP-aware)
│   │   ├── queue.py                 # Redis Streams + Sorted Sets job queue
│   │   ├── registry.py              # In-memory cluster state (RLock-protected)
│   │   ├── scheduler.py             # Queue-path job → replica assignment + admission control
│   │   └── server.py                # Entry point: wires all components, admin API, bootstrap key
│   │
│   ├── node_agent/                  # Worker node daemon
│   │   ├── agent.py                 # FastAPI server implementing NodeAgent endpoints (HTTP)
│   │   ├── gpu_monitor.py           # nvidia-smi scraping, zombie detection, process kill
│   │   ├── grpc_server.py           # gRPC server implementing NodeAgent service (binary)
│   │   ├── launcher.py              # vLLM subprocess management (CUDA_VISIBLE_DEVICES)
│   │   └── server.py                # Entry point: --transport http|grpc
│   │
│   ├── dashboard/                   # Dashboard backend
│   │   ├── api.py                   # FastAPI: WebSocket + REST + user/usage endpoints
│   │   └── static/
│   │       ├── index.html           # Legacy: single-file dashboard (HTML/CSS/JS)
│   │       └── dist/                # React build output (served in production)
│   │
│   └── cli/                         # Command-line tools
│       └── ctl.py                   # cserve-ctl: cluster inspection & management
│
├── dashboard-ui/                    # React + TypeScript + Tailwind dashboard
│   ├── package.json                 # Dependencies: react 19, react-router, recharts, lucide
│   ├── vite.config.ts               # Vite bundler config (proxies to :8002, builds to static/dist/)
│   ├── tailwind.config.js           # CServe dark theme: bg, surface, accent, muted colors
│   ├── tsconfig.json
│   ├── index.html                   # HTML shell
│   └── src/
│       ├── main.tsx                 # React entry point
│       ├── App.tsx                  # Router + nav bar + layout
│       ├── index.css                # Tailwind base + scrollbar styles
│       ├── hooks/
│       │   └── useCluster.ts        # WebSocket hook for 1Hz cluster snapshots
│       └── pages/
│           ├── Topology.tsx         # Nodes, GPUs, replicas, queues, request feed
│           ├── Autoscaling.tsx      # Scale events, per-model cards, event log
│           ├── Usage.tsx            # Per-user per-model usage analytics
│           └── UserManagement.tsx   # User list, API key CRUD, key creation form
│
├── configs/                         # Example configuration files
│   ├── cluster.example.yaml
│   └── models.example.yaml
│
├── tests/                           # 148 tests (unit + integration)
│   ├── test_auth.py                 # API key generation, auth, usage tracking
│   ├── test_autoscaler.py           # Autoscaler decision logic
│   ├── test_config.py               # Config loading + validation
│   ├── test_dashboard.py            # Dashboard snapshot building
│   ├── test_db.py                   # SQLite event log operations
│   ├── test_demand.py               # DemandTracker sliding window + spike filtering
│   ├── test_gateway.py              # Gateway API routes + auth + error handling
│   ├── test_gpu_guard.py            # GPU memory guard FSM + mitigation + migration
│   ├── test_gpu_monitor.py          # nvidia-smi XML parsing
│   ├── test_integration.py          # Multi-component integration
│   ├── test_launcher.py             # vLLM command building + env
│   ├── test_metrics_collector.py    # Prometheus text parser
│   ├── test_models.py               # Pydantic model validation
│   ├── test_placement.py            # GPU placement algorithm
│   ├── test_registry.py             # Registry state machine + concurrency
│   └── test_scheduler.py            # Replica selection strategies
```

**CLI entry points** (registered in `pyproject.toml`):
- `cserve-control` — Start the control plane (head node). Auto-creates bootstrap admin key on first run.
- `cserve-agent` — Start a node agent (worker node). Supports `--transport http|grpc`.
- `cserve-ctl` — Cluster inspection: `cserve-ctl status`, `cserve-ctl nodes`, `cserve-ctl replicas`, etc.

---

## 11. Design Principles

1. **vLLM is the engine, CServe is the orchestrator.** We never re-implement inference. vLLM handles batching, KV cache, CUDA kernels. CServe handles everything around it.

2. **Single brain, thin agents.** The head node makes all decisions. Workers are stateless executors. This avoids distributed consensus complexity.

3. **Every state transition is logged.** Jobs, replicas, autoscaling decisions, health incidents — all stored in SQLite with timestamps. If something goes wrong, you can reconstruct exactly what happened.

4. **Autoscaling is signal-driven, not timer-driven.** Scaling decisions use DemandTracker RPS + sustained inflight pressure (primary), Redis queue metrics (secondary), and vLLM engine telemetry (TTFT, KV cache). Short spikes are filtered — only sustained pressure (10+ seconds) triggers scale-up. Scale-down requires idle timeout (default 120s of zero requests). This prevents both under-scaling (high latency) and over-scaling (wasted GPUs).

5. **GPU lifecycle is explicit.** No process is "assumed dead". Every cleanup is verified via `nvidia-smi`. GPU state is tracked per-device.

6. **The dashboard is not optional.** Observability is built in from day one, not bolted on. The same event log that powers autoscaling powers the dashboard.

7. **Config-driven, not code-driven.** Adding a new model means adding a YAML block, not writing new Python code.

8. **Dual transport:** Both JSON-over-HTTP (debuggable) and gRPC (2-3x faster) for control plane ↔ node agent. Switchable at deploy time via `--transport`. HTTP for public API (compatible with every OpenAI SDK); WebSocket for live dashboard.

9. **Multi-tenant by default.** Every request is authenticated via API keys (SHA-256 hashed, never stored raw). Per-key rate limiting via Redis. Per-user usage tracking (requests, GPU time, tokens) powers billing, quota, and the dashboard's Usage page.
