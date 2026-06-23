"""Prometheus metric definitions for CServe.

All metrics are defined in one place so there's no duplication and every
dashboard / alert can reference this file as the source of truth.

Metrics are registered lazily (on first import) to avoid problems when
prometheus_client is imported in test environments without a running
HTTP server.

Naming convention: cserve_{component}_{metric_name}_{unit}
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# ═══════════════════════════════════════════════════════════════════════════
# Gateway metrics
# ═══════════════════════════════════════════════════════════════════════════

GATEWAY_REQUESTS_TOTAL = Counter(
    "cserve_gateway_requests_total",
    "Total requests received by the API gateway",
    ["model", "status_code", "tenant"],
)

GATEWAY_REQUEST_DURATION = Histogram(
    "cserve_gateway_request_duration_seconds",
    "End-to-end request duration at the gateway (including queue + inference)",
    ["model", "streaming"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)

GATEWAY_ACTIVE_STREAMS = Gauge(
    "cserve_gateway_active_streams",
    "Currently active SSE streaming connections",
    ["model"],
)

GATEWAY_QUEUE_SUBMIT_DURATION = Histogram(
    "cserve_gateway_queue_submit_seconds",
    "Time to enqueue a job in Redis",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05),
)

# ═══════════════════════════════════════════════════════════════════════════
# Scheduler metrics
# ═══════════════════════════════════════════════════════════════════════════

SCHEDULER_JOBS_SCHEDULED = Counter(
    "cserve_scheduler_jobs_scheduled_total",
    "Total jobs successfully assigned to a replica",
    ["model"],
)

SCHEDULER_JOBS_EXPIRED = Counter(
    "cserve_scheduler_jobs_expired_total",
    "Total jobs that expired before being scheduled",
    ["model"],
)

SCHEDULER_SCHEDULING_DURATION = Histogram(
    "cserve_scheduler_scheduling_seconds",
    "Time to pick a job and assign it to a replica",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
)

SCHEDULER_LOOP_DURATION = Histogram(
    "cserve_scheduler_loop_seconds",
    "Duration of one full scheduler loop iteration",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1),
)

# ═══════════════════════════════════════════════════════════════════════════
# Queue metrics
# ═══════════════════════════════════════════════════════════════════════════

QUEUE_DEPTH = Gauge(
    "cserve_queue_depth",
    "Number of pending jobs in the Redis queue",
    ["model"],
)

QUEUE_TIME_IN_QUEUE = Histogram(
    "cserve_queue_time_in_queue_seconds",
    "Time a job spent waiting in the queue before being scheduled",
    ["model"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

# ═══════════════════════════════════════════════════════════════════════════
# Autoscaler metrics
# ═══════════════════════════════════════════════════════════════════════════

AUTOSCALER_DECISIONS = Counter(
    "cserve_autoscaler_decisions_total",
    "Total autoscaling decisions made",
    ["model", "action"],
)

AUTOSCALER_CURRENT_REPLICAS = Gauge(
    "cserve_autoscaler_current_replicas",
    "Current number of replicas per model",
    ["model"],
)

AUTOSCALER_TARGET_REPLICAS = Gauge(
    "cserve_autoscaler_target_replicas",
    "Target number of replicas per model (desired by autoscaler)",
    ["model"],
)

# ═══════════════════════════════════════════════════════════════════════════
# Registry / Cluster metrics
# ═══════════════════════════════════════════════════════════════════════════

CLUSTER_NODES_TOTAL = Gauge(
    "cserve_cluster_nodes_total",
    "Total configured nodes",
)

CLUSTER_NODES_ONLINE = Gauge(
    "cserve_cluster_nodes_online",
    "Nodes currently online",
)

CLUSTER_GPUS_TOTAL = Gauge(
    "cserve_cluster_gpus_total",
    "Total GPUs across all nodes",
    ["gpu_type"],
)

CLUSTER_GPUS_FREE = Gauge(
    "cserve_cluster_gpus_free",
    "GPUs currently free",
    ["gpu_type"],
)

CLUSTER_GPUS_ALLOCATED = Gauge(
    "cserve_cluster_gpus_allocated",
    "GPUs currently allocated to replicas",
    ["gpu_type"],
)

CLUSTER_REPLICAS_TOTAL = Gauge(
    "cserve_cluster_replicas_total",
    "Total replicas across all models",
    ["status"],
)

# ═══════════════════════════════════════════════════════════════════════════
# Health metrics
# ═══════════════════════════════════════════════════════════════════════════

HEALTH_CHECK_DURATION = Histogram(
    "cserve_health_check_duration_seconds",
    "Duration of a single health check",
    ["check_type"],  # "node", "replica", "gpu"
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
)

HEALTH_INCIDENTS = Counter(
    "cserve_health_incidents_total",
    "Total health incidents detected",
    ["incident_type"],  # "node_offline", "replica_unhealthy", "gpu_danger", "zombie_killed"
)

# ═══════════════════════════════════════════════════════════════════════════
# Node Agent metrics (exported by each agent, scraped by collector)
# ═══════════════════════════════════════════════════════════════════════════

AGENT_GPU_MEMORY_USED = Gauge(
    "cserve_agent_gpu_memory_used_mb",
    "GPU memory used in MB",
    ["node", "gpu_index"],
)

AGENT_GPU_MEMORY_TOTAL = Gauge(
    "cserve_agent_gpu_memory_total_mb",
    "GPU memory total in MB",
    ["node", "gpu_index"],
)

AGENT_GPU_UTILIZATION = Gauge(
    "cserve_agent_gpu_utilization_pct",
    "GPU compute utilization percentage",
    ["node", "gpu_index"],
)

AGENT_GPU_TEMPERATURE = Gauge(
    "cserve_agent_gpu_temperature_c",
    "GPU temperature in Celsius",
    ["node", "gpu_index"],
)

# ═══════════════════════════════════════════════════════════════════════════
# Build info
# ═══════════════════════════════════════════════════════════════════════════

BUILD_INFO = Info(
    "cserve_build",
    "CServe build information",
)
