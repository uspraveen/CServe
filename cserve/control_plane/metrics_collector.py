"""Metrics Collector — scrapes vLLM /metrics and node agent GPU stats.

Runs as an async background task.  Periodically:
  1. Scrapes each vLLM replica's /metrics endpoint (Prometheus text format).
  2. Parses key metrics and stores them in the registry's replica state.
  3. Feeds these metrics to the autoscaler (via the registry).

Also updates the Prometheus metrics that CServe itself exports (so Grafana
can scrape a single CServe endpoint for both CServe and vLLM metrics).
"""

from __future__ import annotations

import asyncio
import re

import httpx

from cserve.common.logging import get_logger
from cserve.common.metrics import (
    AGENT_GPU_MEMORY_TOTAL,
    AGENT_GPU_MEMORY_USED,
    AGENT_GPU_TEMPERATURE,
    AGENT_GPU_UTILIZATION,
    CLUSTER_GPUS_ALLOCATED,
    CLUSTER_GPUS_FREE,
    CLUSTER_GPUS_TOTAL,
    CLUSTER_NODES_ONLINE,
    CLUSTER_NODES_TOTAL,
    CLUSTER_REPLICAS_TOTAL,
)
from cserve.common.models import NodeStatus, ReplicaStatus

log = get_logger("metrics_collector")

SCRAPE_INTERVAL_S = 10.0

# vLLM metrics we care about for autoscaling and dashboard
VLLM_METRICS_OF_INTEREST = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:e2e_request_latency_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
]

# Regex to parse Prometheus text format lines: metric_name{labels} value
_PROM_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'       # metric name
    r'(?:\{([^}]*)\})?'                    # optional labels
    r'\s+'                                 # whitespace
    r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'  # value
)


class MetricsCollector:
    def __init__(self, registry) -> None:
        self.registry = registry
        self._running = False
        self._task: asyncio.Task | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._running = True
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
        )
        self._task = asyncio.create_task(self._scrape_loop())
        log.info("metrics collector started", interval_s=SCRAPE_INTERVAL_S)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._http_client:
            await self._http_client.aclose()
        log.info("metrics collector stopped")

    async def _scrape_loop(self) -> None:
        while self._running:
            try:
                await self._scrape_all()
                self._update_cluster_metrics()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("scrape loop error", error=str(e))
            await asyncio.sleep(SCRAPE_INTERVAL_S)

    async def _scrape_all(self) -> None:
        """Scrape vLLM /metrics from all ready replicas."""
        replicas = self.registry.get_all_replicas()
        tasks = []
        for r in replicas:
            if r.status == ReplicaStatus.READY and r.http_endpoint:
                tasks.append(self._scrape_replica(r.replica_id, r.http_endpoint))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _scrape_replica(self, replica_id: str, endpoint: str) -> None:
        """Scrape a single vLLM replica's /metrics."""
        try:
            resp = await self._http_client.get(f"{endpoint}/metrics")
            if resp.status_code != 200:
                return

            metrics = self._parse_prometheus_text(resp.text)
            self.registry.update_replica_metrics(replica_id, metrics)

        except Exception as e:
            log.warning("failed to scrape replica metrics",
                        replica=replica_id, error=str(e))

    @staticmethod
    def _parse_prometheus_text(text: str) -> dict[str, float]:
        """Parse Prometheus text format, extracting metrics of interest.

        For histogram metrics, we extract the quantile/bucket values.
        For gauges/counters, we extract the raw value.
        Returns a flat dict like:
          {"vllm:num_requests_running": 3.0,
           "vllm:gpu_cache_usage_perc": 0.45,
           "vllm:time_to_first_token_seconds_p95": 0.32}
        """
        result: dict[str, float] = {}

        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = _PROM_LINE_RE.match(line)
            if not match:
                continue

            name = match.group(1)
            labels_str = match.group(2) or ""
            value = float(match.group(3))

            # Check if this metric is one we care about
            base_name = name.removesuffix("_bucket").removesuffix("_sum").removesuffix("_count")
            if not any(base_name.startswith(m) for m in VLLM_METRICS_OF_INTEREST):
                continue

            # For histogram quantiles (from _bucket lines), extract percentiles
            if name.endswith("_bucket"):
                if 'le="0.95"' in labels_str or 'le="0.95' in labels_str:
                    result[f"{base_name}_p95"] = value
                elif 'le="0.99"' in labels_str:
                    result[f"{base_name}_p99"] = value
            elif "_sum" not in name and "_count" not in name:
                # Gauge or counter — store directly
                result[name] = value

        return result

    def _update_cluster_metrics(self) -> None:
        """Update CServe's own Prometheus metrics for the cluster."""
        nodes = self.registry.get_all_nodes()
        CLUSTER_NODES_TOTAL.set(len(nodes))
        CLUSTER_NODES_ONLINE.set(sum(1 for n in nodes if n.status != NodeStatus.OFFLINE))

        gpu_summary = self.registry.total_gpus_by_type()
        for gpu_type, (total, free) in gpu_summary.items():
            CLUSTER_GPUS_TOTAL.labels(gpu_type=gpu_type).set(total)
            CLUSTER_GPUS_FREE.labels(gpu_type=gpu_type).set(free)
            CLUSTER_GPUS_ALLOCATED.labels(gpu_type=gpu_type).set(total - free)

        replicas = self.registry.get_all_replicas()
        status_counts: dict[str, int] = {}
        for r in replicas:
            status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1
        for status in ReplicaStatus:
            CLUSTER_REPLICAS_TOTAL.labels(status=status.value).set(status_counts.get(status.value, 0))

        # Update per-node GPU metrics
        for node in nodes:
            for gpu in node.gpus:
                labels = {"node": node.name, "gpu_index": str(gpu.index)}
                AGENT_GPU_MEMORY_USED.labels(**labels).set(gpu.memory_used_mb)
                AGENT_GPU_MEMORY_TOTAL.labels(**labels).set(gpu.memory_total_mb)
                AGENT_GPU_UTILIZATION.labels(**labels).set(gpu.utilization_pct)
                AGENT_GPU_TEMPERATURE.labels(**labels).set(gpu.temperature_c)
