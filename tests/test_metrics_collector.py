"""Tests for the metrics collector's Prometheus text parser."""

from __future__ import annotations

from cserve.control_plane.metrics_collector import MetricsCollector

SAMPLE_VLLM_METRICS = """# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 1.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.452
# HELP vllm:cpu_cache_usage_perc CPU KV-cache usage.
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc 0.0
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST"} 1234
# some random metric we don't care about
random_metric 42.0
"""


class TestPrometheusParser:
    def test_extracts_vllm_metrics(self):
        result = MetricsCollector._parse_prometheus_text(SAMPLE_VLLM_METRICS)
        assert result["vllm:num_requests_running"] == 3.0
        assert result["vllm:num_requests_waiting"] == 1.0
        assert abs(result["vllm:gpu_cache_usage_perc"] - 0.452) < 0.001
        assert result["vllm:cpu_cache_usage_perc"] == 0.0

    def test_ignores_non_vllm_metrics(self):
        result = MetricsCollector._parse_prometheus_text(SAMPLE_VLLM_METRICS)
        assert "http_requests_total" not in result
        assert "random_metric" not in result

    def test_handles_empty_input(self):
        result = MetricsCollector._parse_prometheus_text("")
        assert result == {}

    def test_handles_comments_only(self):
        result = MetricsCollector._parse_prometheus_text("# just a comment\n# another")
        assert result == {}

    def test_scientific_notation(self):
        text = "vllm:num_requests_running 1.5e2\n"
        result = MetricsCollector._parse_prometheus_text(text)
        assert result["vllm:num_requests_running"] == 150.0
