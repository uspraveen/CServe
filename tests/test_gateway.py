"""Tests for the gateway module (unit-level, no HTTP)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from cserve.common.auth import ApiKey, KeyRole
from cserve.common.models import (
    ClusterConfig,
    GatewayConfig,
    HeadConfig,
    ModelConfig,
    NodeAgentConfig,
    NodeConfig,
    NodeStatus,
    RedisConfig,
    ReplicaState,
    ReplicaStatus,
    SafetyConfig,
)
from cserve.control_plane.gateway import Gateway
from cserve.control_plane.registry import ClusterRegistry

VALID_KEY = "csk_test_valid_key_for_gateway_tests_0000000000000000"


class FakeDB:
    """Minimal stub for EventLog with auth support."""
    async def log_job_event(self, event):
        pass

    async def log_autoscale_event(self, event):
        pass

    async def log_usage(self, **kwargs):
        pass

    async def increment_key_requests(self, key_id: str) -> None:
        pass

    async def authenticate_key(self, raw_key: str) -> ApiKey | None:
        if raw_key == VALID_KEY:
            return ApiKey(
                key_id="test-key", key_hash="fake", user_id="test-user",
                role=KeyRole.USER, rate_limit_rpm=0, enabled=True,
            )
        return None


class FakeQueue:
    """Minimal stub for JobQueue."""
    async def ping(self) -> bool:
        return True

    async def queue_depth(self, model: str) -> int:
        return 0


AUTH_HEADER = {"Authorization": f"Bearer {VALID_KEY}"}


def _make_cluster_config() -> ClusterConfig:
    return ClusterConfig(
        head=HeadConfig(name="head", host="10.0.0.1"),
        nodes=[
            NodeConfig(
                name="worker1", host="10.0.0.2",
                gpu_count=2, gpu_type="a40",
                cuda_devices="0,1", agent_port=50051,
            ),
        ],
        gateway=GatewayConfig(port=8002),
        redis=RedisConfig(),
        safety=SafetyConfig(),
        node_agent=NodeAgentConfig(),
    )


def _make_model_config() -> dict[str, ModelConfig]:
    return {
        "test-model": ModelConfig(
            name="test-model",
            served_model_name="test-model",
            hf_model="org/test-model",
            tp=1,
        )
    }


def _setup_gateway_with_replica() -> tuple[Gateway, ClusterRegistry]:
    """Create a gateway with one ready replica."""
    cfg = _make_cluster_config()
    registry = ClusterRegistry(cfg)
    registry.load_models(_make_model_config())
    registry.set_node_status("worker1", NodeStatus.ONLINE)

    replica = ReplicaState(
        replica_id="r1",
        model="test-model",
        node_name="worker1",
        gpu_ids=[0],
        tp_size=1,
        status=ReplicaStatus.STARTING,
        http_endpoint="http://10.0.0.2:8100",
        port=8100,
    )
    registry.allocate_gpus("worker1", [0], "r1")
    registry.add_replica(replica)
    registry.set_replica_status("r1", ReplicaStatus.READY)

    gw = Gateway(registry, FakeQueue(), FakeDB(), _make_model_config())
    return gw, registry


class TestGatewayRoutes:
    def test_health_no_auth_needed(self):
        gw, _ = _setup_gateway_with_replica()
        client = TestClient(gw.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "test-model" in data["models"]

    def test_list_models_requires_auth(self):
        gw, _ = _setup_gateway_with_replica()
        client = TestClient(gw.app)
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_list_models_with_auth(self):
        gw, _ = _setup_gateway_with_replica()
        client = TestClient(gw.app)
        resp = client.get("/v1/models", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        m = data["data"][0]
        assert m["id"] == "test-model"
        assert m["object"] == "model"
        assert "created" in m
        assert "owned_by" in m
        assert "root" in m
        assert "permission" in m
        assert m["parent"] is None

    def test_invalid_key_returns_401(self):
        gw, _ = _setup_gateway_with_replica()
        client = TestClient(gw.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer csk_invalid_garbage"},
        )
        assert resp.status_code == 401

    def test_missing_auth_returns_401(self):
        gw, _ = _setup_gateway_with_replica()
        client = TestClient(gw.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401

    def test_missing_model_returns_400(self):
        gw, _ = _setup_gateway_with_replica()
        client = TestClient(gw.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 400
        assert "model" in resp.json()["error"]["message"].lower()

    def test_unknown_model_returns_404(self):
        gw, _ = _setup_gateway_with_replica()
        client = TestClient(gw.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 404

    def test_no_replicas_returns_503(self):
        cfg = _make_cluster_config()
        registry = ClusterRegistry(cfg)
        models = _make_model_config()
        registry.load_models(models)
        gw = Gateway(registry, FakeQueue(), FakeDB(), models)
        client = TestClient(gw.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After", "") in ("15", "30")
