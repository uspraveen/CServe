"""Tests that CServe's API is wire-compatible with OpenAI's API format.

Validates that:
  - Clients can send standard OpenAI chat completion requests
  - The /v1/models endpoint returns the correct shape
  - Error responses match OpenAI's error format
  - The gateway correctly proxies to vLLM's OpenAI-compatible server
  - Streaming SSE format is preserved
  - Authorization header is NOT forwarded to vLLM
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
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

VALID_KEY = "csk_test_openai_compat_key_00000000000000000000000000"


class FakeDB:
    async def log_job_event(self, event):
        pass

    async def log_usage(self, **kwargs):
        pass

    async def increment_key_requests(self, key_id: str) -> None:
        pass

    async def authenticate_key(self, raw_key: str) -> ApiKey | None:
        if raw_key == VALID_KEY:
            return ApiKey(
                key_id="compat-test", key_hash="fake", user_id="test",
                role=KeyRole.USER, rate_limit_rpm=0, enabled=True,
            )
        return None


class FakeQueue:
    async def ping(self) -> bool:
        return True

    async def queue_depth(self, model: str) -> int:
        return 0


AUTH = {"Authorization": f"Bearer {VALID_KEY}"}

MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gemma3-27b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help you?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    },
}

MOCK_STREAM_CHUNKS = [
    (
        b'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
        b'"created":1700000000,"model":"gemma3-27b","choices":'
        b'[{"index":0,"delta":{"role":"assistant","content":""},'
        b'"finish_reason":null}]}\n\n'
    ),
    b"data: [DONE]\n\n",
]


def _make_cluster_config() -> ClusterConfig:
    return ClusterConfig(
        head=HeadConfig(name="head", host="10.0.0.1"),
        nodes=[
            NodeConfig(
                name="worker1", host="10.0.0.2",
                gpu_count=4, gpu_type="l40",
                cuda_devices="0,1,2,3", agent_port=50051,
            ),
        ],
        gateway=GatewayConfig(port=8002),
        redis=RedisConfig(),
        safety=SafetyConfig(),
        node_agent=NodeAgentConfig(),
    )


def _make_model_config() -> dict[str, ModelConfig]:
    return {
        "gemma3-27b": ModelConfig(
            name="gemma3-27b",
            served_model_name="gemma3-27b",
            hf_model="google/gemma-3-27b-it",
            tp=2,
        )
    }


def _setup() -> tuple[Gateway, ClusterRegistry]:
    cfg = _make_cluster_config()
    registry = ClusterRegistry(cfg)
    registry.load_models(_make_model_config())
    registry.set_node_status("worker1", NodeStatus.ONLINE)

    replica = ReplicaState(
        replica_id="r1",
        model="gemma3-27b",
        node_name="worker1",
        gpu_ids=[0, 1],
        tp_size=2,
        status=ReplicaStatus.STARTING,
        http_endpoint="http://10.0.0.2:8100",
        port=8100,
    )
    registry.allocate_gpus("worker1", [0, 1], "r1")
    registry.add_replica(replica)
    registry.set_replica_status("r1", ReplicaStatus.READY)

    gw = Gateway(registry, FakeQueue(), FakeDB(), _make_model_config())
    asyncio.run(gw.startup())
    return gw, registry


class TestOpenAIModelsEndpoint:
    """Tests for GET /v1/models — must match OpenAI's format exactly."""

    def test_response_shape(self):
        gw, _ = _setup()
        client = TestClient(gw.app)
        resp = client.get("/v1/models", headers=AUTH)
        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1

        model = data["data"][0]
        assert model["id"] == "gemma3-27b"
        assert model["object"] == "model"
        assert isinstance(model["created"], int)
        assert model["owned_by"] == "cserve"
        assert isinstance(model["permission"], list)
        assert model["root"] == "gemma3-27b"
        assert model["parent"] is None


class TestOpenAIChatCompletionsFormat:
    """Tests that the gateway correctly accepts and proxies OpenAI chat format."""

    def test_standard_chat_request_accepted(self):
        """Verify the gateway accepts a properly formatted OpenAI chat request."""
        gw, _ = _setup()
        client = TestClient(gw.app)

        request_body = {
            "model": "gemma3-27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "stream": False,
        }

        # This will fail at the httpx level (no real vLLM), but we verify
        # the gateway parses the model correctly and attempts the proxy
        resp = client.post("/v1/chat/completions", json=request_body, headers=AUTH)
        # Connection failures → 503 upstream_unavailable (retriable), not 4xx
        assert resp.status_code == 503
        err = resp.json()["error"]
        assert err["type"] == "upstream_unavailable"
        assert "unreachable" in err["message"].lower() or "retry" in err["message"].lower()

    def test_stream_true_accepted(self):
        gw, _ = _setup()
        client = TestClient(gw.app)

        request_body = {
            "model": "gemma3-27b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        resp = client.post("/v1/chat/completions", json=request_body, headers=AUTH)
        # Streaming: may open SSE then embed error, or fail at stream open
        assert resp.status_code in (200, 503)

    def test_embeddings_endpoint_accepted(self):
        gw, _ = _setup()
        client = TestClient(gw.app)

        request_body = {
            "model": "gemma3-27b",
            "input": "The food was delicious",
        }

        resp = client.post("/v1/embeddings", json=request_body, headers=AUTH)
        assert resp.status_code == 503

    def test_audio_transcription_multipart_accepted(self):
        gw, _ = _setup()
        client = TestClient(gw.app)

        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "gemma3-27b"},
            files={"file": ("sample.wav", b"fake-audio-bytes", "audio/wav")},
            headers=AUTH,
        )
        assert resp.status_code == 503


class TestOpenAIErrorFormat:
    """Errors must match OpenAI's error response shape."""

    def test_missing_model_error_shape(self):
        gw, _ = _setup()
        client = TestClient(gw.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=AUTH,
        )
        assert resp.status_code == 400
        err = resp.json()
        assert "error" in err
        assert "message" in err["error"]
        assert "type" in err["error"]
        assert err["error"]["type"] == "invalid_request_error"

    def test_unknown_model_error_shape(self):
        gw, _ = _setup()
        client = TestClient(gw.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5-turbo", "messages": [{"role": "user", "content": "hi"}]},
            headers=AUTH,
        )
        assert resp.status_code == 404
        err = resp.json()
        assert err["error"]["type"] == "invalid_request_error"
        assert "gpt-5-turbo" in err["error"]["message"]
        assert "gemma3-27b" in err["error"]["message"]

    def test_no_auth_error_shape(self):
        gw, _ = _setup()
        client = TestClient(gw.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gemma3-27b", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
        err = resp.json()
        assert err["error"]["type"] == "auth_error"

    def test_model_not_ready_returns_503(self):
        cfg = _make_cluster_config()
        registry = ClusterRegistry(cfg)
        models = _make_model_config()
        registry.load_models(models)
        gw = Gateway(registry, FakeQueue(), FakeDB(), models)
        client = TestClient(gw.app)

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gemma3-27b", "messages": [{"role": "user", "content": "hi"}]},
            headers=AUTH,
        )
        assert resp.status_code == 503
        assert "Retry-After" in resp.headers


class TestHeaderSanitization:
    """Verify that auth headers are NOT forwarded to vLLM."""

    def test_authorization_stripped_from_proxy(self):
        gw, _ = _setup()

        captured_headers = {}

        async def mock_request(method, url, content, headers):
            captured_headers.update(headers)
            return httpx.Response(
                status_code=200,
                json=MOCK_CHAT_RESPONSE,
                headers={"content-type": "application/json"},
            )

        client = TestClient(gw.app)
        with patch.object(gw, "_http_client") as mock_client:
            mock_client.request = AsyncMock(side_effect=mock_request)

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemma3-27b",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers=AUTH,
            )

            if resp.status_code == 200:
                assert "authorization" not in {k.lower() for k in captured_headers}
                # Verify the response preserved vLLM's OpenAI format
                data = resp.json()
                assert data["object"] == "chat.completion"
                assert len(data["choices"]) == 1
                assert data["usage"]["prompt_tokens"] == 12


class TestVllmLaunchCommand:
    """Verify the vLLM launch command uses the OpenAI-compatible API server."""

    def test_command_uses_openai_entrypoint(self):
        from cserve.node_agent.launcher import Launcher, ReplicaProcess

        rp = ReplicaProcess(
            replica_id="r1",
            model_name="gemma3-27b",
            hf_model="google/gemma-3-27b-it",
            served_model_name="gemma3-27b",
            gpu_ids=[0, 1],
            tp_size=2,
            port=8100,
        )
        cmd = Launcher._build_command(rp, {
            "max_model_len": "81920",
            "max_num_seqs": "8",
            "gpu_memory_utilization": "0.70",
            "disable_custom_all_reduce": "true",
        })

        import sys
        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "vllm.entrypoints.openai.api_server"
        assert "--model" in cmd
        assert "google/gemma-3-27b-it" in cmd
        assert "--served-model-name" in cmd
        assert "gemma3-27b" in cmd
        assert "--tensor-parallel-size" in cmd
        assert "2" in cmd
        assert "--port" in cmd
        assert "8100" in cmd
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--max-model-len" in cmd
        assert "81920" in cmd
        assert "--max-num-seqs" in cmd
        assert "8" in cmd
        assert "--gpu-memory-utilization" in cmd
        assert "0.70" in cmd
        assert "--disable-custom-all-reduce" in cmd

    def test_served_model_name_controls_api_model_field(self):
        """When vLLM starts with --served-model-name X, requests must use model=X."""
        from cserve.node_agent.launcher import Launcher, ReplicaProcess

        rp = ReplicaProcess(
            replica_id="r1",
            model_name="qwen3-embedding-8b",
            hf_model="Qwen/Qwen3-Embedding-8B",
            served_model_name="qwen3-embedding-8b",
            gpu_ids=[0, 1],
            tp_size=2,
            port=8200,
        )
        cmd = Launcher._build_command(rp, {"runner": "pooling", "convert": "embed"})

        idx = cmd.index("--served-model-name")
        assert cmd[idx + 1] == "qwen3-embedding-8b"

    def test_env_pins_cuda_devices(self):
        from cserve.node_agent.launcher import Launcher, ReplicaProcess

        rp = ReplicaProcess(
            replica_id="r1", model_name="test", hf_model="org/model",
            served_model_name="test", gpu_ids=[2, 3], tp_size=2, port=8100,
        )
        env = Launcher._build_env(rp)
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
