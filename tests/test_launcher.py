"""Tests for the launcher module (process management, command building)."""

from __future__ import annotations

from cserve.node_agent.launcher import Launcher, ReplicaProcess, read_replica_log


class TestCommandBuilding:
    def test_basic_command(self):
        rp = ReplicaProcess(
            replica_id="r1",
            model_name="test",
            hf_model="org/model",
            served_model_name="test-model",
            gpu_ids=[0, 1],
            tp_size=2,
            port=8100,
        )
        cmd = Launcher._build_command(rp, {})
        assert "python" in cmd[0]
        assert "--model" in cmd
        assert "org/model" in cmd
        assert "--served-model-name" in cmd
        assert "test-model" in cmd
        assert "--tensor-parallel-size" in cmd
        assert "2" in cmd
        assert "--port" in cmd
        assert "8100" in cmd

    def test_engine_args_with_dashes(self):
        rp = ReplicaProcess(
            replica_id="r1",
            model_name="test",
            hf_model="org/model",
            served_model_name="test",
            gpu_ids=[0],
            tp_size=1,
            port=8100,
        )
        engine_args = {
            "max_model_len": "4096",
            "gpu_memory_utilization": "0.8",
            "trust_remote_code": "true",
            "enable_chunked_prefill": "false",
        }
        cmd = Launcher._build_command(rp, engine_args)
        assert "--max-model-len" in cmd
        assert "4096" in cmd
        assert "--gpu-memory-utilization" in cmd
        assert "0.8" in cmd
        assert "--trust-remote-code" in cmd
        # false booleans should NOT appear
        assert "--enable-chunked-prefill" not in cmd

    def test_engine_args_with_double_dash_prefix(self):
        rp = ReplicaProcess(
            replica_id="r1",
            model_name="test",
            hf_model="org/model",
            served_model_name="test",
            gpu_ids=[0],
            tp_size=1,
            port=8100,
        )
        engine_args = {"--custom-flag": "value"}
        cmd = Launcher._build_command(rp, engine_args)
        assert "--custom-flag" in cmd
        assert "value" in cmd

    def test_env_building(self):
        rp = ReplicaProcess(
            replica_id="r1",
            model_name="test",
            hf_model="org/model",
            served_model_name="test",
            gpu_ids=[2, 3],
            tp_size=2,
            port=8100,
            env_override={"HF_TOKEN": "secret123"},
        )
        env = Launcher._build_env(rp)
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert env["VLLM_NO_USAGE_STATS"] == "1"
        assert env["HF_TOKEN"] == "secret123"


class TestLauncherState:
    def test_initial_state_empty(self):
        launcher = Launcher("node1", "10.0.0.1")
        assert launcher.all_replicas() == []
        assert launcher.get_replica("nonexistent") is None

    def test_is_alive_returns_false_for_unknown(self):
        launcher = Launcher("node1", "10.0.0.1")
        assert launcher.is_alive("nonexistent") is False


class TestReadReplicaLog:
    def test_incremental_tail(self, tmp_path):
        log = tmp_path / "vllm-r1.log"
        log.write_text("line1\nline2\nline3\n")
        first = read_replica_log(str(log), offset=0, max_lines=10)
        assert "line3" in first["text"]
        assert first["offset"] == log.stat().st_size

        log.write_text("line1\nline2\nline3\nline4\n")
        second = read_replica_log(str(log), offset=first["offset"])
        assert "line4" in second["text"]
        assert second["offset"] == log.stat().st_size
