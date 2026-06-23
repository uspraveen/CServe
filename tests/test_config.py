"""Tests for config loading and validation."""

from pathlib import Path

import pytest
import yaml

from cserve.common.config import (
    ConfigError,
    load_cluster_config,
    load_models_config,
)


def _write_yaml(data: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


class TestClusterConfig:
    def test_load_valid(self, tmp_path):
        cfg = {
            "cluster": {
                "head": {"name": "head1", "host": "10.0.0.1", "node_ip": "10.0.0.1"},
                "nodes": [
                    {"name": "w1", "host": "10.0.0.2", "gpu_count": 4,
                     "gpu_type": "a40", "cuda_devices": "0,1,2,3"},
                ],
            },
            "gateway": {"host": "0.0.0.0", "port": 8002},
            "redis": {"host": "127.0.0.1", "port": 6379},
        }
        p = tmp_path / "cluster.yaml"
        _write_yaml(cfg, p)

        result = load_cluster_config(p)
        assert result.head.name == "head1"
        assert len(result.nodes) == 1
        assert result.nodes[0].gpu_type == "a40"
        assert result.gateway.port == 8002
        assert result.redis.port == 6379

    def test_missing_head_raises(self, tmp_path):
        cfg = {"cluster": {"nodes": []}}
        p = tmp_path / "cluster.yaml"
        _write_yaml(cfg, p)

        with pytest.raises(ConfigError, match="must define cluster.head"):
            load_cluster_config(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_cluster_config(tmp_path / "nonexistent.yaml")


class TestModelsConfig:
    def test_load_valid(self, tmp_path):
        cfg = {
            "global": {
                "env": {"HF_TOKEN": "hf_test"},
                "defaults": {"max_model_len": 4096, "dtype": "auto"},
            },
            "models": {
                "test-model": {
                    "served_model_name": "test-model",
                    "hf_model": "org/model",
                    "tp": 2,
                    "node_type_required": "a40",
                    "node_types_allowed": ["a40"],
                    "engine": {"max_num_seqs": 8},
                    "autoscaling": {"min_replicas": 1, "max_replicas": 4},
                },
            },
        }
        p = tmp_path / "models.yaml"
        _write_yaml(cfg, p)

        global_cfg, models = load_models_config(p)
        assert global_cfg.env["HF_TOKEN"] == "hf_test"
        assert "test-model" in models
        m = models["test-model"]
        assert m.tp == 2
        assert m.engine.max_num_seqs == 8
        # Global default merged in
        assert m.engine.max_model_len == 4096

    def test_global_defaults_merge(self, tmp_path):
        cfg = {
            "global": {
                "defaults": {"gpu_memory_utilization": 0.85},
            },
            "models": {
                "m1": {
                    "served_model_name": "m1",
                    "hf_model": "org/m1",
                },
            },
        }
        p = tmp_path / "models.yaml"
        _write_yaml(cfg, p)

        _, models = load_models_config(p)
        assert models["m1"].engine.gpu_memory_utilization == 0.85

    def test_model_engine_overrides_global(self, tmp_path):
        cfg = {
            "global": {"defaults": {"gpu_memory_utilization": 0.85}},
            "models": {
                "m1": {
                    "served_model_name": "m1",
                    "hf_model": "org/m1",
                    "engine": {"gpu_memory_utilization": 0.90},
                },
            },
        }
        p = tmp_path / "models.yaml"
        _write_yaml(cfg, p)

        _, models = load_models_config(p)
        assert models["m1"].engine.gpu_memory_utilization == 0.90

    def test_invalid_routing_strategy_raises(self, tmp_path):
        cfg = {
            "global": {},
            "models": {
                "m1": {
                    "served_model_name": "m1",
                    "hf_model": "org/m1",
                    "routing_strategy": "invalid_strategy",
                },
            },
        }
        p = tmp_path / "models.yaml"
        _write_yaml(cfg, p)

        with pytest.raises(ConfigError, match="invalid routing_strategy"):
            load_models_config(p)

    def test_node_type_consistency_check(self, tmp_path):
        cfg = {
            "global": {},
            "models": {
                "m1": {
                    "served_model_name": "m1",
                    "hf_model": "org/m1",
                    "node_type_required": "h100",
                    "node_types_allowed": ["a40", "l40"],
                },
            },
        }
        p = tmp_path / "models.yaml"
        _write_yaml(cfg, p)

        with pytest.raises(ConfigError, match="not in node_types_allowed"):
            load_models_config(p)

    def test_hf_token_inherited_from_global(self, tmp_path):
        cfg = {
            "global": {"env": {"HF_TOKEN": "hf_global_token"}},
            "models": {
                "m1": {
                    "served_model_name": "m1",
                    "hf_model": "org/m1",
                },
            },
        }
        p = tmp_path / "models.yaml"
        _write_yaml(cfg, p)

        _, models = load_models_config(p)
        assert models["m1"].hf_token == "hf_global_token"

    def test_upscale_delay_s_alias(self, tmp_path):
        cfg = {
            "global": {},
            "models": {
                "m1": {
                    "served_model_name": "m1",
                    "hf_model": "org/m1",
                    "autoscaling": {"upscale_delay_s": 42},
                },
            },
        }
        p = tmp_path / "models.yaml"
        _write_yaml(cfg, p)

        _, models = load_models_config(p)
        assert models["m1"].autoscaling.upscale_cooldown_s == 42

    def test_schedulable_round_trip(self, tmp_path):
        from cserve.common.config import save_cluster_config

        cfg = {
            "cluster": {
                "head": {"name": "head1", "host": "10.0.0.1"},
                "nodes": [
                    {"name": "w1", "host": "10.0.0.2", "gpu_count": 4,
                     "gpu_type": "a40", "cuda_devices": "0,1,2,3", "schedulable": False},
                ],
            },
        }
        p = tmp_path / "cluster.yaml"
        _write_yaml(cfg, p)

        loaded = load_cluster_config(p)
        assert loaded.nodes[0].schedulable is False

        loaded.nodes[0].schedulable = True
        save_cluster_config(loaded, p)
        reloaded = load_cluster_config(p)
        assert reloaded.nodes[0].schedulable is True
        raw = yaml.safe_load(p.read_text())
        assert "schedulable" not in raw["cluster"]["nodes"][0]
