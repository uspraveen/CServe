"""Tests for environment-based secret resolution in YAML config."""

import os

import pytest
import yaml

from cserve.common.config import load_cluster_config, load_models_config
from cserve.common.config_secrets import resolve_env_placeholders


class TestEnvPlaceholders:
    def test_resolve_top_level(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        assert resolve_env_placeholders("${HF_TOKEN}") == "hf_from_env"

    def test_resolve_nested(self, monkeypatch):
        monkeypatch.setenv("CSERVE_SSH_PASSWORD", "secret")
        raw = {"ssh": {"password": "${CSERVE_SSH_PASSWORD}"}}
        out = resolve_env_placeholders(raw)
        assert out["ssh"]["password"] == "secret"

    def test_missing_env_becomes_empty(self):
        assert resolve_env_placeholders("${NOT_SET_VAR_XYZ}") == ""


class TestLoadConfigSecrets:
    def test_hf_token_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test_env")
        cfg = {
            "global": {"env": {"HF_TOKEN": "${HF_TOKEN}"}},
            "models": {
                "m1": {"served_model_name": "m1", "hf_model": "org/m1"},
            },
        }
        p = tmp_path / "models.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))

        _, models = load_models_config(p)
        assert models["m1"].hf_token == "hf_test_env"

    def test_ssh_password_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CSERVE_SSH_PASSWORD", "ssh_secret")
        cfg = {
            "cluster": {
                "head": {"name": "h1", "host": "10.0.0.1"},
                "nodes": [],
            },
            "ssh": {},
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))

        cluster = load_cluster_config(p)
        assert cluster.ssh.password == "ssh_secret"

    def test_ssh_password_placeholder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CSERVE_SSH_PASSWORD", "via_placeholder")
        cfg = {
            "cluster": {
                "head": {"name": "h1", "host": "10.0.0.1"},
                "nodes": [],
            },
            "ssh": {"password": "${CSERVE_SSH_PASSWORD}"},
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))

        cluster = load_cluster_config(p)
        assert cluster.ssh.password == "via_placeholder"
