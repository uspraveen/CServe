"""Tests for config sync policy (YAML vs SQLite)."""

import time
from pathlib import Path

import pytest
import yaml

from cserve.common.config_sync import (
    config_paths_mtime,
    should_apply_yaml_on_startup,
    sqlite_max_updated_at,
)


def _touch(path: Path, mtime: float) -> None:
    path.touch()
    import os
    os.utime(path, (mtime, mtime))


class TestConfigSyncPolicy:
    def test_empty_sqlite_never_forces_yaml(self, tmp_path: Path):
        cluster = tmp_path / "cluster.yaml"
        models = tmp_path / "models.yaml"
        cluster.write_text("cluster: {}\n")
        models.write_text("models: {}\n")
        assert should_apply_yaml_on_startup(cluster, models, []) is False

    def test_yaml_newer_than_sqlite_wins(self, tmp_path: Path):
        cluster = tmp_path / "cluster.yaml"
        models = tmp_path / "models.yaml"
        cluster.write_text("cluster: {}\n")
        models.write_text("models: {}\n")
        now = time.time()
        _touch(cluster, now + 10)
        _touch(models, now + 10)
        rows = [{"updated_at": now, "source": "ui"}]
        assert should_apply_yaml_on_startup(cluster, models, rows) is True

    def test_sqlite_newer_than_yaml_wins(self, tmp_path: Path):
        cluster = tmp_path / "cluster.yaml"
        models = tmp_path / "models.yaml"
        cluster.write_text("cluster: {}\n")
        models.write_text("models: {}\n")
        now = time.time()
        _touch(cluster, now)
        _touch(models, now)
        rows = [{"updated_at": now + 60, "source": "ui"}]
        assert should_apply_yaml_on_startup(cluster, models, rows) is False

    def test_sqlite_max_updated_at(self):
        assert sqlite_max_updated_at([]) == 0.0
        assert sqlite_max_updated_at([{"updated_at": 1.0}, {"updated_at": 5.5}]) == 5.5

    def test_config_paths_mtime_missing_is_zero(self, tmp_path: Path):
        assert config_paths_mtime(tmp_path / "nope.yaml", tmp_path / "also-nope.yaml") == 0.0
