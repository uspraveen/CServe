"""Config sync policy: YAML files vs SQLite ``ui_runtime_tuning``.

On control-plane startup:
  - If YAML files are newer than the newest SQLite row → reload tunables from YAML.
  - Else if SQLite has rows → overlay SQLite on the YAML-loaded base (UI wins).
  - Else → seed SQLite from in-memory YAML (first boot).

Dashboard ``PUT /admin/config`` writes to SQLite and mirrors tunables back to YAML.
"""

from __future__ import annotations

from pathlib import Path

from cserve.common.config import save_cluster_config, save_models_config_disk
from cserve.common.models import ClusterConfig, ModelConfig


def config_paths_mtime(cluster_yaml: str | Path, models_yaml: str | Path) -> float:
    """Latest modification time of the two config files (0 if missing)."""
    mtimes: list[float] = []
    for path in (cluster_yaml, models_yaml):
        p = Path(path)
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def sqlite_max_updated_at(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return max(float(r.get("updated_at") or 0) for r in rows)


def should_apply_yaml_on_startup(
    cluster_yaml: str | Path,
    models_yaml: str | Path,
    sqlite_rows: list[dict],
) -> bool:
    """True when hand-edited YAML should win over a stale SQLite overlay."""
    if not sqlite_rows:
        return False
    return config_paths_mtime(cluster_yaml, models_yaml) > sqlite_max_updated_at(sqlite_rows)


def persist_ui_tuning_to_yaml(
    cluster_yaml: str | Path,
    models_yaml: str | Path,
    cluster_cfg: ClusterConfig,
    models_cfg: dict[str, ModelConfig],
) -> list[str]:
    """Write current in-memory tunables to YAML (preserves unrelated keys)."""
    cluster_path = str(cluster_yaml)
    models_path = str(models_yaml)
    save_cluster_config(cluster_cfg, cluster_path)
    save_models_config_disk(models_path, models_cfg)
    return [f"yaml: {cluster_path}", f"yaml: {models_path}"]
