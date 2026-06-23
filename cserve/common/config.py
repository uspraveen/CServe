"""Configuration loader with strict validation.

Loads cluster.yaml and models.yaml, validates them through Pydantic models,
and merges global defaults into per-model engine configs.

Fails fast and loud on bad config — never silently assume defaults for
fields that affect GPU allocation or model loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .config_secrets import apply_cluster_secret_env, apply_models_secret_env

from .models import (
    AutoscalePolicy,
    ClusterConfig,
    EngineConfig,
    GatewayConfig,
    GlobalConfig,
    HeadConfig,
    ModelConfig,
    NodeAgentConfig,
    NodeConfig,
    RedisConfig,
    RoutingStrategy,
    SafetyConfig,
    SshConfig,
)


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    with p.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must be a YAML mapping: {p}")
    return data


# ─── Cluster config ──────────────────────────────────────────────────────

def load_cluster_config(path: str | Path) -> ClusterConfig:
    """Load and validate cluster.yaml.

    Secrets (``ssh.password``, ``redis.password``) come from ``${CSERVE_SSH_PASSWORD}``
    / ``${REDIS_PASSWORD}`` placeholders or those environment variables at runtime.
    """
    raw = apply_cluster_secret_env(_load_yaml(path))

    cluster_block = raw.get("cluster")
    if not cluster_block or not isinstance(cluster_block, dict):
        raise ConfigError("cluster.yaml must have a top-level 'cluster' key")

    head_raw = cluster_block.get("head")
    if not head_raw:
        raise ConfigError("cluster.yaml must define cluster.head")

    try:
        head = HeadConfig(**head_raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid head config: {e}") from e

    nodes: list[NodeConfig] = []
    for i, n in enumerate(cluster_block.get("nodes") or []):
        try:
            nodes.append(NodeConfig(**n))
        except ValidationError as e:
            raise ConfigError(f"Invalid node config at index {i}: {e}") from e

    gateway = GatewayConfig(**(raw.get("gateway") or {}))
    redis = RedisConfig(**(raw.get("redis") or {}))
    safety = SafetyConfig(**(raw.get("safety") or {}))
    node_agent = NodeAgentConfig(**(raw.get("node_agent") or {}))
    ssh = SshConfig(**(raw.get("ssh") or {}))

    return ClusterConfig(
        head=head,
        nodes=nodes,
        gateway=gateway,
        redis=redis,
        safety=safety,
        node_agent=node_agent,
        ssh=ssh,
    )


def save_cluster_config(cluster_cfg: ClusterConfig, path: str | Path) -> None:
    """Write the cluster config back to cluster.yaml.

    Reads the existing file first to preserve comments and unknown keys,
    then updates only the sections we own (cluster.nodes, ssh).
    If the file doesn't exist or is malformed, writes a clean YAML.
    """
    p = Path(path)
    try:
        with p.open() as f:
            raw: dict = yaml.safe_load(f) or {}
    except Exception:
        raw = {}

    # Update the nodes list
    cluster_block = raw.setdefault("cluster", {})
    cluster_block["nodes"] = [
        {
            "name": n.name,
            "host": n.host,
            "gpu_count": n.gpu_count,
            "gpu_type": n.gpu_type,
            "cuda_devices": n.cuda_devices,
            **({"schedulable": False} if not n.schedulable else {}),
            **({"labels": dict(n.labels)} if n.labels else {}),
        }
        for n in cluster_cfg.nodes
    ]

    # Update the ssh section
    ssh_block: dict = {
        "username": cluster_cfg.ssh.username,
        "key_path": cluster_cfg.ssh.key_path,
        "port": cluster_cfg.ssh.port,
        "timeout_s": cluster_cfg.ssh.timeout_s,
        "cserve_src": cluster_cfg.ssh.cserve_src,
        "python_path": cluster_cfg.ssh.python_path,
        "pip_path": cluster_cfg.ssh.pip_path,
    }
    # Never persist passwords — use CSERVE_SSH_PASSWORD in the environment.
    raw["ssh"] = ssh_block

    raw["safety"] = cluster_cfg.safety.model_dump()

    with p.open("w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def save_models_config_disk(path: str | Path, models: dict[str, ModelConfig]) -> None:
    """Write per-model ``engine`` and ``autoscaling`` blocks back to models.yaml.

    Merges into the existing file so ``global``, ``hf_model``, routing, etc. are
    preserved. Runtime dashboard edits persist to SQLite; use
    ``scripts/export_ui_tuning_to_yaml.py`` to merge DB overlays into files.
    """
    p = Path(path)
    try:
        with p.open() as f:
            raw: dict = yaml.safe_load(f) or {}
    except Exception:
        raw = {}

    models_raw = raw.get("models")
    if not isinstance(models_raw, dict):
        models_raw = {}
        raw["models"] = models_raw

    for name, cfg in models.items():
        if name not in models_raw or not isinstance(models_raw[name], dict):
            continue
        m = models_raw[name]
        eng = dict(m.get("engine") or {})
        eng.update(cfg.engine.model_dump())
        m["engine"] = eng
        asc = dict(m.get("autoscaling") or {})
        asc.update(cfg.autoscaling.model_dump())
        asc.pop("upscale_delay_s", None)  # legacy alias — canonical key is upscale_cooldown_s
        m["autoscaling"] = asc

    with p.open("w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ─── Models config ───────────────────────────────────────────────────────

def load_models_config(path: str | Path) -> tuple[GlobalConfig, dict[str, ModelConfig]]:
    """Load and validate models.yaml.

    Returns (global_config, {model_key: ModelConfig}).
    Global defaults are merged into each model's engine config.

    ``global.env.HF_TOKEN`` uses ``${HF_TOKEN}`` in YAML or the ``HF_TOKEN`` env var.
    """
    raw = apply_models_secret_env(_load_yaml(path))

    global_raw = raw.get("global") or {}
    global_cfg = GlobalConfig(
        env=global_raw.get("env") or {},
        defaults=EngineConfig(**(global_raw.get("defaults") or {})),
    )

    models_raw = raw.get("models")
    if not models_raw or not isinstance(models_raw, dict):
        raise ConfigError("models.yaml must have a top-level 'models' mapping")

    models: dict[str, ModelConfig] = {}
    for key, m in models_raw.items():
        if not isinstance(m, dict):
            raise ConfigError(f"Model '{key}' must be a mapping")

        # Merge global defaults into engine config
        engine_raw = dict(global_cfg.defaults.model_dump())
        engine_raw.update(m.get("engine") or {})

        # Parse autoscaling (accept legacy upscale_delay_s from older models.yaml)
        autoscale_raw = dict(m.get("autoscaling") or {})
        if "upscale_delay_s" in autoscale_raw and "upscale_cooldown_s" not in autoscale_raw:
            autoscale_raw["upscale_cooldown_s"] = autoscale_raw.pop("upscale_delay_s")

        # Parse routing strategy
        routing_raw = m.get("routing_strategy", "lor")
        try:
            routing = RoutingStrategy(routing_raw)
        except ValueError:
            raise ConfigError(
                f"Model '{key}' has invalid routing_strategy='{routing_raw}'. "
                f"Valid: {[s.value for s in RoutingStrategy]}"
            )

        # Build the model config
        try:
            model_cfg = ModelConfig(
                name=key,
                served_model_name=m.get("served_model_name", key),
                hf_model=m["hf_model"],
                tp=m.get("tp", m.get("min_tp", 1)),
                node_type_required=m.get("node_type_required"),
                node_types_allowed=m.get("node_types_allowed") or [],
                routing_strategy=routing,
                hf_token=m.get("hf_token") or global_cfg.env.get("HF_TOKEN"),
                engine=EngineConfig(**engine_raw),
                autoscaling=AutoscalePolicy(**autoscale_raw),
                deploy_priority=int(m.get("deploy_priority", 50)),
                nodes_allowed=m.get("nodes_allowed") or [],
                gpu_guard_exempt=bool(m.get("gpu_guard_exempt", False)),
            )
        except (ValidationError, KeyError) as e:
            raise ConfigError(f"Invalid model config '{key}': {e}") from e

        # Consistency check: if node_type_required is set but not in allowed list
        if (model_cfg.node_type_required
                and model_cfg.node_types_allowed
                and model_cfg.node_type_required not in model_cfg.node_types_allowed):
            raise ConfigError(
                f"Model '{key}': node_type_required='{model_cfg.node_type_required}' "
                f"is not in node_types_allowed={model_cfg.node_types_allowed}"
            )

        models[key] = model_cfg

    return global_cfg, models
