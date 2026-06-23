"""UI-tunable config: merge YAML base with SQLite overlays.

Startup policy (see ``config_sync.should_apply_yaml_on_startup``):
  - YAML newer than SQLite → file wins (no stale overlay after hand-edits).
  - Else SQLite overlay wins (dashboard saves survive restart).

``PUT /admin/config`` persists to SQLite and mirrors tunables to YAML.
``POST /admin/config/sync-from-yaml`` forces a reload from disk anytime.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from cserve.common.logging import get_logger
from cserve.common.models import (
    AutoscalePolicy,
    EngineConfig,
    ModelConfig,
    SafetyConfig,
)

log = get_logger("ui_tuning")

# Fields exposed in GET/PUT /admin/config (must match server.py)
SAFETY_UI_FIELDS: tuple[str, ...] = (
    "gpu_memory_limit",
    "gpu_warn_threshold",
    "gpu_danger_threshold",
    "gpu_compute_sustain_threshold",
    "gpu_compute_sustain_duration_s",
    "guard_mitigation_window_s",
    "guard_check_interval_s",
)

ENGINE_UI_FIELDS: tuple[str, ...] = (
    "max_num_seqs",
    "gpu_memory_utilization",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "max_model_len",
)

AUTOSCALE_UI_FIELDS: tuple[str, ...] = (
    "min_replicas",
    "max_replicas",
    "allow_scale_to_zero",
    "target_inflight",
    "idle_timeout_s",
    "upscale_cooldown_s",
    "downscale_cooldown_s",
    "max_queue_depth",
    "replica_startup_timeout_s",
)


def safety_payload_from_config(safety: SafetyConfig) -> dict[str, Any]:
    return {k: getattr(safety, k) for k in SAFETY_UI_FIELDS}


def model_tuning_payload_from_config(cfg: ModelConfig) -> dict[str, Any]:
    return {
        "engine": {k: getattr(cfg.engine, k) for k in ENGINE_UI_FIELDS},
        "autoscaling": {k: getattr(cfg.autoscaling, k) for k in AUTOSCALE_UI_FIELDS},
    }


def apply_safety_payload(safety: SafetyConfig, payload: dict[str, Any]) -> SafetyConfig:
    """Return new SafetyConfig with payload fields overlaid (validated)."""
    d = safety.model_dump()
    for k, v in payload.items():
        if k in d:
            d[k] = v
    try:
        return SafetyConfig(**d)
    except ValidationError as e:
        log.warning("invalid safety overlay ignored", error=str(e))
        return safety


def apply_model_tuning_payload(cfg: ModelConfig, payload: dict[str, Any]) -> ModelConfig:
    """Return new ModelConfig with engine/autoscaling overlays (validated)."""
    eng = dict(cfg.engine.model_dump())
    if "engine" in payload and isinstance(payload["engine"], dict):
        for k, v in payload["engine"].items():
            if k in eng:
                eng[k] = v
    asc = dict(cfg.autoscaling.model_dump())
    if "autoscaling" in payload and isinstance(payload["autoscaling"], dict):
        for k, v in payload["autoscaling"].items():
            if k in asc:
                asc[k] = v
    try:
        return cfg.model_copy(
            update={
                "engine": EngineConfig(**eng),
                "autoscaling": AutoscalePolicy(**asc),
            },
        )
    except ValidationError as e:
        log.warning("invalid model tuning overlay ignored", model=cfg.name, error=str(e))
        return cfg


def decode_payload_json(raw: str) -> dict[str, Any]:
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}
