"""Resolve secrets from the environment — keep YAML in git without real tokens.

YAML may use full-string placeholders: ``${HF_TOKEN}``, ``${CSERVE_SSH_PASSWORD}``.
After substitution, known secret keys are filled from the environment when still empty.
"""

from __future__ import annotations

import os
import re
from typing import Any

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

# YAML key path → environment variable (used when value is empty after placeholder pass)
_CLUSTER_SECRET_ENV: dict[tuple[str, ...], str] = {
    ("ssh", "password"): "CSERVE_SSH_PASSWORD",
    ("redis", "password"): "REDIS_PASSWORD",
}

_MODELS_SECRET_ENV: dict[tuple[str, ...], str] = {
    ("global", "env", "HF_TOKEN"): "HF_TOKEN",
}


def resolve_env_placeholders(obj: Any) -> Any:
    """Recursively replace ``${VAR}`` strings with ``os.environ[VAR]`` (or '')."""
    if isinstance(obj, dict):
        return {k: resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_placeholders(v) for v in obj]
    if isinstance(obj, str):
        m = _ENV_REF.match(obj.strip())
        if m:
            return os.environ.get(m.group(1), "")
        return obj
    return obj


def _set_nested(d: dict[str, Any], path: tuple[str, ...], value: str) -> None:
    cur: dict[str, Any] = d
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _inject_env_secrets(raw: dict[str, Any], mapping: dict[tuple[str, ...], str]) -> None:
    for path, env_name in mapping.items():
        cur: Any = raw
        for key in path[:-1]:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(key)
        if not isinstance(cur, dict):
            continue
        leaf = path[-1]
        existing = cur.get(leaf)
        if existing:
            continue
        env_val = os.environ.get(env_name, "")
        if env_val:
            _set_nested(raw, path, env_val)


def apply_cluster_secret_env(raw: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_env_placeholders(raw)
    _inject_env_secrets(resolved, _CLUSTER_SECRET_ENV)
    return resolved


def apply_models_secret_env(raw: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_env_placeholders(raw)
    _inject_env_secrets(resolved, _MODELS_SECRET_ENV)
    return resolved
