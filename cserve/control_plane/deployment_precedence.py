"""Deployment precedence — ordered model rollout on shared GPU clusters.

Lower ``deploy_priority`` values deploy first (e.g. 10 before 20).
A model will not scale up (including to meet ``min_replicas``) until every
higher-precedence model has at least ``min_replicas`` in READY state.
"""

from __future__ import annotations

from cserve.common.models import ModelConfig, ReplicaStatus


def models_by_deploy_priority(
    models_config: dict[str, ModelConfig],
) -> list[tuple[str, ModelConfig]]:
    """Return (name, cfg) pairs sorted by deploy_priority ascending."""
    return sorted(
        models_config.items(),
        key=lambda item: (item[1].deploy_priority, item[0]),
    )


def ready_replica_count(registry, model_name: str) -> int:
    return sum(
        1
        for r in registry.get_replicas_for_model(model_name)
        if r.status == ReplicaStatus.READY
    )


def _resource_scopes_overlap(a: ModelConfig, b: ModelConfig) -> bool:
    """Return True when two models can contend for the same placement pool."""
    if a.nodes_allowed and b.nodes_allowed:
        if set(a.nodes_allowed).isdisjoint(b.nodes_allowed):
            return False

    def gpu_types(cfg: ModelConfig) -> set[str]:
        if cfg.node_type_required:
            return {cfg.node_type_required}
        return set(cfg.node_types_allowed)

    a_types = gpu_types(a)
    b_types = gpu_types(b)
    if a_types and b_types and a_types.isdisjoint(b_types):
        return False

    return True


def precedence_blocks_scale_up(
    model_name: str,
    registry,
    models_config: dict[str, ModelConfig],
) -> tuple[bool, str | None]:
    """True if this model must wait for higher-precedence models."""
    cfg = models_config.get(model_name)
    if not cfg:
        return False, None

    my_priority = cfg.deploy_priority
    for other_name, other_cfg in models_config.items():
        if other_name == model_name:
            continue
        if other_cfg.autoscaling.min_replicas <= 0:
            continue
        if other_cfg.deploy_priority >= my_priority:
            continue
        if not _resource_scopes_overlap(cfg, other_cfg):
            continue
        ready = ready_replica_count(registry, other_name)
        if ready < other_cfg.autoscaling.min_replicas:
            return True, other_name
    return False, None
