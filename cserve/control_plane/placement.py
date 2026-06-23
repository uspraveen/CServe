"""GPU-aware placement algorithm.

Given a model that needs N GPUs of a specific type, find the best node
and GPU set to place it on.

Placement strategy:
  1. Filter nodes by GPU type: if ``node_types_allowed`` is non-empty, only those
     types are eligible. Otherwise require ``node_type_required`` (exact match).
  2. Filter to online nodes only.
  3. For each candidate node, find contiguous free GPU sets of size >= tp.
  4. Score candidates by:
     - Prefer nodes with fewer existing replicas (spread load).
     - Prefer nodes with more free GPUs (leave room for future scaling).
     - Prefer nodes where the free GPUs are contiguous (better NVLink perf).
  5. Return the best (node, gpu_ids) pair, or None if no placement is possible.

This is intentionally simple and deterministic.  We're not solving a
bin-packing NP-hard problem — we're placing a known workload on a small
cluster where GPU types are homogeneous per node.
"""

from __future__ import annotations

from dataclasses import dataclass

from cserve.common.logging import get_logger
from cserve.common.models import GpuState, ModelConfig, NodeStatus

log = get_logger("placement")


@dataclass
class PlacementResult:
    node_name: str
    gpu_indices: list[int]
    score: float


def find_placement(
    model: ModelConfig,
    nodes: list,  # list[NodeState] — imported type to avoid circular
    existing_replica_counts: dict[str, int] | None = None,
    avoid_node: str | None = None,
) -> PlacementResult | None:
    """Find the best node and GPUs for a new replica of `model`.

    Args:
        model: the model config (carries tp, node_type_required, etc.)
        nodes: list of NodeState objects from the registry.
        existing_replica_counts: {node_name: count} of existing replicas
            for this model (used to spread replicas across nodes).
        avoid_node: Prefer other nodes when any alternative can satisfy tp/GPU type.

    Returns:
        PlacementResult with the chosen node and GPU indices, or None.
    """
    tp = model.tp
    required_type = model.node_type_required
    allowed_types = set(model.node_types_allowed) if model.node_types_allowed else None
    existing = existing_replica_counts or {}

    candidates: list[PlacementResult] = []

    allowed_nodes = set(model.nodes_allowed) if model.nodes_allowed else None

    for node in nodes:
        # Filter: node must be online
        if node.status == NodeStatus.OFFLINE:
            continue

        if allowed_nodes and node.name not in allowed_nodes:
            continue

        # Filter: GPU type — whitelist ``node_types_allowed`` when set (authoritative).
        # If unset, fall back to exact ``node_type_required`` only.
        if allowed_types:
            if node.gpu_type not in allowed_types:
                continue
        elif required_type and node.gpu_type != required_type:
            continue

        # Find free GPUs on this node (registry FREE + physically empty enough)
        gpu_util = float(model.engine.gpu_memory_utilization)
        free_indices = sorted(
            g.index for g in node.gpus
            if g.state == GpuState.FREE and _gpu_has_launch_headroom(g, gpu_util)
        )

        if len(free_indices) < tp:
            continue

        # Find contiguous groups of free GPUs
        groups = _find_contiguous_groups(free_indices, tp)
        if not groups:
            # Fall back to any combination of tp free GPUs
            groups = [free_indices[:tp]]

        for gpu_group in groups:
            score = _score_placement(
                node=node,
                gpu_group=gpu_group,
                total_free=len(free_indices),
                existing_replicas=existing.get(node.name, 0),
                contiguous=_is_contiguous(gpu_group),
            )
            candidates.append(PlacementResult(
                node_name=node.name,
                gpu_indices=gpu_group,
                score=score,
            ))

    if not candidates:
        log.warning("no placement found",
                    model=model.name, tp=tp,
                    required_type=required_type)
        return None

    if avoid_node:
        alt = [c for c in candidates if c.node_name != avoid_node]
        if alt:
            candidates = alt
            log.info("placement excluding node (retry / migration)",
                     model=model.name, avoid_node=avoid_node)

    # Sort by score descending (higher = better)
    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    log.info("placement found",
             model=model.name, node=best.node_name,
             gpus=best.gpu_indices, score=f"{best.score:.2f}")
    return best


def _find_contiguous_groups(indices: list[int], size: int) -> list[list[int]]:
    """Find all contiguous subsequences of length `size` in sorted indices."""
    groups: list[list[int]] = []
    for i in range(len(indices) - size + 1):
        group = indices[i : i + size]
        if _is_contiguous(group):
            groups.append(group)
    return groups


def _gpu_has_launch_headroom(gpu, gpu_util: float) -> bool:
    """Use heartbeat memory stats so placement does not trust registry FREE alone."""
    total = int(gpu.memory_total_mb or 0)
    used = int(gpu.memory_used_mb or 0)
    if total <= 0:
        return True
    # Foreign jobs / stale usage: more than 2 GiB or 8% of VRAM in use.
    if used > max(2048, int(total * 0.08)):
        return False
    needed = total * gpu_util
    free = total - used
    return free >= needed * 0.85


def _is_contiguous(indices: list[int]) -> bool:
    if len(indices) <= 1:
        return True
    return all(indices[i + 1] - indices[i] == 1 for i in range(len(indices) - 1))


def _score_placement(
    node,
    gpu_group: list[int],
    total_free: int,
    existing_replicas: int,
    contiguous: bool,
) -> float:
    """Score a candidate placement.  Higher is better.

    Factors (weighted):
      - Spread: fewer existing replicas of this model on this node = better.
      - Headroom: more free GPUs remaining after placement = better.
      - Contiguity: contiguous GPUs = better NVLink performance.
      - Launch reliability: deprioritize nodes with recent launch failures
        (circuit may not be open yet — e.g. after one failure).
    """
    tp = len(gpu_group)

    # Spread score: penalize nodes that already have replicas of this model.
    # Range: 0 (many replicas) to 10 (no replicas)
    spread = max(0.0, 10.0 - existing_replicas * 3.0)

    # Headroom score: GPUs remaining free after this placement.
    # Range: 0 to 5
    remaining = total_free - tp
    headroom = min(5.0, remaining * 1.0)

    # Contiguity bonus: +3 if GPUs are contiguous (NVLink benefit)
    contiguity = 3.0 if contiguous else 0.0

    failures = int(getattr(node, "consecutive_launch_failures", 0) or 0)
    launch_penalty = min(6.0, failures * 2.0)

    # Prefer nodes whose FREE GPUs report low VRAM use (avoids stale / foreign jobs).
    free_gpus = [g for g in node.gpus if g.index in gpu_group]
    if free_gpus:
        avg_used = sum(int(g.memory_used_mb or 0) for g in free_gpus) / len(free_gpus)
        memory_penalty = min(12.0, avg_used / 3000.0)
    else:
        memory_penalty = 0.0

    return spread + headroom + contiguity - launch_penalty - memory_penalty
