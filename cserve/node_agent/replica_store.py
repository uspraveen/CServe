"""Persistent replica memory on each GPU node.

Survives node-agent restarts so we can stop/reconcile processes the in-memory
Launcher no longer tracks.  Also used for orphan detection during heartbeats.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cserve.common.logging import get_logger

log = get_logger("replica_store")

DEFAULT_PATH = Path.home() / ".cserve" / "replicas.json"


@dataclass
class StoredReplica:
    replica_id: str
    model_name: str
    served_model_name: str
    hf_model: str
    gpu_ids: list[int]
    tp_size: int
    port: int
    pid: int
    node_name: str
    launched_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StoredReplica:
        return cls(
            replica_id=data["replica_id"],
            model_name=data["model_name"],
            served_model_name=data.get("served_model_name", data["model_name"]),
            hf_model=data.get("hf_model", ""),
            gpu_ids=[int(x) for x in data.get("gpu_ids", [])],
            tp_size=int(data.get("tp_size", 1)),
            port=int(data.get("port", 0)),
            pid=int(data.get("pid", 0)),
            node_name=data.get("node_name", ""),
            launched_at=float(data.get("launched_at", 0)),
        )


class ReplicaStore:
    def __init__(self, path: Path | None = None, node_name: str = "") -> None:
        self.path = path or DEFAULT_PATH
        self.node_name = node_name
        self._replicas: dict[str, StoredReplica] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._replicas = {}
            return
        try:
            raw = json.loads(self.path.read_text())
            self._replicas = {
                rid: StoredReplica.from_dict(entry)
                for rid, entry in raw.get("replicas", {}).items()
            }
        except Exception as e:
            log.warning("failed to load replica store", path=str(self.path), error=str(e))
            self._replicas = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "node_name": self.node_name,
                "updated_at": time.time(),
                "replicas": {rid: r.to_dict() for rid, r in self._replicas.items()},
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.path)
        except Exception as e:
            log.warning("failed to save replica store", path=str(self.path), error=str(e))

    def upsert(self, replica: StoredReplica) -> None:
        self._replicas[replica.replica_id] = replica
        self.save()

    def remove(self, replica_id: str) -> StoredReplica | None:
        removed = self._replicas.pop(replica_id, None)
        if removed:
            self.save()
        return removed

    def get(self, replica_id: str) -> StoredReplica | None:
        return self._replicas.get(replica_id)

    def all(self) -> list[StoredReplica]:
        return list(self._replicas.values())

    def clear(self) -> None:
        self._replicas.clear()
        self.save()
