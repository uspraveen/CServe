"""GPU Monitor — scrapes nvidia-smi and detects zombie processes.

Runs on each GPU node.  Provides:
  1. Per-GPU metrics (memory, utilization, temperature) via nvidia-smi.
  2. Detection of zombie GPU processes (processes using GPU memory
     that don't belong to any known replica).
  3. Kill API for cleaning up GPU processes.
"""

from __future__ import annotations

import asyncio
import os
import signal
import xml.etree.ElementTree as ET

from cserve.common.logging import get_logger
from cserve.common.models import GpuInfo, GpuState

log = get_logger("gpu_monitor")


class GpuMonitor:
    def __init__(self, node_name: str) -> None:
        self.node_name = node_name

    async def query_gpus(self) -> list[GpuInfo]:
        """Query nvidia-smi for GPU status."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "-q", "-x",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            if proc.returncode != 0:
                log.error("nvidia-smi failed", exit_code=proc.returncode,
                          stderr=stderr.decode()[:200])
                return []

            return self._parse_nvidia_smi_xml(stdout.decode())
        except FileNotFoundError:
            log.warning("nvidia-smi not found — running without GPU monitoring")
            return []
        except TimeoutError:
            log.error("nvidia-smi timed out")
            return []
        except Exception as e:
            log.error("nvidia-smi query error", error=str(e))
            return []

    async def find_gpu_processes(self, gpu_indices: list[int] | None = None) -> list[dict]:
        """List processes using GPU memory.

        Returns list of dicts with keys: pid, gpu_index, name, used_memory_mb.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,name,used_memory",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            if proc.returncode != 0:
                return []

            gpu_uuid_to_index = await self._gpu_uuid_map()
            processes = []

            for line in stdout.decode().strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue

                pid = int(parts[0])
                gpu_uuid = parts[1]
                name = parts[2]
                mem = int(parts[3]) if parts[3].isdigit() else 0
                gpu_idx = gpu_uuid_to_index.get(gpu_uuid, -1)

                if gpu_indices and gpu_idx not in gpu_indices:
                    continue

                processes.append({
                    "pid": pid,
                    "gpu_index": gpu_idx,
                    "name": name,
                    "used_memory_mb": mem,
                })

            return processes
        except Exception as e:
            log.error("failed to list GPU processes", error=str(e))
            return []

    async def find_zombies(self, known_pids: set[int]) -> list[dict]:
        """Find GPU processes not belonging to any known replica."""
        all_procs = await self.find_gpu_processes()
        return [p for p in all_procs if p["pid"] not in known_pids]

    async def kill_processes(
        self, gpu_indices: list[int], vllm_only: bool = False,
    ) -> list[int]:
        """Kill processes on specified GPUs. Returns list of killed PIDs."""
        processes = await self.find_gpu_processes(gpu_indices)
        killed = []

        for p in processes:
            if vllm_only:
                name_lower = p["name"].lower()
                if "vllm" not in name_lower and "worker_tp" not in name_lower:
                    continue

            pid = p["pid"]
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
                log.info("killed GPU process", pid=pid, gpu=p["gpu_index"],
                         name=p["name"], memory_mb=p["used_memory_mb"])
            except ProcessLookupError:
                pass
            except PermissionError:
                log.warning("permission denied killing process", pid=pid)

        return killed

    async def find_vllm_api_server_pids(
        self, gpu_indices: list[int] | None = None,
    ) -> list[dict]:
        """Return api_server PIDs on GPUs with parsed served-model-name."""
        processes = await self.find_gpu_processes(gpu_indices)
        out: list[dict] = []

        def _cmdline(pid: int) -> str:
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    return f.read().decode("utf-8", errors="replace").replace("\x00", " ")
            except OSError:
                return ""

        seen: set[int] = set()
        for p in processes:
            pid = int(p["pid"])
            if pid in seen:
                continue
            cmd = await asyncio.to_thread(_cmdline, pid)
            if "vllm.entrypoints" not in cmd:
                # Walk up to api_server parent (workers report as VLLM::Worker_TP*)
                parent = await asyncio.to_thread(self._parent_pid, pid)
                depth = 0
                while parent and parent > 1 and depth < 8:
                    cmd = await asyncio.to_thread(_cmdline, parent)
                    if "vllm.entrypoints" in cmd:
                        pid = parent
                        break
                    parent = await asyncio.to_thread(self._parent_pid, parent)
                    depth += 1
                else:
                    continue
            seen.add(pid)
            served = ""
            if "--served-model-name" in cmd:
                parts = cmd.split("--served-model-name", 1)[1].strip().split()
                if parts:
                    served = parts[0]
            out.append({
                "pid": pid,
                "gpu_index": p["gpu_index"],
                "served_model_name": served,
                "cmdline": cmd[:200],
            })
        return out

    @staticmethod
    def _parent_pid(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        return int(line.split()[1])
        except OSError:
            pass
        return 0

    async def kill_untracked_vllm_api_servers(
        self,
        gpu_indices: list[int],
        tracked_api_pids: set[int],
    ) -> list[int]:
        """SIGKILL vLLM api_server roots on these GPUs that are not in tracked_api_pids."""
        import signal

        killed: list[int] = []
        for entry in await self.find_vllm_api_server_pids(gpu_indices):
            pid = int(entry["pid"])
            if pid in tracked_api_pids:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
                log.warning(
                    "reconcile: killed untracked vLLM api_server",
                    pid=pid,
                    model=entry.get("served_model_name"),
                    gpu=entry.get("gpu_index"),
                )
            except ProcessLookupError:
                pass
            except PermissionError:
                log.warning("permission denied killing untracked vLLM", pid=pid)
        return killed

    async def kill_vllm_api_server_on_gpus(self, gpu_indices: list[int]) -> list[int]:
        """SIGKILL only PIDs that look like `python -m vllm.entrypoints...` on these GPUs.

        Safer than kill_processes(vllm_only=True) on shared clusters, where many
        jobs appear as generic "python" in nvidia-smi.
        """
        processes = await self.find_gpu_processes(gpu_indices)
        killed: list[int] = []

        def _cmdline(pid: int) -> bytes:
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    return f.read()
            except OSError:
                return b""

        for p in processes:
            pid = int(p["pid"])
            data = await asyncio.to_thread(_cmdline, pid)
            if b"vllm.entrypoints" not in data:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
                log.info(
                    "killed orphan vLLM api_server",
                    pid=pid, gpu=p["gpu_index"], name=p["name"],
                )
            except ProcessLookupError:
                pass
            except PermissionError:
                log.warning("permission denied killing orphan vLLM", pid=pid)

        return killed

    async def _gpu_uuid_map(self) -> dict[str, int]:
        """Build a mapping from GPU UUID to GPU index."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            result = {}
            for line in stdout.decode().strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    result[parts[1]] = int(parts[0])
            return result
        except Exception:
            return {}

    @staticmethod
    def _parse_nvidia_smi_xml(xml_text: str) -> list[GpuInfo]:
        """Parse nvidia-smi -q -x output into GpuInfo objects."""
        gpus = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            log.error("failed to parse nvidia-smi XML", error=str(e))
            return []

        for i, gpu_elem in enumerate(root.findall("gpu")):
            try:
                uuid = _xml_text(gpu_elem, "uuid", "")
                name = _xml_text(gpu_elem, "product_name", "")

                fb = gpu_elem.find("fb_memory_usage")
                mem_used = _parse_mib(fb, "used") if fb is not None else 0
                mem_total = _parse_mib(fb, "total") if fb is not None else 0

                util_elem = gpu_elem.find("utilization")
                gpu_util = _parse_pct(util_elem, "gpu_util") if util_elem is not None else 0.0

                temp_elem = gpu_elem.find("temperature")
                temp = _parse_temp(temp_elem, "gpu_temp") if temp_elem is not None else 0.0

                minor = _xml_text(gpu_elem, "minor_number", str(i))
                index = int(minor) if minor.isdigit() else i

                gpus.append(GpuInfo(
                    index=index,
                    uuid=uuid,
                    name=name,
                    memory_used_mb=mem_used,
                    memory_total_mb=mem_total,
                    utilization_pct=gpu_util,
                    temperature_c=temp,
                    state=GpuState.FREE,
                ))
            except Exception as e:
                log.warning("failed to parse GPU entry", index=i, error=str(e))

        return gpus


def _xml_text(elem, tag: str, default: str = "") -> str:
    child = elem.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return default


def _parse_mib(parent, tag: str) -> int:
    text = _xml_text(parent, tag)
    return int(text.split()[0]) if text and text.split()[0].isdigit() else 0


def _parse_pct(parent, tag: str) -> float:
    text = _xml_text(parent, tag)
    cleaned = text.replace("%", "").strip() if text else ""
    return float(cleaned) if cleaned and cleaned.replace(".", "").isdigit() else 0.0


def _parse_temp(parent, tag: str) -> float:
    text = _xml_text(parent, tag)
    cleaned = text.replace("C", "").strip() if text else ""
    return float(cleaned) if cleaned and cleaned.replace(".", "").isdigit() else 0.0
