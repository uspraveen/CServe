"""vLLM Process Launcher — manages vLLM subprocesses on a single node.

Responsibilities:
  - Build the vllm serve command with the correct arguments.
  - Start the subprocess with pinned CUDA_VISIBLE_DEVICES.
  - Wait for the HTTP health endpoint to become ready.
  - Graceful stop (SIGTERM → timeout → SIGKILL).
  - Post-mortem GPU cleanup via nvidia-smi.

GPU Lifecycle Safety (inspired by cosmos-gpu-cluster-new):
  vLLM spawns child processes (EngineCore, Worker_TP0..N) that can survive
  parent death because vLLM uses multiprocessing with start_method='spawn'
  or 'forkserver', which double-forks.  These orphans get reparented to init
  and hold GPU memory indefinitely.

  To prevent zombie GPU accumulation we add 3 mandatory safety layers:

    1. PRE-LAUNCH SCRUB — Before every launch, nvidia-smi finds all PIDs
       occupying the target GPUs and kills any vLLM-related ones.  Then
       verifies free memory meets the model's requirements.

    2. POST-STOP SWEEP — After stopping a replica (graceful or forced), we
       query nvidia-smi for the target GPUs and SIGKILL any leftover PIDs.
       We loop with a timeout until the GPUs report clean.

    3. MEMORY GATE — After the scrub and before the actual Popen, we read
       memory.free per GPU and abort if it's below 90% of the model's
       gpu_memory_utilization * memory.total.  This prevents vLLM from
       crash-looping against full GPUs.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx

from cserve.common.logging import get_logger
from cserve.node_agent.replica_store import ReplicaStore, StoredReplica

log = get_logger("launcher")

HEALTH_POLL_INTERVAL_S = 2.0
HEALTH_TIMEOUT_S = 300.0
GRACEFUL_STOP_TIMEOUT_S = 30.0
GPU_CLEANUP_TIMEOUT_S = 15.0
GPU_CLEANUP_POLL_S = 2.0
# Do not reconcile-kill replicas the control plane just launched (registry lag).
RECONCILE_GRACE_S = 300.0


class ReplicaProcess:
    """Wrapper around a vLLM subprocess."""
    __slots__ = (
        "replica_id", "model_name", "hf_model", "served_model_name",
        "gpu_ids", "tp_size", "port", "proc", "pid",
        "started_at", "variant", "env_override", "stdout_log_path",
    )

    def __init__(
        self,
        replica_id: str,
        model_name: str,
        hf_model: str,
        served_model_name: str,
        gpu_ids: list[int],
        tp_size: int,
        port: int,
        variant: str = "default",
        env_override: dict[str, str] | None = None,
    ) -> None:
        self.replica_id = replica_id
        self.model_name = model_name
        self.hf_model = hf_model
        self.served_model_name = served_model_name
        self.gpu_ids = gpu_ids
        self.tp_size = tp_size
        self.port = port
        self.variant = variant
        self.env_override = env_override or {}
        self.proc: subprocess.Popen | None = None
        self.pid: int = 0
        self.started_at: float = 0.0
        self.stdout_log_path: str = ""


class Launcher:
    """Manages vLLM replica processes on this node."""

    def __init__(self, node_name: str, node_host: str) -> None:
        self.node_name = node_name
        self.node_host = node_host
        self._replicas: dict[str, ReplicaProcess] = {}
        self.store = ReplicaStore(node_name=node_name)

    def get_replica(self, replica_id: str) -> ReplicaProcess | None:
        return self._replicas.get(replica_id)

    def all_replicas(self) -> list[ReplicaProcess]:
        return list(self._replicas.values())

    async def launch(
        self,
        replica_id: str,
        model_name: str,
        hf_model: str,
        served_model_name: str,
        variant: str,
        gpu_ids: list[int],
        tp_size: int,
        port: int = 0,
        engine_args: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ReplicaProcess:
        """Launch a vLLM subprocess with full GPU safety lifecycle."""
        if replica_id in self._replicas:
            raise RuntimeError(f"Replica {replica_id} already running on this node")

        if port == 0:
            port = _find_free_port()

        engine_args = engine_args or {}
        env_vars = env_vars or {}

        rp = ReplicaProcess(
            replica_id=replica_id,
            model_name=model_name,
            hf_model=hf_model,
            served_model_name=served_model_name,
            gpu_ids=gpu_ids,
            tp_size=tp_size,
            port=port,
            variant=variant,
            env_override=env_vars,
        )

        # ── SAFETY LAYER 1: Pre-launch GPU scrub ──
        killed = await _scrub_gpus(gpu_ids, label=f"pre-launch:{replica_id}")
        if killed:
            await asyncio.sleep(3)

        # ── SAFETY LAYER 2: Memory gate ──
        gpu_util = float(engine_args.get("gpu_memory_utilization", "0.70"))
        mem_ok = await _check_gpu_memory(gpu_ids, gpu_util, replica_id)
        if not mem_ok:
            log.warning("insufficient GPU memory after scrub, running aggressive cleanup",
                        replica=replica_id, gpus=gpu_ids)
            await _scrub_gpus(gpu_ids, label=f"mem-pressure:{replica_id}")
            await asyncio.sleep(5)
            mem_ok = await _check_gpu_memory(gpu_ids, gpu_util, replica_id)
            if not mem_ok:
                raise RuntimeError(
                    f"GPUs {gpu_ids} still lack free memory after aggressive cleanup. "
                    f"Cannot launch replica {replica_id}."
                )

        # ── Launch ──
        cmd = self._build_command(rp, engine_args)
        env = self._build_env(rp)

        log.info("launching vLLM", replica=replica_id, model=hf_model,
                 gpus=gpu_ids, tp=tp_size, port=port, cmd=" ".join(cmd))

        log_dir = Path.home() / ".cserve" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log_path = log_dir / f"vllm-{replica_id}.log"
        rp.stdout_log_path = str(stdout_log_path)
        with stdout_log_path.open("ab", buffering=0) as stdout_log:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        rp.proc = proc
        rp.pid = proc.pid
        rp.started_at = time.time()
        self._replicas[replica_id] = rp
        self.store.upsert(StoredReplica(
            replica_id=replica_id,
            model_name=model_name,
            served_model_name=served_model_name,
            hf_model=hf_model,
            gpu_ids=list(gpu_ids),
            tp_size=tp_size,
            port=port,
            pid=proc.pid,
            node_name=self.node_name,
            launched_at=rp.started_at,
        ))

        log.info("vLLM process started", replica=replica_id, pid=proc.pid)
        return rp

    async def wait_for_health(self, replica_id: str) -> bool:
        """Poll the vLLM /health endpoint until it responds 200."""
        rp = self._replicas.get(replica_id)
        if not rp:
            return False

        url = f"http://127.0.0.1:{rp.port}/health"
        deadline = time.time() + HEALTH_TIMEOUT_S

        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            while time.time() < deadline:
                if rp.proc and rp.proc.poll() is not None:
                    self._log_process_output(rp)
                    return False

                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        startup_s = time.time() - rp.started_at
                        log.info("vLLM replica ready", replica=replica_id,
                                 startup_s=f"{startup_s:.1f}")
                        return True
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass

                await asyncio.sleep(HEALTH_POLL_INTERVAL_S)

        log.error("vLLM health timeout", replica=replica_id, timeout_s=HEALTH_TIMEOUT_S)
        return False

    @staticmethod
    def _log_process_output(rp: ReplicaProcess) -> None:
        """Read and log the last output from a crashed vLLM process."""
        exit_code = rp.proc.returncode if rp.proc else -1
        tail = _tail_text_file(rp.stdout_log_path, max_lines=30)
        log.error(
            "vLLM process exited during startup",
            replica=rp.replica_id,
            exit_code=exit_code,
            output_tail=tail or "(no output captured)",
        )

    async def stop(self, replica_id: str, force: bool = False) -> bool:
        """Stop a vLLM replica with full post-stop GPU cleanup."""
        rp = self._replicas.pop(replica_id, None)
        stored = self.store.remove(replica_id)

        if not rp or not rp.proc:
            if stored and stored.gpu_ids:
                log.warning(
                    "stop: replica not in memory, scrubbing from persistent store",
                    replica=replica_id, gpus=stored.gpu_ids, pid=stored.pid,
                )
                if stored.pid:
                    _kill_process_tree(stored.pid)
                await _post_stop_gpu_sweep(stored.gpu_ids, label=f"stop-stored:{replica_id}")
            return True

        proc = rp.proc
        pid = proc.pid
        gpu_ids = rp.gpu_ids

        if force:
            log.info("force killing vLLM", replica=replica_id, pid=pid)
            _kill_process_tree(pid)
        else:
            log.info("stopping vLLM (graceful)", replica=replica_id, pid=pid)
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, proc.wait
                        ),
                        timeout=GRACEFUL_STOP_TIMEOUT_S,
                    )
                    log.info("vLLM stopped gracefully", replica=replica_id, pid=pid)
                except TimeoutError:
                    log.warning("graceful stop timed out, sending SIGKILL",
                                replica=replica_id, pid=pid)
                    _kill_process_tree(pid)

        # ── SAFETY LAYER 3: Post-stop GPU sweep ──
        await _post_stop_gpu_sweep(gpu_ids, label=f"post-stop:{replica_id}")
        return True

    async def stop_all(self) -> None:
        """Stop all replicas on this node."""
        for replica_id in list(self._replicas.keys()):
            await self.stop(replica_id, force=True)
        self.store.clear()

    async def reconcile_orphans(
        self,
        expected_replica_ids: set[str],
        gpu_monitor,
    ) -> dict:
        """Align local GPU processes with control-plane expected replica set.

        Kills vLLM that CServe no longer tracks and scrubs GPUs for stale store entries.
        """
        report: dict = {
            "removed_from_store": [],
            "stopped_untracked": [],
            "killed_untracked_pids": [],
        }
        managed = _managed_gpu_indices_from_env()
        now = time.time()

        for stored in list(self.store.all()):
            if stored.replica_id in expected_replica_ids:
                continue
            if stored.launched_at and (now - stored.launched_at) < RECONCILE_GRACE_S:
                continue
            log.warning(
                "reconcile: dropping replica absent from control plane",
                replica=stored.replica_id,
                model=stored.model_name,
                pid=stored.pid,
            )
            if stored.pid:
                _kill_process_tree(stored.pid)
            if stored.gpu_ids:
                await _scrub_gpus(stored.gpu_ids, label=f"reconcile:{stored.replica_id}")
            self.store.remove(stored.replica_id)
            report["removed_from_store"].append(stored.replica_id)

        for rp in list(self.all_replicas()):
            if rp.replica_id in expected_replica_ids:
                continue
            if rp.started_at and (now - rp.started_at) < RECONCILE_GRACE_S:
                continue
            log.warning(
                "reconcile: stopping local replica absent from control plane",
                replica=rp.replica_id,
            )
            await self.stop(rp.replica_id, force=True)
            report["stopped_untracked"].append(rp.replica_id)

        tracked_pids: set[int] = set()
        for rp in self.all_replicas():
            if rp.pid:
                tracked_pids.add(rp.pid)
        for stored in self.store.all():
            if stored.pid:
                tracked_pids.add(stored.pid)

        # Untracked VLLM workers/api_server on managed GPUs (e.g. parent died,
        # Worker_TP* left ~33GB allocated). Safe when no live replica owns those GPUs.
        report["killed_untracked_pids"] = await self._scrub_untracked_vllm_workers(
            tracked_pids,
        )

        return report

    async def _scrub_untracked_vllm_workers(self, tracked_pids: set[int]) -> list[int]:
        """Kill vLLM processes on managed GPUs not tied to a live local replica."""
        managed = _managed_gpu_indices_from_env()
        if not managed:
            return []

        alive_gpu_sets: list[set[int]] = []
        for rp in self.all_replicas():
            if self.is_alive(rp.replica_id) and rp.gpu_ids:
                alive_gpu_sets.append(set(rp.gpu_ids))

        gpus_to_scrub: list[int] = []
        for gpu in managed:
            if any(gpu in s for s in alive_gpu_sets):
                continue
            gpus_to_scrub.append(gpu)

        if not gpus_to_scrub:
            return []

        pids_info = await _get_gpu_pids(gpus_to_scrub)
        orphan_pids = [
            pid for pid, name, _ in pids_info
            if pid not in tracked_pids and _is_vllm_process(pid, name)
        ]
        if not orphan_pids:
            return []

        log.warning(
            "reconcile: killing untracked vLLM on managed GPUs",
            gpus=gpus_to_scrub,
            pids=orphan_pids,
        )
        for pid in orphan_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        await asyncio.sleep(3)
        for pid in orphan_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        await asyncio.sleep(1)
        return orphan_pids

    def is_alive(self, replica_id: str) -> bool:
        """Check if a replica's process is still running."""
        rp = self._replicas.get(replica_id)
        if not rp or not rp.proc:
            return False
        return rp.proc.poll() is None

    @staticmethod
    def _build_command(rp: ReplicaProcess, engine_args: dict[str, str]) -> list[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", rp.hf_model,
            "--served-model-name", rp.served_model_name,
            "--host", "0.0.0.0",
            "--port", str(rp.port),
            "--tensor-parallel-size", str(rp.tp_size),
        ]

        for key, value in engine_args.items():
            if key.startswith("--"):
                flag = key
            else:
                flag = f"--{key.replace('_', '-')}"

            if value.lower() in ("true", "false"):
                if value.lower() == "true":
                    cmd.append(flag)
            elif value:
                cmd.extend([flag, value])

        return cmd

    @staticmethod
    def _build_env(rp: ReplicaProcess) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in rp.gpu_ids)
        env["VLLM_NO_USAGE_STATS"] = "1"
        env.update(rp.env_override)
        return env


def _managed_gpu_indices_from_env() -> list[int]:
    raw = os.environ.get("CSERVE_CUDA_DEVICES", "")
    if not raw:
        return []
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# GPU Safety Functions
# ─────────────────────────────────────────────────────────────────────────────

async def _get_gpu_pids(gpu_ids: list[int]) -> list[tuple[int, str, int]]:
    """Query nvidia-smi for processes on specific GPUs.

    Returns [(pid, process_name, gpu_index), ...].
    """
    gpu_list = ",".join(str(g) for g in gpu_ids)
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            f"--id={gpu_list}",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        results = []
        for line in stdout.decode().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                results.append((int(parts[0]), parts[1], 0))
        return results
    except Exception as e:
        log.warning("nvidia-smi query failed", error=str(e))
        return []


def _is_vllm_process(pid: int, name: str) -> bool:
    """Check if a process is vLLM-related (by name or cmdline)."""
    name_lower = name.lower()
    if "vllm" in name_lower or "worker_tp" in name_lower:
        return True

    try:
        result = subprocess.run(
            ["ps", "-o", "args=", "-p", str(pid)],
            capture_output=True, text=True, timeout=3,
        )
        cmdline = result.stdout.strip().lower()
        return "vllm" in cmdline
    except Exception:
        return False


def _tail_text_file(path: str, *, max_lines: int) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536), os.SEEK_SET)
            text = f.read().decode("utf-8", errors="replace")
        return "\n".join(text.strip().splitlines()[-max_lines:])
    except Exception:
        return ""


def read_replica_log(
    path: str,
    *,
    offset: int = 0,
    max_lines: int = 200,
    max_bytes: int = 262144,
) -> dict:
    """Read vLLM log file from a worker node.

    offset=0: return the last ``max_lines`` lines (initial dashboard load).
    offset>0: return bytes from ``offset`` to EOF (incremental tail for live view).
    """
    empty = {"text": "", "offset": 0, "size": 0, "log_path": path or "", "truncated": False}
    if not path:
        return empty
    try:
        p = Path(path)
        if not p.is_file():
            return empty
        size = p.stat().st_size
        if offset < 0 or offset > size:
            offset = 0
        if offset == 0:
            text = _tail_text_file(path, max_lines=max_lines)
            return {
                "text": text,
                "offset": size,
                "size": size,
                "log_path": path,
                "truncated": False,
            }
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read(max_bytes)
        return {
            "text": chunk.decode("utf-8", errors="replace"),
            "offset": size,
            "size": size,
            "log_path": path,
            "truncated": len(chunk) >= max_bytes,
        }
    except Exception:
        return {**empty, "offset": offset}


async def _scrub_gpus(gpu_ids: list[int], label: str = "scrub") -> list[int]:
    """Kill all vLLM-related processes on the given GPUs.

    Uses nvidia-smi as the source of truth (not process trees).
    SIGTERM first, wait, then SIGKILL survivors.
    """
    pids_info = await _get_gpu_pids(gpu_ids)
    if not pids_info:
        return []

    vllm_pids = [
        pid for pid, name, _ in pids_info
        if _is_vllm_process(pid, name)
    ]

    if not vllm_pids:
        return []

    log.info("scrubbing vLLM processes from GPUs",
             label=label, gpus=gpu_ids, pids=vllm_pids)

    for pid in vllm_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    await asyncio.sleep(5)

    for pid in vllm_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    await asyncio.sleep(2)

    remaining = await _get_gpu_pids(gpu_ids)
    remaining_vllm = [
        pid for pid, name, _ in remaining
        if _is_vllm_process(pid, name)
    ]
    if remaining_vllm:
        log.warning("force killing stubborn GPU processes",
                     label=label, pids=remaining_vllm)
        for pid in remaining_vllm:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        await asyncio.sleep(2)

    log.info("GPU scrub complete", label=label, gpus=gpu_ids,
             killed=len(vllm_pids))
    return vllm_pids


async def _post_stop_gpu_sweep(gpu_ids: list[int], label: str = "sweep") -> None:
    """After stopping a replica, ensure the GPUs are actually clean.

    Polls nvidia-smi with a timeout to confirm no vLLM processes remain.
    """
    deadline = time.time() + GPU_CLEANUP_TIMEOUT_S
    attempt = 0

    while time.time() < deadline:
        pids_info = await _get_gpu_pids(gpu_ids)
        vllm_pids = [
            pid for pid, name, _ in pids_info
            if _is_vllm_process(pid, name)
        ]

        if not vllm_pids:
            if attempt > 0:
                log.info("post-stop GPU sweep confirmed clean",
                         label=label, gpus=gpu_ids, attempts=attempt)
            return

        attempt += 1
        log.warning("orphaned vLLM processes found after stop",
                     label=label, gpus=gpu_ids, pids=vllm_pids,
                     attempt=attempt)

        for pid in vllm_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        await asyncio.sleep(GPU_CLEANUP_POLL_S)

    log.error("post-stop GPU sweep timed out — GPUs may still be occupied",
              label=label, gpus=gpu_ids, timeout_s=GPU_CLEANUP_TIMEOUT_S)


async def _check_gpu_memory(
    gpu_ids: list[int], gpu_util: float, replica_id: str,
) -> bool:
    """Verify that each target GPU has enough free memory for the model.

    Returns True if all GPUs pass, False if any fail.
    """
    gpu_list = ",".join(str(g) for g in gpu_ids)
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            f"--id={gpu_list}",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)

        for line in stdout.decode().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue

            try:
                gpu_idx = int(parts[0])
                free_mib = float(parts[1])
                total_mib = float(parts[2])
            except ValueError:
                continue

            needed_mib = total_mib * gpu_util
            threshold = needed_mib * 0.85

            if free_mib < threshold:
                log.warning("GPU lacks free memory",
                            replica=replica_id, gpu=gpu_idx,
                            free_mib=f"{free_mib:.0f}",
                            needed_mib=f"{needed_mib:.0f}",
                            total_mib=f"{total_mib:.0f}",
                            gpu_util=f"{gpu_util:.0%}")
                return False

        return True
    except Exception as e:
        log.warning("GPU memory check failed, proceeding anyway",
                    replica=replica_id, error=str(e))
        return True


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its children via process group."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        log.warning("permission denied killing process tree", pid=pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
