"""SSH-based node management for CServe.

Handles:
  1. GPU discovery (probe_node)  — SSH in and run nvidia-smi to enumerate GPUs.
  2. Agent deployment (deploy_agent) — rsync code, install deps, start node agent.
  3. Agent teardown (stop_agent) — gracefully kill the node agent process.

Uses asyncssh for all SSH operations and asyncio.create_subprocess_exec for
local rsync (which manages its own SSH connection using the system's key agent).

All operations are async and safe to call from the FastAPI event loop.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import asyncssh

from cserve.common.logging import get_logger
from cserve.common.models import SshConfig

log = get_logger("ssh_manager")


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class GpuProbeInfo:
    index: int
    name: str
    memory_total_mb: int
    utilization_pct: float = 0.0


@dataclass
class NodeProbeResult:
    gpus: list[GpuProbeInfo] = field(default_factory=list)
    hostname: str = ""
    os_info: str = ""
    error: str | None = None


@dataclass
class DeployResult:
    ok: bool
    log: list[str] = field(default_factory=list)
    error: str | None = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _conn_kwargs(ssh_cfg: SshConfig, host: str) -> dict:
    """Build asyncssh.connect() kwargs from SshConfig.

    Auth precedence:
      1. Password (if set) — overrides key-based auth entirely.
      2. Key file (if the path exists on disk).
      3. No explicit creds — asyncssh falls back to the SSH agent / default keys.
    """
    key_path = Path(ssh_cfg.key_path).expanduser()
    kwargs: dict = {
        "host": host,
        "username": ssh_cfg.username,
        "port": ssh_cfg.port,
        "known_hosts": None,          # accept any host key — internal cluster
        "connect_timeout": ssh_cfg.timeout_s,
        "login_timeout": ssh_cfg.timeout_s,
    }
    if ssh_cfg.password:
        kwargs["password"] = ssh_cfg.password
        kwargs["client_keys"] = []    # disable key-based auth when password is set
    elif key_path.exists():
        kwargs["client_keys"] = [str(key_path)]
    return kwargs


async def _run(conn: asyncssh.SSHClientConnection, cmd: str) -> tuple[int, str, str]:
    """Run a command on an existing SSH connection. Returns (rc, stdout, stderr)."""
    result = await conn.run(cmd, check=False)
    return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()


def _remote_path(p: str) -> str:
    """Convert a path for safe use inside a remote shell command.

    shlex.quote wraps values in single-quotes, which prevents the remote
    shell from expanding '~'. Replace leading ~ with $HOME so expansion
    works regardless of quoting style.
    """
    if p.startswith("~/"):
        return "$HOME/" + p[2:]
    if p == "~":
        return "$HOME"
    return p


# ─── Public API ───────────────────────────────────────────────────────────────

async def probe_node(host: str, ssh_cfg: SshConfig) -> NodeProbeResult:
    """SSH into *host* and discover available GPUs via nvidia-smi.

    Returns a NodeProbeResult with a list of GpuProbeInfo entries.
    On any error, returns NodeProbeResult(gpus=[], error=...).
    """
    try:
        async with asyncssh.connect(**_conn_kwargs(ssh_cfg, host)) as conn:
            # Hostname for display
            _, hostname, _ = await _run(conn, "hostname -s 2>/dev/null || echo unknown")

            # OS fingerprint
            _, os_info, _ = await _run(
                conn,
                "lsb_release -d 2>/dev/null | cut -d: -f2 || uname -sr",
            )

            # GPU inventory
            smi_cmd = (
                "nvidia-smi "
                "--query-gpu=index,name,memory.total,utilization.gpu "
                "--format=csv,noheader,nounits 2>/dev/null"
            )
            rc, stdout, stderr = await _run(conn, smi_cmd)

            if rc != 0 or not stdout:
                return NodeProbeResult(
                    hostname=hostname,
                    os_info=os_info.strip(),
                    error="nvidia-smi not available or no GPUs found on this host.",
                )

            gpus: list[GpuProbeInfo] = []
            for line in stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    idx = int(parts[0])
                    name = parts[1]
                    mem_mb = int(re.sub(r"[^\d]", "", parts[2])) if parts[2] else 0
                    util = float(re.sub(r"[^\d.]", "", parts[3])) if len(parts) > 3 and parts[3] else 0.0
                    gpus.append(GpuProbeInfo(index=idx, name=name, memory_total_mb=mem_mb, utilization_pct=util))
                except (ValueError, IndexError):
                    continue

            return NodeProbeResult(gpus=gpus, hostname=hostname.strip(), os_info=os_info.strip())

    except asyncssh.Error as e:
        return NodeProbeResult(error=f"SSH error: {e}")
    except TimeoutError:
        return NodeProbeResult(error=f"Connection to {host} timed out after {ssh_cfg.timeout_s:.0f}s")
    except Exception as e:  # noqa: BLE001
        return NodeProbeResult(error=f"Unexpected error: {e}")


async def deploy_agent(
    *,
    host: str,
    node_name: str,
    cuda_devices: str,
    agent_port: int,
    control_plane_url: str,
    transport: str = "http",
    ssh_cfg: SshConfig,
    sync_code: bool = True,
    local_cserve_src: str = "/home/services/CServe",
) -> DeployResult:
    """Deploy (or redeploy) the CServe node agent on *host*.

    Steps:
      1. (Optional) rsync CServe source to the remote host.
      2. pip install -e ~/CServe (idempotent).
      3. Kill any existing node agent.
      4. Start the node agent via nohup.

    Returns a DeployResult with a structured step log.
    """
    lines: list[str] = []

    def _log(msg: str) -> None:
        log.info(msg, host=host, node=node_name)
        lines.append(msg)

    try:
        # ── Step 1: rsync (run locally, SSH handles auth) ────────────────────
        if sync_code:
            _log(f"[1/4] Syncing CServe source to {ssh_cfg.username}@{host}:~/CServe/ …")

            # Build the ssh sub-command for rsync.
            # -o StrictHostKeyChecking=no : don't abort on host key mismatch
            # -o UserKnownHostsFile=/dev/null : don't read/write known_hosts at all
            # This prevents "REMOTE HOST IDENTIFICATION HAS CHANGED" errors when
            # cluster nodes are re-provisioned or their IPs change.
            ssh_opts = (
                f"ssh -p {ssh_cfg.port}"
                f" -o StrictHostKeyChecking=no"
                f" -o UserKnownHostsFile=/dev/null"
                f" -o LogLevel=ERROR"   # suppress the verbose SSH warnings
            )
            key_path = Path(ssh_cfg.key_path).expanduser()
            if not ssh_cfg.password and key_path.exists():
                ssh_opts += f" -i {key_path}"
            elif not ssh_cfg.password and not key_path.exists():
                _log(f"    ⚠  Key file {ssh_cfg.key_path} not found — will try SSH agent / default keys")

            rsync_cmd: list[str] = []

            if ssh_cfg.password:
                # Use sshpass for password-based rsync if available
                from shutil import which
                if which("sshpass"):
                    rsync_cmd = ["sshpass", "-p", ssh_cfg.password]
                    _log("    using sshpass for password-based sync")
                else:
                    _log("    ⚠  sshpass not found — falling back to asyncssh SFTP for code sync")
                    rsync_cmd = []  # signals we should use SFTP fallback below

            if rsync_cmd == [] and ssh_cfg.password:
                # SFTP fallback: use asyncssh to transfer a tar archive
                _log("    [sftp] creating archive on control plane …")
                try:
                    archive = "/tmp/_cserve_sync.tar.gz"
                    tar_proc = await asyncio.create_subprocess_exec(
                        "tar", "-czf", archive,
                        "--exclude=.git", "--exclude=__pycache__",
                        "--exclude=*.pyc", "--exclude=node_modules",
                        "--exclude=dashboard-ui/node_modules",
                        "--exclude=.ruff_cache",
                        "-C", local_cserve_src, ".",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                    await asyncio.wait_for(tar_proc.communicate(), timeout=60)
                    async with asyncssh.connect(**_conn_kwargs(ssh_cfg, host)) as sftp_conn:
                        async with sftp_conn.start_sftp_client() as sftp:
                            await sftp.put(archive, "/tmp/_cserve_sync.tar.gz")
                        await sftp_conn.run(
                            "mkdir -p ~/CServe && "
                            "tar -xzf /tmp/_cserve_sync.tar.gz -C ~/CServe && "
                            "rm /tmp/_cserve_sync.tar.gz",
                            check=True,
                        )
                    _log("    ✓ code synced via SFTP")
                except Exception as e:
                    _log(f"    SFTP sync failed: {e}")
                    return DeployResult(ok=False, log=lines, error=f"Code sync failed: {e}")
            else:
                full_rsync_cmd = rsync_cmd + [
                    "rsync", "-az", "--delete",
                    "--exclude=.git", "--exclude=__pycache__",
                    "--exclude=*.pyc", "--exclude=node_modules",
                    "--exclude=dashboard-ui/node_modules",
                    "--exclude=.ruff_cache",
                    "-e", ssh_opts,
                    f"{local_cserve_src}/",
                    f"{ssh_cfg.username}@{host}:~/CServe/",
                ]
                proc = await asyncio.create_subprocess_exec(
                    *full_rsync_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = (stdout_bytes or b"").decode()
                if proc.returncode and proc.returncode != 0:
                    _log(f"    rsync exited {proc.returncode}: {output[:400]}")
                    err = f"rsync failed (exit {proc.returncode}): {output[:200]}"
                    return DeployResult(ok=False, log=lines, error=err)
                _log("    ✓ code synced")
        else:
            _log("[1/4] Skipping code sync (sync_code=false)")

        # ── Steps 2–4 via SSH ────────────────────────────────────────────────
        async with asyncssh.connect(**_conn_kwargs(ssh_cfg, host)) as conn:

            _log(f"[2/4] Installing CServe package on {host} …")
            rc, out, err = await _run(
                conn,
                f"{_remote_path(ssh_cfg.pip_path)} install -e ~/CServe/ -q 2>&1 | tail -5",
            )
            if rc != 0:
                _log(f"    pip install failed: {err or out}")
                return DeployResult(ok=False, log=lines, error=f"pip install failed: {err or out}")
            _log("    ✓ package installed")

            _log("[3/4] Stopping any existing node agent …")
            await _run(
                conn,
                f"pkill -f 'cserve.node_agent.server' 2>/dev/null; "
                f"fuser -k {agent_port}/tcp 2>/dev/null; true",
            )
            await asyncio.sleep(1)
            _log("    ✓ agent stopped")

            _log("[4/4] Starting node agent …")
            start_cmd = (
                f"cd ~/CServe && "
                f"CSERVE_CUDA_DEVICES={shlex.quote(cuda_devices)} "
                f"nohup {_remote_path(ssh_cfg.python_path)} -m cserve.node_agent.server "
                f"--node-name {shlex.quote(node_name)} "
                f"--node-host {shlex.quote(host)} "
                f"--control-plane {shlex.quote(control_plane_url)} "
                f"--port {agent_port} "
                f"--transport {transport} "
                f"> ~/cserve-agent.log 2>&1 &"
            )
            rc, out, err = await _run(conn, start_cmd)
            if rc != 0:
                _log(f"    start failed: {err or out}")
                return DeployResult(ok=False, log=lines, error=f"Failed to start agent: {err or out}")

            _log(f"    ✓ node agent launched (cuda={cuda_devices})")

        _log("Deployment complete ✓")
        return DeployResult(ok=True, log=lines)

    except asyncssh.Error as e:
        _log(f"SSH error: {e}")
        return DeployResult(ok=False, log=lines, error=f"SSH error: {e}")
    except TimeoutError:
        msg = f"Operation timed out (>{ssh_cfg.timeout_s:.0f}s)"
        _log(msg)
        return DeployResult(ok=False, log=lines, error=msg)
    except Exception as e:  # noqa: BLE001
        _log(f"Unexpected error: {e}")
        return DeployResult(ok=False, log=lines, error=str(e))


async def stop_agent(host: str, agent_port: int, ssh_cfg: SshConfig) -> DeployResult:
    """Kill the node agent process on *host*."""
    lines: list[str] = []

    def _log(msg: str) -> None:
        log.info(msg, host=host)
        lines.append(msg)

    try:
        async with asyncssh.connect(**_conn_kwargs(ssh_cfg, host)) as conn:
            _log(f"Stopping CServe node agent on {host}:{agent_port} …")
            await _run(
                conn,
                f"pkill -f 'cserve.node_agent.server' 2>/dev/null; "
                f"fuser -k {agent_port}/tcp 2>/dev/null; true",
            )
            _log("    ✓ agent stopped")
        return DeployResult(ok=True, log=lines)
    except asyncssh.Error as e:
        _log(f"SSH error: {e}")
        return DeployResult(ok=False, log=lines, error=f"SSH error: {e}")
    except Exception as e:  # noqa: BLE001
        _log(f"Unexpected error: {e}")
        return DeployResult(ok=False, log=lines, error=str(e))
