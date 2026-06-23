"""CServe Node Agent Server — entry point for GPU worker nodes.

Supports two transport modes:
  --transport http  (default) — JSON-over-HTTP via FastAPI/uvicorn
  --transport grpc             — binary gRPC via compiled protobuf stubs

Usage:
  cserve-agent --node-name worker1 --node-host 10.0.1.5 \
               --control-plane http://10.0.1.1:8002 \
               --port 50051 --transport grpc
"""

from __future__ import annotations

import argparse
import asyncio
import resource

import uvicorn

from cserve.common.logging import get_logger
from cserve.node_agent.agent import NodeAgent

log = get_logger("agent_server")


def _raise_fd_limit() -> None:
    """Raise RLIMIT_NOFILE so vLLM and nvidia-smi don't hit [Errno 24] Too many open files."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 65535:
            target = min(65535, hard) if hard != resource.RLIM_INFINITY else 65535
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            log.info("raised RLIMIT_NOFILE", from_=soft, to=target)
    except (ValueError, OSError) as e:
        log.warning("could not raise fd limit", error=str(e))


def main() -> None:
    _raise_fd_limit()
    parser = argparse.ArgumentParser(description="CServe Node Agent")
    parser.add_argument("--node-name", required=True, help="Name of this node (must match cluster.yaml)")
    parser.add_argument("--node-host", required=True, help="IP address of this node")
    parser.add_argument("--control-plane", required=True, help="URL of the control plane (e.g. http://10.0.1.1:8002)")
    parser.add_argument("--port", type=int, default=50051, help="Port for the agent server")
    parser.add_argument("--heartbeat-interval", type=int, default=10, help="Heartbeat interval in seconds")
    parser.add_argument(
        "--transport", choices=["http", "grpc"], default="http",
        help="Transport protocol: 'http' (JSON-over-HTTP) or 'grpc' (binary protobuf)",
    )
    args = parser.parse_args()

    if args.transport == "grpc":
        _run_grpc(args)
    else:
        _run_http(args)


def _run_http(args) -> None:
    agent = NodeAgent(
        node_name=args.node_name,
        node_host=args.node_host,
        control_plane_url=args.control_plane,
        agent_port=args.port,
        heartbeat_interval_s=args.heartbeat_interval,
    )

    config = uvicorn.Config(
        app=agent.app, host="0.0.0.0", port=args.port,
        log_level="info", access_log=False,
    )
    uvicorn_server = uvicorn.Server(config)

    async def run():
        await agent.startup()
        try:
            await uvicorn_server.serve()
        finally:
            await agent.shutdown()

    log.info("starting node agent (HTTP)", node=args.node_name,
             host=args.node_host, port=args.port)
    asyncio.run(run())


def _run_grpc(args) -> None:
    from cserve.node_agent.grpc_server import serve_grpc

    async def run():
        server = await serve_grpc(
            node_name=args.node_name,
            node_host=args.node_host,
            control_plane_url=args.control_plane,
            port=args.port,
        )
        log.info("starting node agent (gRPC)", node=args.node_name,
                 host=args.node_host, port=args.port)
        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            await server.stop(grace=5)

    asyncio.run(run())


if __name__ == "__main__":
    main()
