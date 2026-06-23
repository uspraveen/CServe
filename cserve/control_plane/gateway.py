"""API Gateway — the public-facing HTTP server.

Exposes an OpenAI-compatible API.  Clients talk to this and nothing else.

Dual routing:
  - Fast path: Gateway picks best replica via LOR, proxies directly to vLLM.
  - Queue path: When all replicas are saturated, falls through to Redis queue.

Auth:
  - Every request must include `Authorization: Bearer csk_...`
  - Key is validated against SQLite (hashed lookup), rate-limited via Redis.
  - Per-key usage is logged for attribution and billing.

This is a FastAPI app.  It is mounted by server.py alongside the dashboard API.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from cserve.common.auth import AuthenticatedUser
from cserve.common.logging import get_logger
from cserve.common.metrics import (
    GATEWAY_ACTIVE_STREAMS,
    GATEWAY_REQUEST_DURATION,
    GATEWAY_REQUESTS_TOTAL,
)
from cserve.common.models import Job, JobEvent, JobEventRecord

log = get_logger("gateway")

INFERENCE_TIMEOUT_S = 600.0


def _is_upstream_connection_error(exc: BaseException) -> bool:
    """True when the gateway could not reach vLLM (TCP/TLS/connect), not an HTTP 4xx/5xx body."""
    if isinstance(exc, httpx.RequestError):
        return True
    err = str(exc).lower()
    return any(
        tok in err
        for tok in (
            "connection",
            "connect",
            "refused",
            "unreachable",
            "all connection attempts",
            "name or service not known",
            "getaddrinfo failed",
        )
    )


def _should_cool_down_replica(exc: BaseException) -> bool:
    """Only hide a replica for true TCP connect failures.

    ReadTimeout and RemoteProtocolError usually mean a loaded vLLM accepted
    work but did not respond cleanly.  Hiding the only READY replica for those
    cases caused user-visible 503 storms.
    """
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


def _503_upstream_unavailable(model: str, detail: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": (
                    f"Upstream replicas for '{model}' are temporarily unreachable "
                    f"({detail}). This is usually a brief restart or network blip — "
                    "please retry with exponential back-off."
                ),
                "type": "upstream_unavailable",
                "param": None,
                "code": "upstream_unavailable",
            },
        },
        status_code=503,
        headers={"Retry-After": "3"},
    )

# After a true TCP connect failure, hide it briefly from fast-path.
_UPSTREAM_COOLDOWN_S = 3.0

# Prefer vLLM Prometheus load over gateway inflight (avoids counter drift).
_METRICS_RUNNING = "vllm:num_requests_running"
_METRICS_WAITING = "vllm:num_requests_waiting"


def _replica_dispatch_load(replica) -> int:
    """How loaded a replica is for admission control."""
    metrics = replica.metrics_snapshot or {}
    running = metrics.get(_METRICS_RUNNING)
    if running is not None:
        waiting = int(metrics.get(_METRICS_WAITING, 0))
        return int(running) + waiting
    return replica.inflight_requests

# Headers to strip when proxying to vLLM — never forward auth or hop-by-hop headers
_STRIP_HEADERS = frozenset({
    "host", "content-length", "connection", "transfer-encoding",
    "authorization", "proxy-authorization",
})


class Gateway:
    """Encapsulates the FastAPI app and its dependencies."""

    def __init__(
        self, registry, queue, db, models_config,
        demand_tracker=None, rate_limiter=None,
    ):
        self.registry = registry
        self.queue = queue
        self.db = db
        self.models_config = models_config
        self.demand_tracker = demand_tracker
        self.rate_limiter = rate_limiter
        self._http_client: httpx.AsyncClient | None = None
        self._key_cache: dict[str, tuple[float, AuthenticatedUser | None]] = {}
        self._cache_ttl = 5.0

        self.app = FastAPI(title="CServe", version="0.1.0", docs_url=None, redoc_url=None)
        self._register_routes()

    async def startup(self) -> None:
        try:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(INFERENCE_TIMEOUT_S, connect=5.0),
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
                http2=True,
            )
            log.info("gateway httpx client started with HTTP/2")
        except Exception:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(INFERENCE_TIMEOUT_S, connect=5.0),
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
            )
            log.info("gateway httpx client started with HTTP/1.1 (h2 unavailable)")

    async def shutdown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()

    @property
    def http_client(self) -> httpx.AsyncClient:
        assert self._http_client is not None, "Gateway not started"
        return self._http_client

    async def _authenticate(self, request: Request) -> AuthenticatedUser | JSONResponse:
        """Validate the Authorization header and return the authenticated user.

        Returns AuthenticatedUser on success, JSONResponse (401/403) on failure.
        Uses a short TTL cache to avoid hitting SQLite on every request.
        """
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": {"message": "Missing or invalid Authorization header. Use: Bearer csk_...",
                           "type": "auth_error"}},
                status_code=401,
            )
        raw_key = auth_header[7:].strip()
        if not raw_key:
            return JSONResponse(
                {"error": {"message": "Empty API key", "type": "auth_error"}},
                status_code=401,
            )

        now = time.time()
        cached = self._key_cache.get(raw_key)
        if cached and (now - cached[0]) < self._cache_ttl and cached[1] is not None:
            return cached[1]

        api_key = await self.db.authenticate_key(raw_key)
        if not api_key:
            self._key_cache[raw_key] = (now, None)
            return JSONResponse(
                {"error": {"message": "Invalid or disabled API key", "type": "auth_error"}},
                status_code=401,
            )

        user = AuthenticatedUser(
            key_id=api_key.key_id,
            user_id=api_key.user_id,
            role=api_key.role,
            rate_limit_rpm=api_key.rate_limit_rpm,
        )
        self._key_cache[raw_key] = (now, user)
        return user

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/health")
        async def health():
            models = sorted(self.models_config.keys())
            redis_ok = await self.queue.ping()
            return {"ok": True, "models": models, "redis": redis_ok}

        @app.get("/v1/models")
        async def list_models(request: Request):
            auth = await self._authenticate(request)
            if isinstance(auth, JSONResponse):
                return auth
            data = [
                {
                    "id": cfg.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "cserve",
                    "permission": [],
                    "root": cfg.served_model_name,
                    "parent": None,
                }
                for cfg in self.models_config.values()
            ]
            return {"object": "list", "data": data}

        @app.api_route("/v1/{path:path}", methods=["POST", "GET", "PUT", "DELETE"])
        async def proxy_v1(request: Request, path: str):
            return await self._handle_request(request, f"/v1/{path}")

    async def _first_pubsub_message(self, pubsub) -> dict | None:
        """Get the first message from a subscribed callback channel."""
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                return json.loads(msg["data"])
        return None

    async def _wait_for_callback_or_disconnect(
        self, request: Request, job_id: str, model: str, timeout_s: float, job: Job, path: str
    ) -> dict | None:
        """Wait for scheduler callback. Returns callback data or None on timeout/disconnect.
        Subscribes before enqueue to avoid missing fast scheduler responses.
        Cancels the job on timeout or disconnect."""
        async def wait_callback() -> dict | None:
            # Subscribe first, then enqueue, then wait — avoids missing callback if scheduler is fast
            pubsub = await self.queue.subscribe_callback(job_id)
            await self.queue.enqueue(job)
            asyncio.create_task(self.db.log_job_event(JobEventRecord(
                job_id=job_id, event=JobEvent.ENQUEUED,
                metadata={"model": model, "path": path, "streaming": job.streaming},
            )))
            try:
                return await asyncio.wait_for(
                    self._first_pubsub_message(pubsub),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                return None
            except asyncio.CancelledError:
                return None
            finally:
                await pubsub.unsubscribe(self.queue.callback_channel(job_id))
                await pubsub.close()

        async def wait_disconnect() -> None:
            poll_interval = 0.2
            while True:
                await asyncio.sleep(poll_interval)
                if await request.is_disconnected():
                    return

        callback_task = asyncio.create_task(wait_callback())
        disconnect_task = asyncio.create_task(wait_disconnect())

        done, pending = await asyncio.wait(
            [callback_task, disconnect_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        for t in done:
            if t == callback_task:
                try:
                    data = t.result()
                    if data is not None:
                        return data
                except Exception:
                    pass
                await self.queue.cancel(job_id, model)
                return None
            else:
                await self.queue.cancel(job_id, model)
                return None

        return None

    async def _recover_via_queue_after_upstream_fail(
        self,
        request: Request,
        path: str,
        body: bytes,
        model: str,
        streaming: bool,
        tenant: str,
        start: float,
        user: AuthenticatedUser,
        model_cfg,
        base_job_id: str,
    ) -> Response | JSONResponse | StreamingResponse | None:
        """When every fast-path replica failed to connect, try Redis + scheduler once."""
        if not callable(getattr(self.queue, "enqueue", None)):
            return None
        max_q = model_cfg.autoscaling.max_queue_depth
        if max_q <= 0:
            return None
        try:
            if await self.queue.queue_depth(model) >= max_q:
                return None
        except Exception:
            return None

        job_id = f"{base_job_id}qrec"
        deadline_ms = max(model_cfg.autoscaling.max_queue_wait_ms or 30_000, 15_000)
        job = Job(
            job_id=job_id,
            tenant_id=tenant,
            model=model,
            payload=body,
            headers=dict(request.headers),
            streaming=streaming,
            deadline_ms=deadline_ms,
        )
        timeout_s = min(deadline_ms / 1000.0, 300.0)
        log.info(
            "upstream connect failures — trying Redis queue recovery",
            model=model, job_id=job_id,
        )
        try:
            callback_data = await self._wait_for_callback_or_disconnect(
                request, job_id, model, timeout_s, job, path,
            )
        except Exception as e:
            log.warning("queue recovery enqueue/wait failed", model=model, error=str(e))
            return None

        if not callback_data:
            return None

        replica = self.registry.get_replica(callback_data["replica_id"])
        if not replica:
            try:
                await self.queue.cancel(job_id, model)
            except Exception:
                pass
            return None

        if self.demand_tracker:
            rr = self.registry.get_routable_replicas(model)
            total_inflight = sum(r.inflight_requests for r in rr) + 1
            self.demand_tracker.record_request(model, total_inflight)

        try:
            if streaming:
                return await self._stream_request(
                    request, replica, path, body, model, tenant, start, job_id, user,
                )
            return await self._forward_request(
                request, replica, path, body, model, tenant, start, job_id, user,
            )
        except Exception as e:
            self.registry.decrement_inflight(replica.replica_id)
            asyncio.create_task(self.db.log_job_event(JobEventRecord(
                job_id=job_id, event=JobEvent.FAILED,
                replica_id=replica.replica_id, node_name=replica.node_name,
                metadata={"error": str(e), "phase": "queue_recovery"},
            )))
            log.error("queue recovery forward failed", model=model, error=str(e))
            return None

    async def _handle_request(self, request: Request, path: str) -> Response:
        start = time.time()

        # ── Auth ─────────────────────────────────────────────────────────────
        auth = await self._authenticate(request)
        if isinstance(auth, JSONResponse):
            return auth
        user: AuthenticatedUser = auth

        # Bump key usage on every request (including cache hits) so Keys page
        # total_requests matches Usage page
        asyncio.create_task(self.db.increment_key_requests(user.key_id))

        # ── Rate limiting ────────────────────────────────────────────────────
        if self.rate_limiter and user.rate_limit_rpm > 0:
            allowed, current, limit = await self.rate_limiter.check(
                user.key_id, user.rate_limit_rpm,
            )
            if not allowed:
                return JSONResponse(
                    {"error": {
                        "message": f"Rate limit exceeded ({current}/{limit} requests/min)",
                        "type": "rate_limit_error",
                    }},
                    status_code=429,
                    headers={
                        "Retry-After": "5",
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )

        body = await request.body()
        model = None
        streaming = False
        tenant = user.user_id
        content_type = request.headers.get("content-type", "").lower()

        # Parse model from request body. Most OpenAI-compatible endpoints send
        # JSON, but Whisper-style transcription requests use multipart form data.
        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            try:
                form = await request.form()
                model = form.get("model")
            except Exception:
                pass
        else:
            try:
                payload = json.loads(body) if body else {}
                model = payload.get("model")
                streaming = payload.get("stream", False)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        if not model:
            return JSONResponse(
                {"error": {"message": "Missing 'model' field in request body", "type": "invalid_request_error"}},
                status_code=400,
            )

        # Verify model exists
        model_cfg = self.models_config.get(model)
        if not model_cfg:
            available = sorted(self.models_config.keys())
            msg = f"Model '{model}' not found. Available: {available}"
            return JSONResponse(
                {"error": {"message": msg, "type": "invalid_request_error"}},
                status_code=404,
            )

        # ── Queue depth backpressure (lazy — only check Redis when saturated) ──
        # Moved below: we only query Redis queue_depth when the fast path
        # is full (all replicas at dispatch ceiling).  On the normal path
        # this saves one Redis RTT (~0.3ms) per request.

        # ── Check for available replicas — fast path ──────────────────────────
        replicas = self.registry.get_routable_replicas(model)
        if not replicas:
            healthy = self.registry.get_healthy_replicas(model)
            if healthy:
                # READY but every replica is in a post-connect-failure cooldown
                # window.  Do not surface this as a 503; route anyway and let
                # the normal retry/queue path handle transient upstream errors.
                replicas = healthy
                log.warning(
                    "all routable replicas cooling down; routing to READY replicas",
                    model=model, replicas=len(replicas),
                )
            else:
                # No routable/healthy replica yet.  This commonly happens just
                # after a control-plane restart while min_replicas are launching.
                # Hold the request until a replica appears instead of returning
                # a user-visible "model has no replicas" 503.
                if self.demand_tracker:
                    self.demand_tracker.record_request(model, 1)
                    log.info("scale-from-zero triggered", model=model)

                configured_wait = model_cfg.autoscaling.max_queue_wait_ms / 1000
                max_wait = min(max(configured_wait, 300.0), 300.0)
                poll_interval = 0.5
                waited = 0.0
                while waited < max_wait:
                    if await request.is_disconnected():
                        log.info("client disconnected while waiting for replica", model=model)
                        return Response(status_code=499)
                    await asyncio.sleep(poll_interval)
                    waited += poll_interval
                    replicas = self.registry.get_routable_replicas(model)
                    if not replicas:
                        healthy = self.registry.get_healthy_replicas(model)
                        if healthy:
                            replicas = healthy
                    if replicas:
                        break

                if not replicas:
                    elapsed = time.time() - start
                    GATEWAY_REQUESTS_TOTAL.labels(model=model, status_code="503", tenant=tenant).inc()
                    GATEWAY_REQUEST_DURATION.labels(model=model, streaming=str(streaming)).observe(elapsed)
                    all_replicas = self.registry.get_replicas_for_model(model)
                    starting_count = sum(
                        1 for r in all_replicas if r.status.value == "STARTING"
                    ) if all_replicas else 0
                    msg = (
                        f"Model '{model}' is still starting after {max_wait:.0f}s "
                        f"({starting_count} replica(s) launching)"
                        if starting_count > 0
                        else f"Model '{model}' has no replicas after waiting {max_wait:.0f}s"
                    )
                    return JSONResponse(
                        {"error": {"message": msg, "type": "server_error", "model": model}},
                        status_code=503,
                        headers={"Retry-After": "15"},
                    )

        # ── vLLM admission control on the fast path ───────────────────────────
        # vLLM has its own internal scheduler with a bounded waiting queue
        # (max_num_seqs).  If we push more requests than it can batch, they
        # pile up inside vLLM's KV-cache scheduler — invisible to us, burning
        # GPU memory, and causing unpredictable latency spikes.
        #
        # We cap fast-path dispatch at 90% of max_num_seqs per replica.
        # Requests that arrive when ALL replicas are at capacity get enqueued
        # to Redis for the background scheduler to drain, rather than
        # double-queuing inside vLLM.
        engine_max = model_cfg.engine.max_num_seqs
        # Match direct vLLM: let vLLM own batching/queueing up to max_num_seqs.
        dispatch_ceiling = max(1, engine_max)
        eligible = [
            r for r in replicas if _replica_dispatch_load(r) < dispatch_ceiling
        ]

        if not eligible:
            # All replicas at vLLM capacity.  Now (and only now) check Redis
            # queue depth for backpressure — saves a Redis RTT on the normal path.
            max_q = model_cfg.autoscaling.max_queue_depth
            if max_q > 0:
                try:
                    current_depth = await self.queue.queue_depth(model)
                    if current_depth >= max_q:
                        GATEWAY_REQUESTS_TOTAL.labels(
                            model=model, status_code="429", tenant=tenant).inc()
                        return JSONResponse(
                            {"error": {
                                "message": (
                                    f"Queue for '{model}' full ({current_depth}/{max_q}). "
                                    "Please retry after back-off."
                                ),
                                "type": "rate_limit_error",
                                "model": model,
                            }},
                            status_code=429,
                            headers={"Retry-After": "5",
                                     "X-CServe-Queue-Depth": str(current_depth)},
                        )
                except Exception:
                    pass

            # Hold the connection and poll for a free slot.
            max_wait = min(model_cfg.autoscaling.max_queue_wait_ms / 1000, 300.0)
            poll_interval = 0.2
            waited = 0.0
            total_inflight = sum(r.inflight_requests for r in replicas)
            log.info("all replicas at vLLM capacity, waiting for slot",
                     model=model, inflight=total_inflight,
                     ceiling=dispatch_ceiling, replicas=len(replicas))

            while waited < max_wait:
                if await request.is_disconnected():
                    log.info("client disconnected while waiting for slot", model=model)
                    return Response(status_code=499)  # Client Closed Request

                await asyncio.sleep(poll_interval)
                waited += poll_interval
                replicas = self.registry.get_routable_replicas(model)
                eligible = [
                    r for r in replicas
                    if _replica_dispatch_load(r) < dispatch_ceiling
                ]
                if eligible:
                    break

            if not eligible:
                # Queue path: enqueue and wait for scheduler to assign a replica
                job_id = f"{int(time.time()*1e6):x}"
                deadline_ms = model_cfg.autoscaling.max_queue_wait_ms or 30_000
                job = Job(
                    job_id=job_id,
                    tenant_id=tenant,
                    model=model,
                    payload=body,
                    headers=dict(request.headers),
                    streaming=streaming,
                    deadline_ms=deadline_ms,
                )
                timeout_s = min(deadline_ms / 1000, 300.0)
                # Subscribe before enqueue so we don't miss the callback if scheduler is fast
                callback_data = await self._wait_for_callback_or_disconnect(
                    request, job_id, model, timeout_s, job, path,
                )

                if callback_data is None:
                    if await request.is_disconnected():
                        return Response(status_code=499)
                    elapsed = time.time() - start
                    GATEWAY_REQUESTS_TOTAL.labels(
                        model=model, status_code="429", tenant=tenant).inc()
                    GATEWAY_REQUEST_DURATION.labels(
                        model=model, streaming=str(streaming)).observe(elapsed)
                    return JSONResponse(
                        {"error": {
                            "message": (
                                f"All {len(replicas)} replicas at capacity and no "
                                f"slot opened within {timeout_s:.0f}s. "
                                "Please retry with exponential back-off."
                            ),
                            "type": "rate_limit_error",
                            "model": model,
                        }},
                        status_code=429,
                        headers={"Retry-After": "3"},
                    )

                replica = self.registry.get_replica(callback_data["replica_id"])
                if not replica:
                    await self.queue.cancel(job_id, model)
                    return JSONResponse(
                        {"error": {
                            "message": "Replica unavailable after assignment",
                            "type": "server_error",
                            "model": model,
                        }},
                        status_code=503,
                        headers={"Retry-After": "5"},
                    )

                # Scheduler already incremented inflight; forward/stream will decrement
                if self.demand_tracker:
                    total_inflight = sum(r.inflight_requests for r in replicas) + 1
                    self.demand_tracker.record_request(model, total_inflight)

                try:
                    if streaming:
                        return await self._stream_request(
                            request, replica, path, body, model, tenant, start, job_id, user,
                        )
                    else:
                        return await self._forward_request(
                            request, replica, path, body, model, tenant, start, job_id, user,
                        )
                except Exception as e:
                    self.registry.decrement_inflight(replica.replica_id)
                    asyncio.create_task(self.db.log_job_event(JobEventRecord(
                        job_id=job_id, event=JobEvent.FAILED,
                        replica_id=replica.replica_id, node_name=replica.node_name,
                        metadata={"error": str(e)},
                    )))
                    elapsed = time.time() - start
                    if _is_upstream_connection_error(e):
                        GATEWAY_REQUESTS_TOTAL.labels(
                            model=model, status_code="503", tenant=tenant).inc()
                        return _503_upstream_unavailable(model, str(e)[:160])
                    GATEWAY_REQUESTS_TOTAL.labels(model=model, status_code="502", tenant=tenant).inc()
                    return JSONResponse(
                        {"error": {"message": f"Upstream error: {e}", "type": "server_error"}},
                        status_code=502,
                    )

        # Select best replica (least outstanding requests) from eligible set.
        # We try up to len(eligible) replicas before giving up — if a replica's
        # vLLM process just died, we skip it and retry on the next best one.
        # This prevents a replica crash from causing 502s during its restart window.
        tried: set[str] = set()
        last_error: Exception | None = None
        job_id = f"{int(time.time()*1e6):x}"

        # Emit demand signal so the autoscaler sees fast-path load
        if self.demand_tracker:
            total_inflight = sum(r.inflight_requests for r in replicas) + 1
            self.demand_tracker.record_request(model, total_inflight)

        while True:
            # Pick best untried replica
            candidates = [r for r in eligible if r.replica_id not in tried]
            if not candidates:
                break
            replica = min(candidates, key=_replica_dispatch_load)
            tried.add(replica.replica_id)

            self.registry.increment_inflight(replica.replica_id)
            asyncio.create_task(self.db.log_job_event(JobEventRecord(
                job_id=job_id, event=JobEvent.SCHEDULED,
                replica_id=replica.replica_id, node_name=replica.node_name,
                metadata={"model": model, "path": path, "streaming": streaming},
            )))

            try:
                if streaming:
                    return await self._stream_request(
                        request, replica, path, body, model, tenant, start, job_id, user,
                    )
                else:
                    return await self._forward_request(
                        request, replica, path, body, model, tenant, start, job_id, user,
                    )
            except Exception as e:
                self.registry.decrement_inflight(replica.replica_id)
                conn = _is_upstream_connection_error(e)
                if _should_cool_down_replica(e):
                    self.registry.mark_upstream_connection_failed(
                        replica.replica_id, _UPSTREAM_COOLDOWN_S)
                asyncio.create_task(self.db.log_job_event(JobEventRecord(
                    job_id=job_id, event=JobEvent.FAILED,
                    replica_id=replica.replica_id, node_name=replica.node_name,
                    metadata={"error": str(e)},
                )))
                if conn and len(tried) < len(eligible):
                    log.warning(
                        "replica unreachable, retrying on next replica",
                        model=model, failed_replica=replica.replica_id,
                        node=replica.node_name, error=str(e),
                        remaining=len(eligible) - len(tried),
                    )
                    last_error = e
                    continue
                last_error = e
                break

        # All replicas tried and failed — optional Redis recovery, then 503 vs 502
        elapsed = time.time() - start
        if last_error is not None and _is_upstream_connection_error(last_error):
            recovered = await self._recover_via_queue_after_upstream_fail(
                request, path, body, model, streaming, tenant, start, user, model_cfg, job_id,
            )
            if recovered is not None:
                return recovered

        sc = "503" if (last_error is not None and _is_upstream_connection_error(last_error)) else "502"
        GATEWAY_REQUESTS_TOTAL.labels(model=model, status_code=sc, tenant=tenant).inc()
        GATEWAY_REQUEST_DURATION.labels(model=model, streaming=str(streaming)).observe(elapsed)
        asyncio.create_task(self.db.log_usage(
            key_id=user.key_id, user_id=user.user_id, model=model,
            job_id=job_id, replica_id=replica.replica_id,
            node_name=replica.node_name,
            gpu_ids=",".join(str(g) for g in replica.gpu_ids),
            status_code=int(sc), latency_s=elapsed, streaming=streaming,
        ))
        log.error("request failed on all replicas", model=model,
                  tried=len(tried), error=str(last_error))
        if last_error is not None and _is_upstream_connection_error(last_error):
            return _503_upstream_unavailable(model, str(last_error)[:160])
        return JSONResponse(
            {"error": {"message": f"Upstream error: {last_error}", "type": "server_error"}},
            status_code=502,
        )

    async def _forward_request(
        self, request: Request, replica, path: str, body: bytes,
        model: str, tenant: str, start: float, job_id: str,
        user: AuthenticatedUser,
    ) -> Response:
        """Forward a non-streaming request to a vLLM replica."""
        url = f"{replica.http_endpoint}{path}"
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _STRIP_HEADERS
        }

        try:
            resp = await self.http_client.request(
                method=request.method,
                url=url,
                content=body,
                headers=headers,
            )
        except httpx.RequestError as e:
            if _should_cool_down_replica(e):
                self.registry.mark_upstream_connection_failed(
                    replica.replica_id, _UPSTREAM_COOLDOWN_S)
            raise

        self.registry.decrement_inflight(replica.replica_id)

        elapsed = time.time() - start
        GATEWAY_REQUESTS_TOTAL.labels(model=model, status_code=str(resp.status_code), tenant=tenant).inc()
        GATEWAY_REQUEST_DURATION.labels(model=model, streaming="False").observe(elapsed)

        # Extract token counts from vLLM response if available
        prompt_tokens = completion_tokens = 0
        try:
            resp_data = resp.json()
            usage = resp_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        except Exception:
            pass

        asyncio.create_task(self.db.log_job_event(JobEventRecord(
            job_id=job_id, event=JobEvent.COMPLETED,
            replica_id=replica.replica_id, node_name=replica.node_name,
            metadata={"model": model, "latency_s": elapsed, "status_code": resp.status_code},
        )))
        asyncio.create_task(self.db.log_usage(
            key_id=user.key_id, user_id=user.user_id, model=model,
            job_id=job_id, replica_id=replica.replica_id,
            node_name=replica.node_name,
            gpu_ids=",".join(str(g) for g in replica.gpu_ids),
            status_code=resp.status_code, latency_s=elapsed,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        ))

        response_headers = dict(resp.headers)
        response_headers["X-CServe-Replica"] = replica.replica_id
        response_headers["X-CServe-Node"] = replica.node_name
        response_headers["X-CServe-Latency-Ms"] = str(int(elapsed * 1000))

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
        )

    async def _stream_request(
        self, request: Request, replica, path: str, body: bytes,
        model: str, tenant: str, start: float, job_id: str,
        user: AuthenticatedUser,
    ) -> StreamingResponse:
        """Open a direct SSE stream from the vLLM replica to the client."""
        url = f"{replica.http_endpoint}{path}"
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _STRIP_HEADERS
        }

        GATEWAY_ACTIVE_STREAMS.labels(model=model).inc()

        async def stream_generator():
            first_token_logged = False
            try:
                async with self.http_client.stream(
                    method=request.method, url=url, content=body, headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        if not first_token_logged:
                            first_token_logged = True
                            asyncio.create_task(self.db.log_job_event(JobEventRecord(
                                job_id=job_id, event=JobEvent.FIRST_TOKEN,
                                replica_id=replica.replica_id,
                                metadata={"ttft_s": time.time() - start},
                            )))
                        yield chunk
            except httpx.RequestError as e:
                if _should_cool_down_replica(e):
                    self.registry.mark_upstream_connection_failed(
                        replica.replica_id, _UPSTREAM_COOLDOWN_S)
                log.error(
                    "stream could not connect to replica",
                    model=model, replica=replica.replica_id, error=str(e),
                )
                payload = json.dumps({
                    "error": "upstream_unreachable",
                    "message": str(e)[:240],
                })
                yield f"data: {payload}\n\n".encode()
            except Exception as e:
                log.error("stream error", model=model, replica=replica.replica_id, error=str(e))
                yield f'data: {{"error": "{e}"}}\n\n'.encode()
            finally:
                self.registry.decrement_inflight(replica.replica_id)
                GATEWAY_ACTIVE_STREAMS.labels(model=model).dec()
                elapsed = time.time() - start
                GATEWAY_REQUESTS_TOTAL.labels(model=model, status_code="200", tenant=tenant).inc()
                GATEWAY_REQUEST_DURATION.labels(model=model, streaming="True").observe(elapsed)
                asyncio.create_task(self.db.log_job_event(JobEventRecord(
                    job_id=job_id, event=JobEvent.COMPLETED,
                    replica_id=replica.replica_id, node_name=replica.node_name,
                    metadata={"model": model, "latency_s": elapsed, "streaming": True},
                )))
                asyncio.create_task(self.db.log_usage(
                    key_id=user.key_id, user_id=user.user_id, model=model,
                    job_id=job_id, replica_id=replica.replica_id,
                    node_name=replica.node_name,
                    gpu_ids=",".join(str(g) for g in replica.gpu_ids),
                    status_code=200, latency_s=elapsed, streaming=True,
                ))

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-CServe-Replica": replica.replica_id,
                "X-CServe-Node": replica.node_name,
            },
        )
