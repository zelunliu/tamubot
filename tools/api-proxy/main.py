"""
Rate-limiting reverse proxy for TAMU and Voyage AI APIs.

Routes:
  POST /tamu/{path}   -> https://chat-api.tamu.ai/openai/{path}
  POST /voyage/{path} -> https://api.voyageai.com/{path}

Rate limits read from env:
  API_PROXY_TAMU_RPM        (default 30)
  API_PROXY_TAMU_SESSION    (default 100)
  API_PROXY_VOYAGE_RPM      (default 60)
  API_PROXY_VOYAGE_SESSION  (default 200)
"""

import asyncio
import csv
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TAMU_BASE = "https://chat-api.tamu.ai/openai"
VOYAGE_BASE = "https://api.voyageai.com"

LIMITS = {
    "tamu": {
        "rpm": int(os.getenv("API_PROXY_TAMU_RPM", "30")),
        "session": int(os.getenv("API_PROXY_TAMU_SESSION", "100")),
    },
    "voyage": {
        "rpm": int(os.getenv("API_PROXY_VOYAGE_RPM", "60")),
        "session": int(os.getenv("API_PROXY_VOYAGE_SESSION", "200")),
    },
}

LOG_PATH = Path("/workspace/logs/api-proxy.csv")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

# Sliding window of request timestamps (last 60s) per service
rpm_windows: dict[str, deque] = {
    "tamu": deque(),
    "voyage": deque(),
}

# Per-service, per-session request counts
session_counts: dict[str, dict[str, int]] = {
    "tamu": defaultdict(int),
    "voyage": defaultdict(int),
}

# One lock per service
locks: dict[str, asyncio.Lock] = {
    "tamu": asyncio.Lock(),
    "voyage": asyncio.Lock(),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

app = FastAPI(title="API Rate-Limiting Proxy")


def _ensure_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "service", "path", "session_id", "status_code"])


def _log(service: str, path: str, session_id: str, status_code: int):
    _ensure_log()
    with LOG_PATH.open("a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(),
            service,
            path,
            session_id,
            status_code,
        ])


def _prune_window(window: deque, now: float):
    """Remove timestamps older than 60 seconds."""
    cutoff = now - 60.0
    while window and window[0] < cutoff:
        window.popleft()


async def _check_and_record(service: str, session_id: str) -> JSONResponse | None:
    """
    Check rate limits and record the request.
    Returns a JSONResponse (429) if a limit is hit, else None.
    """
    limits = LIMITS[service]
    rpm_limit = limits["rpm"]
    session_limit = limits["session"]

    async with locks[service]:
        now = time.monotonic()
        window = rpm_windows[service]
        _prune_window(window, now)

        # Per-minute cap
        if len(window) >= rpm_limit:
            return JSONResponse(
                status_code=429,
                content={"error": "rate limit exceeded", "service": service, "rpm_limit": rpm_limit},
            )

        # Session budget cap
        used = session_counts[service][session_id]
        if used >= session_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "session budget exceeded",
                    "service": service,
                    "used": used,
                    "limit": session_limit,
                },
            )

        # Record
        window.append(now)
        session_counts[service][session_id] += 1

    return None


def _budget_headers(service: str, session_id: str) -> dict[str, str]:
    """Return warning header if session usage >= 80% of budget."""
    limit = LIMITS[service]["session"]
    used = session_counts[service][session_id]
    if used >= limit * 0.8:
        return {"X-Budget-Warning": "true"}
    return {}


# ---------------------------------------------------------------------------
# Proxy handler
# ---------------------------------------------------------------------------

async def _proxy(
    request: Request,
    service: str,
    upstream_url: str,
    path: str,
) -> Response:
    session_id = request.headers.get("X-Session-ID", "default")

    # Rate-limit check
    limit_response = await _check_and_record(service, session_id)
    if limit_response is not None:
        _log(service, path, session_id, 429)
        return limit_response

    # Forward headers
    forward_headers = {}
    if "authorization" in request.headers:
        forward_headers["Authorization"] = request.headers["authorization"]
    if "content-type" in request.headers:
        forward_headers["Content-Type"] = request.headers["content-type"]

    body = await request.body()

    extra_headers = _budget_headers(service, session_id)

    # TAMU always returns SSE — use streaming for both services for safety
    async def _stream_upstream():
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                upstream_url,
                headers=forward_headers,
                content=body,
            ) as upstream_resp:
                status = upstream_resp.status_code
                _log(service, path, session_id, status)
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk

    # We need the status code before streaming; peek with a non-streaming
    # request only when NOT hitting TAMU (which mandates streaming).
    # For simplicity — and because TAMU mandates SSE — always stream.
    # Status code is logged inside the generator on first response.
    return StreamingResponse(
        _stream_upstream(),
        media_type="text/event-stream" if service == "tamu" else "application/json",
        headers=extra_headers,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/tamu/{path:path}")
async def tamu_proxy(path: str, request: Request):
    upstream_url = f"{TAMU_BASE}/{path}"
    return await _proxy(request, "tamu", upstream_url, path)


@app.post("/voyage/{path:path}")
async def voyage_proxy(path: str, request: Request):
    upstream_url = f"{VOYAGE_BASE}/{path}"
    return await _proxy(request, "voyage", upstream_url, path)
