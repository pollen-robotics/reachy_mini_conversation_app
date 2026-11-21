"""Where a conversation's long-term memory lives: the local JSON store, or the speech server's own memory.

REACHY_MINI_MEMORY_BACKEND is "auto" (the default), "local" or "server". With "auto" the app asks the speech server at
each session start whether it keeps a memory itself (the "memory" section of its /v1/realtime/capabilities) and uses
it if so. A remote connection, an unreachable server or a server without memory means the local store, as before.
"""

from __future__ import annotations
import logging
from urllib.parse import urlsplit, urlunsplit

import httpx

from reachy_mini_conversation_app.config import (
    AUTO_MEMORY_BACKEND,
    LOCAL_MEMORY_BACKEND,
    SERVER_MEMORY_BACKEND,
    HF_LOCAL_CONNECTION_MODE,
    config,
    get_hf_direct_ws_url,
)


logger = logging.getLogger(__name__)

# What "auto" resolved to at the last session start; local until a session has asked the server.
_resolved = LOCAL_MEMORY_BACKEND


async def server_memory_status(timeout_s: float = 3.0) -> dict[str, object]:
    """Return the speech server's memory status from its capabilities endpoint."""
    ws_url = get_hf_direct_ws_url()
    if config.HF_REALTIME_CONNECTION_MODE != HF_LOCAL_CONNECTION_MODE or not ws_url:
        return {"reachable": False, "reason": "no local speech server is configured"}
    parts = urlsplit(ws_url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    url = urlunsplit((scheme, parts.netloc, "/v1/realtime/capabilities", "", ""))
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
            capabilities = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Failed to read the speech server's memory status: %s", exc)
        return {"reachable": False, "reason": str(exc)}
    memory = capabilities.get("memory") if isinstance(capabilities, dict) else None
    return {"reachable": True, **(memory if isinstance(memory, dict) else {"enabled": False})}


def effective_memory_backend() -> str:
    """Return the memory this conversation uses: the configured backend, or what "auto" resolved to."""
    if config.MEMORY_BACKEND == AUTO_MEMORY_BACKEND:
        return _resolved
    return config.MEMORY_BACKEND


async def resolve_memory_backend() -> str:
    """At a session start: with "auto", use the speech server's memory when it reports one, else the local store."""
    global _resolved
    if config.MEMORY_BACKEND != AUTO_MEMORY_BACKEND:
        return config.MEMORY_BACKEND
    status = await server_memory_status(timeout_s=2.0)
    _resolved = (
        SERVER_MEMORY_BACKEND if status.get("reachable") and status.get("enabled") is True else LOCAL_MEMORY_BACKEND
    )
    return _resolved
