"""JSON-RPC methods for long-term memory."""

import asyncio
import logging
from typing import Any
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from collections.abc import Callable, Awaitable

import httpx

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import (
    SERVER_MEMORY_BACKEND,
    HF_LOCAL_CONNECTION_MODE,
    config,
    get_hf_direct_ws_url,
)
from reachy_mini_conversation_app.memory import MAX_FACTS, list_memory_facts, clear_memory_facts, forget_memory_fact


logger = logging.getLogger(__name__)


def _str_param(params: dict[str, Any], name: str) -> str | None:
    value = params.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise JsonRpcError(f"{name} must be a non-empty string", reason="invalid_params", code=-32602)
    return value.strip()


async def _server_memory_status() -> dict[str, object]:
    """Return the speech server's memory status from its capabilities endpoint."""
    ws_url = get_hf_direct_ws_url()
    if config.HF_REALTIME_CONNECTION_MODE != HF_LOCAL_CONNECTION_MODE or not ws_url:
        return {"reachable": False, "reason": "no local speech server is configured"}
    parts = urlsplit(ws_url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    url = urlunsplit((scheme, parts.netloc, "/v1/realtime/capabilities", "", ""))
    try:
        async with httpx.AsyncClient(timeout=3.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
            capabilities = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Failed to read the speech server's memory status: %s", exc)
        return {"reachable": False, "reason": str(exc)}
    memory = capabilities.get("memory") if isinstance(capabilities, dict) else None
    return {"reachable": True, **(memory if isinstance(memory, dict) else {"enabled": False})}


def _refuse_on_server() -> None:
    if config.MEMORY_BACKEND == SERVER_MEMORY_BACKEND:
        raise JsonRpcError("Memory is kept by the speech server", reason="memory_on_server")


def register_memory_methods(
    rpc: JsonRpcServer,
    *,
    instance_path: str | Path | None,
    set_enabled: Callable[[bool], None],
    refresh_instructions: Callable[[], Awaitable[bool]],
) -> None:
    """Register memory.* methods for clients that manage stored facts."""

    async def _refresh() -> bool:
        # Facts live in the prompt, so the model only sees a change once its
        # instructions are replaced. Best-effort: the store is already written,
        # and the next session rebuilds them regardless.
        try:
            return await refresh_instructions()
        except Exception:
            logger.exception("Memory changed but the live session could not be refreshed")
            return False

    async def _list(_params: dict[str, Any]) -> dict[str, object]:
        if config.MEMORY_BACKEND == SERVER_MEMORY_BACKEND:
            return {
                "facts": [],
                "max_facts": MAX_FACTS,
                "enabled": config.MEMORY_ENABLED,
                "backend": SERVER_MEMORY_BACKEND,
                "server": await _server_memory_status(),
            }
        try:
            facts = await asyncio.to_thread(list_memory_facts, instance_path)
        except OSError as exc:
            logger.exception("Failed to read memory facts")
            raise JsonRpcError(str(exc), reason="memory_unavailable") from exc
        return {
            "facts": [fact.to_json() for fact in facts],
            "max_facts": MAX_FACTS,
            "enabled": config.MEMORY_ENABLED,
            "backend": config.MEMORY_BACKEND,
        }

    async def _forget(params: dict[str, Any]) -> dict[str, object]:
        _refuse_on_server()
        fact_id = _str_param(params, "id")
        query = _str_param(params, "query")
        if fact_id is None and query is None:
            raise JsonRpcError("forget requires 'id' or 'query'", reason="invalid_params", code=-32602)
        try:
            result = await asyncio.to_thread(
                forget_memory_fact,
                instance_path,
                query=query,
                fact_id=fact_id,
            )
        except OSError as exc:
            logger.exception("Failed to forget a memory fact")
            raise JsonRpcError(str(exc), reason="memory_unavailable") from exc
        applied = await _refresh() if result.removed else False
        return {
            "ok": True,
            "removed": result.removed.to_json() if result.removed else None,
            "applied_live": applied,
        }

    async def _clear(_params: dict[str, Any]) -> dict[str, object]:
        _refuse_on_server()
        try:
            await asyncio.to_thread(clear_memory_facts, instance_path)
        except OSError as exc:
            logger.exception("Failed to clear memory facts")
            raise JsonRpcError(str(exc), reason="memory_unavailable") from exc
        return {"ok": True, "applied_live": await _refresh()}

    async def _set_enabled(params: dict[str, Any]) -> dict[str, object]:
        _refuse_on_server()
        enabled = params.get("enabled")
        if not isinstance(enabled, bool):
            raise JsonRpcError("enabled must be a boolean", reason="invalid_params", code=-32602)
        set_enabled(enabled)
        return {"enabled": config.MEMORY_ENABLED, "applied_live": await _refresh()}

    rpc.register("memory.list", _list)
    rpc.register("memory.forget", _forget)
    rpc.register("memory.clear", _clear)
    rpc.register("memory.set_enabled", _set_enabled)
