"""JSON-RPC methods for long-term memory."""

import asyncio
import logging
from typing import Any
from pathlib import Path
from collections.abc import Callable, Awaitable

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.memory import MAX_FACTS, list_memory_facts, clear_memory_facts, forget_memory_fact


logger = logging.getLogger(__name__)


def _str_param(params: dict[str, Any], name: str) -> str | None:
    value = params.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise JsonRpcError(f"{name} must be a non-empty string", reason="invalid_params", code=-32602)
    return value.strip()


def register_memory_methods(
    rpc: JsonRpcServer,
    *,
    instance_path: str | Path | None,
    set_enabled: Callable[[bool], str],
    refresh_instructions: Callable[[], Awaitable[bool]],
) -> None:
    """Register memory.* methods for clients that manage stored facts."""

    async def _list(_params: dict[str, Any]) -> dict[str, object]:
        try:
            facts = await asyncio.to_thread(list_memory_facts, instance_path)
        except OSError as exc:
            logger.exception("Failed to read memory facts")
            raise JsonRpcError(str(exc), reason="memory_unavailable") from exc
        return {
            "facts": [fact.to_json() for fact in facts],
            "max_facts": MAX_FACTS,
            "enabled": config.MEMORY_ENABLED,
        }

    async def _forget(params: dict[str, Any]) -> dict[str, object]:
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
        return {"ok": True, "removed": result.removed.to_json() if result.removed else None}

    async def _clear(_params: dict[str, Any]) -> dict[str, object]:
        try:
            await asyncio.to_thread(clear_memory_facts, instance_path)
        except OSError as exc:
            logger.exception("Failed to clear memory facts")
            raise JsonRpcError(str(exc), reason="memory_unavailable") from exc
        # The facts are in the prompt, so the model only forgets once its
        # instructions are replaced. Best-effort: the store is already empty,
        # and the next session rebuilds them regardless.
        applied = False
        try:
            applied = await refresh_instructions()
        except Exception:
            logger.exception("Cleared memory but could not refresh the live session")
        return {"ok": True, "applied_live": applied}

    async def _set_enabled(params: dict[str, Any]) -> dict[str, object]:
        enabled = params.get("enabled")
        if not isinstance(enabled, bool):
            raise JsonRpcError("enabled must be a boolean", reason="invalid_params", code=-32602)
        message = set_enabled(enabled)
        return {"enabled": config.MEMORY_ENABLED, "message": message}

    rpc.register("memory.list", _list)
    rpc.register("memory.forget", _forget)
    rpc.register("memory.clear", _clear)
    rpc.register("memory.set_enabled", _set_enabled)
