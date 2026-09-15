"""JSON-RPC methods for the camera (vision) tool switch."""

import logging
from typing import Any
from collections.abc import Callable

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)


def register_vision_methods(
    rpc: JsonRpcServer,
    deps: ToolDependencies,
    persist: Callable[[bool], None],
) -> None:
    """Register vision.* methods toggling the camera tool at runtime.

    Tools read ``deps.camera_enabled`` on every call, so a change takes effect on
    the next one. The camera stays in the model's tool schema and answers
    ``{"error": ...}`` while disabled; removing it entirely is profile_tools.save.
    """

    def _get(_params: dict[str, Any]) -> dict[str, object]:
        return {"enabled": deps.camera_enabled}

    def _set(params: dict[str, Any]) -> dict[str, object]:
        enabled = params.get("enabled")
        if not isinstance(enabled, bool):
            raise JsonRpcError("enabled must be a boolean", reason="invalid_params", code=-32602)
        deps.camera_enabled = enabled
        persist(enabled)
        logger.info("Camera tool %s over /rpc", "enabled" if enabled else "disabled")
        return {"enabled": deps.camera_enabled}

    rpc.register("vision.get", _get)
    rpc.register("vision.set", _set)
