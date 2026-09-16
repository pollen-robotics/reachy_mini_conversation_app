"""JSON-RPC methods for the speech transcription language."""

import re
import logging
from typing import Any
from collections.abc import Callable

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

# Matches the language ids the clients use ("en", "fr", "pt-BR").
_LANGUAGE_RE = re.compile(r"^[a-zA-Z]{2}(-[a-zA-Z]{2})?$")


def register_language_methods(
    rpc: JsonRpcServer,
    *,
    set_language: Callable[[str], str],
) -> None:
    """Register language.* methods for the speech transcription language."""

    def _get(_params: dict[str, Any]) -> dict[str, object]:
        return {"language": config.REALTIME_TRANSCRIPTION_LANGUAGE}

    def _set(params: dict[str, Any]) -> dict[str, object]:
        language = params.get("language")
        if not isinstance(language, str) or not _LANGUAGE_RE.match(language.strip()):
            raise JsonRpcError(
                "language must be a code like 'en' or 'pt-BR'",
                reason="invalid_language",
                code=-32602,
            )
        message = set_language(language.strip().lower())
        return {"language": config.REALTIME_TRANSCRIPTION_LANGUAGE, "message": message}

    rpc.register("language.get", _get)
    rpc.register("language.set", _set)
