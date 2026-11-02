"""JSON-RPC methods for the conversation language."""

import re
import logging
from typing import Any, Literal
from dataclasses import asdict
from collections.abc import Callable

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import HF_SPEECH_LANGUAGES, config
from reachy_mini_conversation_app.prompts import get_session_language
from reachy_mini_conversation_app.profile_store import canonical_profile_name


logger = logging.getLogger(__name__)

# Transcription takes ISO-639-1 codes ("en", "fr") or "auto"; a region suffix is not one.
_LANGUAGE_RE = re.compile(r"^(?:[a-z]{2}|auto)$", re.IGNORECASE)

# "profile" saves the language for the active personality only; "all" for every personality.
LanguageScope = Literal["profile", "all"]


def register_language_methods(
    rpc: JsonRpcServer,
    *,
    set_language: Callable[[str, LanguageScope], str],
) -> None:
    """Register language.* methods for the conversation language."""

    def _get(_params: dict[str, Any]) -> dict[str, object]:
        profile = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
        return {
            "language": get_session_language(),
            "default_language": config.REALTIME_TRANSCRIPTION_LANGUAGE,
            "profile": profile,
            "profile_language": config.PROFILE_LANGUAGES.get(profile),
            "choices": [asdict(choice) for choice in HF_SPEECH_LANGUAGES],
        }

    def _set(params: dict[str, Any]) -> dict[str, object]:
        language = params.get("language")
        if not isinstance(language, str) or not _LANGUAGE_RE.match(language.strip()):
            raise JsonRpcError(
                "language must be 'auto' or a two-letter code like 'en' or 'fr'",
                reason="invalid_language",
                code=-32602,
            )
        scope = params.get("scope", "all")
        if scope not in ("profile", "all"):
            raise JsonRpcError("scope must be 'profile' or 'all'", reason="invalid_params", code=-32602)
        message = set_language(language.strip().lower(), scope)
        return {**_get(params), "message": message}

    rpc.register("language.get", _get)
    rpc.register("language.set", _set)
