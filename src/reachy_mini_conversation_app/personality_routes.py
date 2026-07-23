"""Personality and voice management, exposed over JSON-RPC.

Each operation lives on :class:`PersonalityOps` (transport-agnostic).
``build_personality_ops`` constructs it and ``register_personality_methods``
registers the ops as ``personalities.*`` / ``voices.*`` methods on a
:class:`~reachy_mini.apps.jsonrpc_server.JsonRpcServer`, so the local UI and
remote WebRTC clients drive personalities through one control surface.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable, Optional, Awaitable, Coroutine

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from .config import (
    LOCKED_PROFILE,
    config,
    get_default_voice,
    get_available_voices,
)
from .avatars import avatar_id_for, read_avatar_svg
from .personality import (
    DEFAULT_OPTION,
    _sanitize_name,
    _write_profile,
    read_tools_for,
    read_greeting_for,
    delete_personality,
    list_personalities,
    available_tools_for,
    resolve_profile_dir,
    read_instructions_for,
)
from .conversation_handler import ConversationHandler


logger = logging.getLogger(__name__)


class RouteError(Exception):
    """A domain error carrying a stable ``reason`` and extras.

    Rendered as a JSON-RPC error with ``data.reason == reason`` (and any
    ``extra`` merged into ``data``).
    """

    def __init__(
        self,
        reason: str,
        *,
        extra: Optional[dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> None:
        """Build the error (``reason`` doubles as the message if none given)."""
        super().__init__(message or reason)
        self.reason = reason
        self.extra = extra or {}
        self.message = message or reason


class PersonalityOps:
    """Transport-agnostic personality/voice operations (return plain dicts)."""

    def __init__(
        self,
        handler: ConversationHandler,
        get_loop: Callable[[], asyncio.AbstractEventLoop | None],
        *,
        persist_personality: Callable[[Optional[str], Optional[str]], None] | None = None,
        get_persisted_personality: Callable[[], Optional[str]] | None = None,
        apply_personality: Callable[[Optional[str]], Awaitable[str]] | None = None,
        get_voices: Callable[[], Awaitable[list[str]]] | None = None,
        get_current_voice: Callable[[], str] | None = None,
        change_voice: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        """Capture the handler and the LocalStream callbacks the ops delegate to."""
        self._handler = handler
        self._get_loop = get_loop
        self._persist_personality = persist_personality
        self._get_persisted_personality = get_persisted_personality
        self._apply_personality = apply_personality
        self._get_voices = get_voices
        self._get_current_voice = get_current_voice
        self._change_voice = change_voice
        self._startup_choice: Any = self._configured_startup_choice()

    # -- startup/current choice helpers -----------------------------------

    def _configured_startup_choice(self) -> Any:
        try:
            if self._get_persisted_personality is not None:
                stored = self._get_persisted_personality()
                if stored:
                    return stored
            env_val = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
            if env_val:
                return env_val
        except Exception as e:
            logger.warning("Failed to read configured startup personality: %s", e)
        return DEFAULT_OPTION

    def _startup_choice_value(self) -> Any:
        try:
            if self._get_persisted_personality is not None:
                stored = self._get_persisted_personality()
                if stored:
                    return stored
        except Exception as e:
            logger.warning("Failed to read persisted startup personality: %s", e)
        return self._startup_choice

    def _set_startup_choice(self, selected_name: str) -> None:
        self._startup_choice = DEFAULT_OPTION if selected_name == DEFAULT_OPTION else selected_name

    def _current_choice(self) -> str:
        try:
            cur = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
            return cur or DEFAULT_OPTION
        except Exception:
            return DEFAULT_OPTION

    def _voice_override(self) -> Optional[str]:
        cb = self._get_current_voice or getattr(self._handler, "get_current_voice", None)
        return cb() if callable(cb) else None

    async def _run_on_loop(self, coro: Coroutine[Any, Any, Any], timeout: float = 10.0) -> Any:
        """Await a coroutine on the LocalStream loop without blocking the caller."""
        loop = self._get_loop()
        if loop is None:
            raise RouteError("loop_unavailable")
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout)

    # -- operations --------------------------------------------------------

    def get_choices(self) -> dict[str, Any]:
        """List personalities plus current/startup selection and lock state."""
        return {
            "choices": [DEFAULT_OPTION, *list_personalities()],
            "current": self._current_choice(),
            "startup": self._startup_choice_value(),
            "locked": LOCKED_PROFILE is not None,
            "locked_to": LOCKED_PROFILE,
        }

    def load(self, name: str) -> dict[str, Any]:
        """Return one personality's instructions, greeting, tools, and voice."""
        instr = read_instructions_for(name)
        tools_txt = read_tools_for(name)
        greeting = read_greeting_for(name)
        voice = get_default_voice()
        uses_default_voice = True
        if name != DEFAULT_OPTION:
            vf = resolve_profile_dir(name) / "voice.txt"
            if vf.exists():
                v = vf.read_text(encoding="utf-8").strip()
                voice = v or get_default_voice()
                uses_default_voice = not bool(v)
        enabled = [ln.strip() for ln in tools_txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        return {
            "instructions": instr,
            "greeting": greeting,
            "tools_text": tools_txt,
            "voice": voice,
            "uses_default_voice": uses_default_voice,
            "available_tools": available_tools_for(name),
            "enabled_tools": enabled,
        }

    def get_all(self) -> dict[str, Any]:
        """Return every personality with its full config in a single call."""
        # No inline SVG: an avatar can be 100+ KB; clients fetch them lazily via
        # personalities.avatar and cache by avatar_id.
        items: list[dict[str, Any]] = []
        for selection in [DEFAULT_OPTION, *list_personalities()]:
            entry = self.load(selection)
            entry["name"] = selection
            entry["avatar_id"] = avatar_id_for(selection)
            items.append(entry)
        return {
            "personalities": items,
            "current": self._current_choice(),
            "startup": self._startup_choice_value(),
            "locked": LOCKED_PROFILE is not None,
            "locked_to": LOCKED_PROFILE,
        }

    def avatar(self, name: str) -> dict[str, Any]:
        """Return the SVG markup for a personality (falls back to the default)."""
        svg = read_avatar_svg(name)
        if svg is None:
            raise RouteError("avatar_unavailable")
        return {"name": name, "avatar_id": avatar_id_for(name), "svg": svg}

    def save(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Create or update a user personality from a raw payload dict."""
        name = str(raw.get("name", ""))
        instructions = str(raw.get("instructions", ""))
        greeting = str(raw["greeting"]) if raw.get("greeting") is not None else None
        tools_text = str(raw.get("tools_text", ""))
        voice = str(raw.get("voice", get_default_voice())) if raw.get("voice") is not None else get_default_voice()
        sanitized_name = _sanitize_name(name)
        if not sanitized_name:
            raise RouteError("invalid_name")
        try:
            _write_profile(sanitized_name, instructions, tools_text, voice or get_default_voice(), greeting)
        except Exception as e:
            raise RouteError(str(e)) from e
        return {
            "ok": True,
            "value": f"user_personalities/{sanitized_name}",
            "choices": [DEFAULT_OPTION, *list_personalities()],
        }

    def delete(self, name: str) -> dict[str, Any]:
        """Delete a user personality (never the active/startup or a built-in one)."""
        choices = [DEFAULT_OPTION, *list_personalities()]
        if name in (self._current_choice(), self._startup_choice_value()):
            raise RouteError("profile_in_use", extra={"choices": choices})
        if not delete_personality(name):
            raise RouteError("not_deletable", extra={"choices": choices})
        return {"ok": True, "choices": [DEFAULT_OPTION, *list_personalities()]}

    async def apply(self, name: str, persist: bool = False) -> dict[str, Any]:
        """Apply a personality (optionally persisting it as the startup choice)."""
        if LOCKED_PROFILE is not None:
            raise RouteError("profile_locked", extra={"locked_to": LOCKED_PROFILE})
        selected_name = name or DEFAULT_OPTION
        persisted_choice = self._startup_choice_value()

        def _persist_if_asked() -> Any:
            nonlocal persisted_choice
            if persist and self._persist_personality is not None:
                try:
                    self._persist_personality(
                        None if selected_name == DEFAULT_OPTION else selected_name,
                        self._voice_override(),
                    )
                    self._set_startup_choice(selected_name)
                    persisted_choice = self._startup_choice_value()
                except Exception as e:
                    logger.warning("Failed to persist startup personality: %s", e)

        if selected_name == self._current_choice():
            _persist_if_asked()
            return {"ok": True, "status": "Personality unchanged.", "startup": persisted_choice}

        async def _do_apply() -> str:
            profile = None if selected_name == DEFAULT_OPTION else selected_name
            if self._apply_personality is not None:
                return await self._apply_personality(profile)
            return await self._handler.apply_personality(profile)

        try:
            status = await self._run_on_loop(_do_apply())
        except RouteError:
            raise
        except Exception as e:
            raise RouteError(str(e)) from e
        _persist_if_asked()
        return {"ok": True, "status": status, "startup": persisted_choice}

    async def voices(self) -> list[str]:
        """Return the voices available for the active backend."""
        if self._get_loop() is None:
            return get_available_voices()  # no session loop yet; use the static catalog
        try:

            async def _get_v() -> list[str]:
                if self._get_voices is not None:
                    return await self._get_voices()
                return await self._handler.get_available_voices()

            return list(await self._run_on_loop(_get_v()))
        except Exception:
            return get_available_voices()

    def current_voice(self) -> dict[str, str]:
        """Return the current voice."""
        try:
            if self._get_current_voice is not None:
                return {"voice": self._get_current_voice()}
            return {"voice": self._handler.get_current_voice()}
        except Exception:
            return {"voice": get_default_voice()}

    async def apply_voice(self, voice: str) -> dict[str, Any]:
        """Change the current voice live (no backend rebuild)."""
        voice = str(voice or "")
        if not voice:
            raise RouteError("missing_voice")

        async def _do() -> str:
            if self._change_voice is not None:
                return await self._change_voice(voice)
            return await self._handler.change_voice(voice)

        try:
            status = await self._run_on_loop(_do())
        except RouteError:
            raise
        except Exception as e:
            raise RouteError(str(e)) from e
        return {"ok": True, "status": status}


def build_personality_ops(
    handler: ConversationHandler,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    **kwargs: Any,
) -> PersonalityOps:
    """Build the shared personality/voice ops (register them with a transport)."""
    return PersonalityOps(handler, get_loop, **kwargs)


def register_personality_methods(rpc: JsonRpcServer, ops: PersonalityOps) -> None:
    """Register the same personality/voice ops as JSON-RPC methods."""

    def _wrap(fn: Callable[..., Any]) -> Callable[[dict[str, Any]], Any]:
        async def _method(params: dict[str, Any]) -> Any:
            try:
                result = fn(params)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except RouteError as e:
                raise JsonRpcError(e.message, reason=e.reason, data=e.extra, code=-32000) from e

        return _method

    rpc.register("personalities.list", _wrap(lambda p: ops.get_choices()))
    rpc.register("personalities.all", _wrap(lambda p: ops.get_all()))
    rpc.register("personalities.load", _wrap(lambda p: ops.load(str(p["name"]))))
    rpc.register("personalities.avatar", _wrap(lambda p: ops.avatar(str(p["name"]))))
    rpc.register("personalities.save", _wrap(lambda p: ops.save(p)))
    rpc.register("personalities.delete", _wrap(lambda p: ops.delete(str(p["name"]))))
    rpc.register(
        "personalities.apply",
        _wrap(lambda p: ops.apply(str(p.get("name", "")), bool(p.get("persist", False)))),
    )
    rpc.register("voices.list", _wrap(lambda p: ops.voices()))
    rpc.register("voices.current", _wrap(lambda p: ops.current_voice()))
    rpc.register("voices.apply", _wrap(lambda p: ops.apply_voice(str(p.get("voice", "")))))
