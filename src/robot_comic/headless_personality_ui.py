"""Settings UI routes for headless personality management.

Exposes REST endpoints on the provided FastAPI settings app. The
implementation schedules backend actions (apply personality, fetch voices)
onto the running LocalStream asyncio loop using the supplied get_loop
callable to avoid cross-thread issues.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable, Optional
from pathlib import Path

from fastapi import Query, FastAPI, Request
from pydantic import BaseModel

from .config import (
    LOCKED_PROFILE,
    config,
    get_default_voice_for_provider,
    get_available_voices_for_provider,
)
from robot_comic import telemetry
from .startup_settings import write_startup_settings
from .conversation_handler import ConversationHandler
from .headless_personality import (
    DEFAULT_OPTION,
    _sanitize_name,
    _write_profile,
    read_tools_for,
    list_personalities,
    available_tools_for,
    resolve_profile_dir,
    read_instructions_for,
)


logger = logging.getLogger(__name__)


class ApplyPayload(BaseModel):
    """Request body for POST /personalities/apply."""

    name: str
    persist: Optional[bool] = False


def mount_personality_routes(
    app: FastAPI,
    handler: ConversationHandler,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    *,
    persist_personality: Callable[[Optional[str], Optional[str]], None] | None = None,
    get_persisted_personality: Callable[[], Optional[str]] | None = None,
    instance_path: str | Path | None = None,
) -> None:
    """Register personality management endpoints on a FastAPI app."""
    try:
        from fastapi.responses import JSONResponse
    except Exception:  # pragma: no cover - only when settings app not available
        return

    def _startup_choice() -> Any:
        """Return the persisted startup personality or default."""
        try:
            if get_persisted_personality is not None:
                stored = get_persisted_personality()
                if stored:
                    return stored
            env_val = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
            if env_val:
                return env_val
        except Exception:
            pass
        return DEFAULT_OPTION

    def _current_choice() -> str:
        try:
            cur = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
            return cur or DEFAULT_OPTION
        except Exception:
            return DEFAULT_OPTION

    @app.get("/personalities")
    def _list() -> dict:  # type: ignore
        choices = [DEFAULT_OPTION, *list_personalities()]
        return {
            "choices": choices,
            "current": _current_choice(),
            "startup": _startup_choice(),
            "locked": LOCKED_PROFILE is not None,
            "locked_to": LOCKED_PROFILE,
        }

    @app.get("/personalities/load")
    def _load(name: str) -> dict:  # type: ignore
        instr = read_instructions_for(name)
        tools_txt = read_tools_for(name)
        voice = get_default_voice_for_provider()
        uses_default_voice = True
        if name != DEFAULT_OPTION:
            pdir = resolve_profile_dir(name)
            vf = pdir / "voice.txt"
            if vf.exists():
                v = vf.read_text(encoding="utf-8").strip()
                voice = v or get_default_voice_for_provider()
                uses_default_voice = not bool(v)
        avail = available_tools_for(name)
        enabled = [ln.strip() for ln in tools_txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        return {
            "instructions": instr,
            "tools_text": tools_txt,
            "voice": voice,
            "uses_default_voice": uses_default_voice,
            "available_tools": avail,
            "enabled_tools": enabled,
        }

    @app.post("/personalities/save")
    async def _save(request: Request) -> dict:  # type: ignore
        # Accept raw JSON only to avoid validation-related 422s
        try:
            raw = await request.json()
        except Exception:
            raw = {}
        name = str(raw.get("name", ""))
        instructions = str(raw.get("instructions", ""))
        tools_text = str(raw.get("tools_text", ""))
        voice = (
            str(raw.get("voice", get_default_voice_for_provider()))
            if raw.get("voice") is not None
            else get_default_voice_for_provider()
        )

        name_s = _sanitize_name(name)
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)  # type: ignore
        try:
            logger.info(
                "Headless save: name=%r voice=%r instr_len=%d tools_len=%d",
                name_s,
                voice,
                len(instructions),
                len(tools_text),
            )
            _write_profile(name_s, instructions, tools_text, voice or get_default_voice_for_provider())
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.post("/personalities/save_raw")
    async def _save_raw(
        request: Request,
        name: Optional[str] = None,
        instructions: Optional[str] = None,
        tools_text: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> dict:  # type: ignore
        # Accept query params, form-encoded, or raw JSON
        data = {"name": name, "instructions": instructions, "tools_text": tools_text, "voice": voice}
        # Prefer form if present
        try:
            form = await request.form()
            for k in ("name", "instructions", "tools_text", "voice"):
                if k in form and form[k] is not None:
                    data[k] = str(form[k])
        except Exception:
            pass
        # Try JSON
        try:
            raw = await request.json()
            if isinstance(raw, dict):
                for k in ("name", "instructions", "tools_text", "voice"):
                    if raw.get(k) is not None:
                        data[k] = str(raw.get(k))
        except Exception:
            pass

        name_s = _sanitize_name(str(data.get("name") or ""))
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)  # type: ignore
        instr = str(data.get("instructions") or "")
        tools = str(data.get("tools_text") or "")
        v = str(data.get("voice") or get_default_voice_for_provider())
        try:
            logger.info(
                "Headless save_raw: name=%r voice=%r instr_len=%d tools_len=%d", name_s, v, len(instr), len(tools)
            )
            _write_profile(name_s, instr, tools, v)
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.get("/personalities/save_raw")
    async def _save_raw_get(name: str, instructions: str = "", tools_text: str = "", voice: str | None = None) -> dict:  # type: ignore
        name_s = _sanitize_name(name)
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)  # type: ignore
        try:
            normalized_voice = voice or get_default_voice_for_provider()
            logger.info(
                "Headless save_raw(GET): name=%r voice=%r instr_len=%d tools_len=%d",
                name_s,
                normalized_voice,
                len(instructions),
                len(tools_text),
            )
            _write_profile(name_s, instructions, tools_text, normalized_voice)
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.post("/personalities/apply")
    async def _apply(
        request: Request,
        payload: ApplyPayload | None = None,
        name: str | None = None,
        persist: Optional[bool] = None,
    ) -> dict:  # type: ignore
        # Capture the active persona *before* doing anything so the span attrs
        # reflect the real before/after pair. ``_current_choice()`` reads the
        # in-process config and falls back to DEFAULT_OPTION on error.
        from_persona = _current_choice()

        # Open a ``persona.switch`` span around the entire request so operators
        # can see in the boot timeline / trace stream when a persona swap was
        # requested, who initiated it, and how it resolved (#303). The span is
        # tagged ``event.kind=supporting`` so the monitor TUI can render it on
        # the same supporting-events lane introduced by #301/#321 once that
        # plumbing lands on main.
        tracer = telemetry.get_tracer()
        with tracer.start_as_current_span(
            "persona.switch",
            attributes={
                "event.kind": "supporting",
                "from_persona": from_persona,
            },
        ) as _switch_span:
            if LOCKED_PROFILE is not None:
                _switch_span.set_attribute("to_persona", from_persona)
                _switch_span.set_attribute("outcome", "locked")
                return JSONResponse(
                    {"ok": False, "error": "profile_locked", "locked_to": LOCKED_PROFILE},
                    status_code=403,
                )  # type: ignore
            loop = get_loop()
            if loop is None:
                _switch_span.set_attribute("to_persona", from_persona)
                _switch_span.set_attribute("outcome", "loop_unavailable")
                return JSONResponse({"ok": False, "error": "loop_unavailable"}, status_code=503)  # type: ignore

            # Accept both JSON payload and query param for convenience
            sel_name: Optional[str] = None
            persist_flag = bool(persist) if persist is not None else False
            if payload and getattr(payload, "name", None):
                sel_name = payload.name
                persist_flag = bool(getattr(payload, "persist", False))
            elif name:
                sel_name = name
            else:
                try:
                    body = await request.json()
                    if isinstance(body, dict) and body.get("name"):
                        sel_name = str(body.get("name"))
                    if isinstance(body, dict) and "persist" in body:
                        persist_flag = bool(body.get("persist"))
                except Exception:
                    sel_name = None
            try:
                q_persist = request.query_params.get("persist")
                if q_persist is not None:
                    persist_flag = str(q_persist).lower() in {"1", "true", "yes", "on"}
            except Exception:
                pass
            if not sel_name:
                sel_name = DEFAULT_OPTION

            _switch_span.set_attribute("to_persona", sel_name)

            async def _do_apply() -> tuple[str, Optional[str]]:
                sel = None if sel_name == DEFAULT_OPTION else sel_name
                # Phase 5d: ``apply_personality`` is no longer on the
                # ``ConversationHandler`` ABC (moved onto ``ComposablePipeline``
                # in 5c.2; bundled-realtime handlers keep their own impl).
                # Reach via ``getattr`` so concrete handlers that still expose
                # it are called and any future handler without it returns a
                # clear error rather than an AttributeError.
                apply_personality = getattr(handler, "apply_personality", None)
                if not callable(apply_personality):
                    return ("Handler does not support personality switching.", None)
                status = await apply_personality(sel)
                get_current_voice = getattr(handler, "get_current_voice", None)
                voice_override = get_current_voice() if callable(get_current_voice) else None
                return status, voice_override

            try:
                logger.info("Headless apply: requested name=%r", sel_name)
                fut = asyncio.run_coroutine_threadsafe(_do_apply(), loop)
                status, voice_override = fut.result(timeout=10)
                persisted_choice = _startup_choice()
                if persist_flag and persist_personality is not None:
                    try:
                        persist_personality(None if sel_name == DEFAULT_OPTION else sel_name, voice_override)
                        persisted_choice = _startup_choice()
                    except Exception as e:
                        logger.warning("Failed to persist startup personality: %s", e)
                _switch_span.set_attribute("outcome", "success")
                return {"ok": True, "status": status, "startup": persisted_choice}
            except Exception as e:
                _switch_span.set_attribute("outcome", "error")
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.get("/voices")
    async def _voices() -> list[str]:
        loop = get_loop()
        if loop is None:
            return get_available_voices_for_provider()

        async def _get_v() -> list[str]:
            # Phase 5d: ``get_available_voices`` is no longer on the ABC
            # (moved onto ``TTSBackend`` in 5c.1). Duck-type the call so
            # mypy stays clean and any future handler without the method
            # falls back to the static provider catalog.
            get_available_voices = getattr(handler, "get_available_voices", None)
            if not callable(get_available_voices):
                return get_available_voices_for_provider()
            try:
                voices = await get_available_voices()
                return list(voices) if voices is not None else get_available_voices_for_provider()
            except Exception:
                return get_available_voices_for_provider()

        try:
            fut = asyncio.run_coroutine_threadsafe(_get_v(), loop)
            return fut.result(timeout=10)
        except Exception:
            return get_available_voices_for_provider()

    @app.get("/voices/current")
    async def _current_voice() -> dict[str, str]:
        loop = get_loop()
        fallback_voice = get_default_voice_for_provider()
        if loop is None:
            return {"voice": fallback_voice}

        def _get_current() -> str:
            # Phase 5d: ``get_current_voice`` is no longer on the ABC
            # (moved onto ``TTSBackend`` in 5c.1). Duck-type so concrete
            # handlers that still expose it are called and others fall back.
            get_current_voice = getattr(handler, "get_current_voice", None)
            if not callable(get_current_voice):
                return fallback_voice
            try:
                result = get_current_voice()
                return str(result) if result is not None else fallback_voice
            except Exception:
                return fallback_voice

        try:
            fut = asyncio.run_coroutine_threadsafe(asyncio.to_thread(_get_current), loop)
            return {"voice": fut.result(timeout=10)}
        except Exception:
            return {"voice": fallback_voice}

    @app.post("/voices/apply")
    async def _apply_voice(request: Request, voice: str | None = Query(None)) -> dict:  # type: ignore
        voice = str(voice or "")
        if not voice:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            voice = str(raw.get("voice", "") or "")
        if not voice:
            return JSONResponse({"ok": False, "error": "missing_voice"}, status_code=400)  # type: ignore
        loop = get_loop()
        if loop is None:
            return JSONResponse({"ok": False, "error": "loop_unavailable"}, status_code=503)  # type: ignore

        async def _do() -> str:
            # Phase 5d: ``change_voice`` is no longer on the ABC (moved onto
            # ``TTSBackend`` in 5c.1). Duck-type so callers get a clear
            # response when a handler genuinely lacks voice switching.
            change_voice = getattr(handler, "change_voice", None)
            if not callable(change_voice):
                return "Handler does not support voice switching."
            result = await change_voice(voice)
            return str(result) if result is not None else "Voice change requested."

        try:
            fut = asyncio.run_coroutine_threadsafe(_do(), loop)
            status = fut.result(timeout=10)
            try:
                write_startup_settings(instance_path, voice=voice)
            except Exception as e:
                logger.warning("Failed to persist voice to startup_settings: %s", e)
            return {"ok": True, "status": status}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore
