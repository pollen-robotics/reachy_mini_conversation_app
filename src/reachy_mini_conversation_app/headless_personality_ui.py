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

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from .config import config
from .openai_realtime import OpenaiRealtimeHandler
from .headless_personality import (
    DEFAULT_OPTION,
    _sanitize_name,
    _write_profile,
    list_personalities,
    available_tools_for,
    resolve_profile_dir,
    read_instructions_for,
)


logger = logging.getLogger(__name__)


class SavePayload(BaseModel):
    """Payload for saving a personality profile."""

    name: str
    instructions: str = ""
    tools_text: str = ""
    voice: Optional[str] = "cedar"


class ApplyPayload(BaseModel):
    """Payload for applying a personality profile."""

    name: str
    persist: Optional[bool] = False


def mount_personality_routes(
    app: FastAPI,
    handler: OpenaiRealtimeHandler,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    *,
    persist_personality: Callable[[Optional[str]], None] | None = None,
    get_persisted_personality: Callable[[], Optional[str]] | None = None,
) -> None:
    """Register personality management endpoints on a FastAPI app."""

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
        return {"choices": choices, "current": _current_choice(), "startup": _startup_choice()}

    @app.get("/personalities/load")
    def _load(name: str) -> dict:  # type: ignore
        instr = read_instructions_for(name)
        tools_txt = ""
        voice = "cedar"
        if name != DEFAULT_OPTION:
            pdir = resolve_profile_dir(name)
            tp = pdir / "tools.txt"
            if tp.exists():
                tools_txt = tp.read_text(encoding="utf-8")
            vf = pdir / "voice.txt"
            if vf.exists():
                v = vf.read_text(encoding="utf-8").strip()
                voice = v or "cedar"
        avail = available_tools_for(name)
        enabled = [ln.strip() for ln in tools_txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        return {
            "instructions": instr,
            "tools_text": tools_txt,
            "voice": voice,
            "available_tools": avail,
            "enabled_tools": enabled,
        }

    @app.post("/personalities/save")
    def _save(payload: SavePayload) -> dict:  # type: ignore
        name_s = _sanitize_name(payload.name)
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)  # type: ignore
        try:
            voice = payload.voice or "cedar"
            logger.info(
                "Headless save: name=%r voice=%r instr_len=%d tools_len=%d",
                name_s,
                voice,
                len(payload.instructions),
                len(payload.tools_text),
            )
            _write_profile(name_s, payload.instructions, payload.tools_text, voice)
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.post("/personalities/save_raw")
    def _save_raw_post(
        name: str,
        instructions: str = "",
        tools_text: str = "",
        voice: str = "cedar",
    ) -> dict:  # type: ignore
        name_s = _sanitize_name(name)
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)  # type: ignore
        try:
            logger.info(
                "Headless save_raw(POST): name=%r voice=%r instr_len=%d tools_len=%d",
                name_s,
                voice,
                len(instructions),
                len(tools_text),
            )
            _write_profile(name_s, instructions, tools_text, voice or "cedar")
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.get("/personalities/save_raw")
    def _save_raw_get(name: str, instructions: str = "", tools_text: str = "", voice: str = "cedar") -> dict:  # type: ignore
        name_s = _sanitize_name(name)
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)  # type: ignore
        try:
            logger.info(
                "Headless save_raw(GET): name=%r voice=%r instr_len=%d tools_len=%d",
                name_s,
                voice,
                len(instructions),
                len(tools_text),
            )
            _write_profile(name_s, instructions, tools_text, voice or "cedar")
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.post("/personalities/apply")
    async def _apply(
        payload: ApplyPayload | None = None,
        name: str | None = None,
        persist: Optional[bool] = None,
    ) -> dict:  # type: ignore
        loop = get_loop()
        if loop is None:
            return JSONResponse({"ok": False, "error": "loop_unavailable"}, status_code=503)  # type: ignore

        # Accept both JSON payload and query param for convenience
        sel_name: Optional[str] = None
        persist_flag = bool(persist) if persist is not None else False
        if payload and getattr(payload, "name", None):
            sel_name = payload.name
            persist_flag = bool(getattr(payload, "persist", False)) if persist is None else persist_flag
        elif name:
            sel_name = name
        if not sel_name:
            sel_name = DEFAULT_OPTION

        async def _do_apply() -> str:
            sel = None if sel_name == DEFAULT_OPTION else sel_name
            status = await handler.apply_personality(sel)
            return status

        try:
            logger.info("Headless apply: requested name=%r", sel_name)
            fut = asyncio.run_coroutine_threadsafe(_do_apply(), loop)
            status = fut.result(timeout=10)
            persisted_choice = _startup_choice()
            if persist_flag and persist_personality is not None:
                try:
                    persist_personality(None if sel_name == DEFAULT_OPTION else sel_name)
                    persisted_choice = _startup_choice()
                except Exception as e:
                    logger.warning("Failed to persist startup personality: %s", e)
            return {"ok": True, "status": status, "startup": persisted_choice}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)  # type: ignore

    @app.get("/voices")
    def _voices() -> list[str]:
        loop = get_loop()
        if loop is None:
            return ["cedar"]

        try:
            coro = handler.get_available_voices()
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            return fut.result(timeout=10)
        except Exception:
            return ["cedar"]
