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

from .config import LOCKED_PROFILE, config
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


def mount_personality_routes(
    app: FastAPI,
    handler: OpenaiRealtimeHandler,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    *,
    persist_personality: Callable[[Optional[str]], None] | None = None,
    get_persisted_personality: Callable[[], Optional[str]] | None = None,
) -> None:
    """Register personality management endpoints on a FastAPI app."""
    try:
        from fastapi import Request
        from pydantic import BaseModel
        from fastapi.responses import JSONResponse
    except Exception:  # pragma: no cover - only when settings app not available
        return

    class SavePayload(BaseModel):
        name: str
        instructions: str
        tools_text: str
        voice: Optional[str] = "cedar"

    class ApplyPayload(BaseModel):
        name: str
        persist: Optional[bool] = False

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
    async def _save(request: Request) -> dict:  # type: ignore
        # Accept raw JSON only to avoid validation-related 422s
        try:
            raw = await request.json()
        except Exception:
            raw = {}
        name = str(raw.get("name", ""))
        instructions = str(raw.get("instructions", ""))
        tools_text = str(raw.get("tools_text", ""))
        voice = str(raw.get("voice", "cedar")) if raw.get("voice") is not None else "cedar"

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
            _write_profile(name_s, instructions, tools_text, voice or "cedar")
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
        v = str(data.get("voice") or "cedar")
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
    async def _save_raw_get(name: str, instructions: str = "", tools_text: str = "", voice: str = "cedar") -> dict:  # type: ignore
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

    logger = logging.getLogger(__name__)

    @app.post("/personalities/apply")
    async def _apply(
        payload: ApplyPayload | None = None,
        name: str | None = None,
        persist: Optional[bool] = None,
        request: Optional[Request] = None,
    ) -> dict:  # type: ignore
        if LOCKED_PROFILE is not None:
            return JSONResponse(
                {"ok": False, "error": "profile_locked", "locked_to": LOCKED_PROFILE},
                status_code=403,
            )  # type: ignore
        loop = get_loop()
        if loop is None:
            return JSONResponse({"ok": False, "error": "loop_unavailable"}, status_code=503)  # type: ignore

        # Accept both JSON payload and query param for convenience
        sel_name: Optional[str] = None
        persist_flag = bool(persist) if persist is not None else False
        if payload and getattr(payload, "name", None):
            sel_name = payload.name
            persist_flag = bool(getattr(payload, "persist", False))
        elif name:
            sel_name = name
        elif request is not None:
            try:
                body = await request.json()
                if isinstance(body, dict) and body.get("name"):
                    sel_name = str(body.get("name"))
                if isinstance(body, dict) and "persist" in body:
                    persist_flag = bool(body.get("persist"))
            except Exception:
                sel_name = None
        if request is not None:
            try:
                q_persist = request.query_params.get("persist")
                if q_persist is not None:
                    persist_flag = str(q_persist).lower() in {"1", "true", "yes", "on"}
            except Exception:
                pass
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
    async def _voices() -> list[str]:
        loop = get_loop()
        if loop is None:
            return ["cedar"]

        async def _get_v() -> list[str]:
            try:
                return await handler.get_available_voices()
            except Exception:
                return ["cedar"]

        try:
            fut = asyncio.run_coroutine_threadsafe(_get_v(), loop)
            return fut.result(timeout=10)
        except Exception:
            return ["cedar"]
