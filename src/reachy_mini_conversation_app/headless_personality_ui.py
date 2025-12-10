"""Settings UI routes for headless personality management.

Exposes REST endpoints on the provided FastAPI settings app. The
implementation schedules backend actions (apply personality, fetch voices)
onto the running LocalStream asyncio loop using the supplied get_loop
callable to avoid cross-thread issues.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Dict, List, Optional

import logging

from .headless_personality import (
    DEFAULT_OPTION,
    _sanitize_name,
    available_tools_for,
    list_personalities,
    read_instructions_for,
    resolve_profile_dir,
    _write_profile,
)


def mount_personality_routes(app, handler, get_loop: Callable[[], asyncio.AbstractEventLoop | None]) -> None:
    """Register personality management endpoints on a FastAPI app.

    Parameters
    - app: FastAPI instance
    - handler: OpenaiRealtimeHandler instance used to apply personalities and fetch voices
    - get_loop: callable returning the asyncio loop running the audio/handler tasks
    """

    try:
        from pydantic import BaseModel
        from fastapi import Request
        from fastapi.responses import JSONResponse
    except Exception:  # pragma: no cover - only when settings app not available
        return

    class SavePayload(BaseModel):  # type: ignore[misc,valid-type]
        name: str
        instructions: str
        tools_text: str
        voice: Optional[str] = "cedar"

    class ApplyPayload(BaseModel):  # type: ignore[misc,valid-type]
        name: str

    @app.get("/personalities")  # type: ignore[union-attr]
    def _list():  # type: ignore[no-redef]
        choices = [DEFAULT_OPTION, *list_personalities()]
        return {"choices": choices}

    @app.get("/personalities/load")  # type: ignore[union-attr]
    def _load(name: str):  # type: ignore[no-redef]
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

    @app.post("/personalities/save")  # type: ignore[union-attr]
    async def _save(request: Request):  # type: ignore[no-redef]
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
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)
        try:
            logger.info("Headless save: name=%r voice=%r instr_len=%d tools_len=%d", name_s, voice, len(instructions), len(tools_text))
            _write_profile(name_s, instructions, tools_text, voice or "cedar")
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.post("/personalities/save_raw")  # type: ignore[union-attr]
    async def _save_raw(request: Request, name: Optional[str] = None, instructions: Optional[str] = None, tools_text: Optional[str] = None, voice: Optional[str] = None):  # type: ignore[no-redef]
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
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)
        instr = str(data.get("instructions") or "")
        tools = str(data.get("tools_text") or "")
        v = str(data.get("voice") or "cedar")
        try:
            logger.info("Headless save_raw: name=%r voice=%r instr_len=%d tools_len=%d", name_s, v, len(instr), len(tools))
            _write_profile(name_s, instr, tools, v)
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.get("/personalities/save_raw")  # type: ignore[union-attr]
    async def _save_raw_get(name: str, instructions: str = "", tools_text: str = "", voice: str = "cedar"):  # type: ignore[no-redef]
        name_s = _sanitize_name(name)
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)
        try:
            logger.info("Headless save_raw(GET): name=%r voice=%r instr_len=%d tools_len=%d", name_s, voice, len(instructions), len(tools_text))
            _write_profile(name_s, instructions, tools_text, voice or "cedar")
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    logger = logging.getLogger(__name__)

    @app.post("/personalities/apply")  # type: ignore[union-attr]
    async def _apply(payload: ApplyPayload | None = None, name: str | None = None, request: Optional[Request] = None):  # type: ignore[no-redef]
        loop = get_loop()
        if loop is None:
            return JSONResponse({"ok": False, "error": "loop_unavailable"}, status_code=503)

        # Accept both JSON payload and query param for convenience
        sel_name: Optional[str] = None
        if payload and getattr(payload, "name", None):
            sel_name = payload.name
        elif name:
            sel_name = name
        elif request is not None:
            try:
                body = await request.json()
                if isinstance(body, dict) and body.get("name"):
                    sel_name = str(body.get("name"))
            except Exception:
                sel_name = None
        if not sel_name:
            sel_name = DEFAULT_OPTION

        async def _do_apply():
            sel = None if sel_name == DEFAULT_OPTION else sel_name
            status = await handler.apply_personality(sel)
            return status

        try:
            logger.info("Headless apply: requested name=%r", sel_name)
            fut = asyncio.run_coroutine_threadsafe(_do_apply(), loop)
            status = fut.result(timeout=10)
            return {"ok": True, "status": status}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.get("/voices")  # type: ignore[union-attr]
    async def _voices():  # type: ignore[no-redef]
        loop = get_loop()
        if loop is None:
            return ["cedar"]

        async def _get_v():
            try:
                return await handler.get_available_voices()
            except Exception:
                return ["cedar"]

        try:
            fut = asyncio.run_coroutine_threadsafe(_get_v(), loop)
            return fut.result(timeout=10)
        except Exception:
            return ["cedar"]
