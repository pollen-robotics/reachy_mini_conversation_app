"""Settings UI routes for headless personality management.

Exposes REST endpoints on the provided FastAPI settings app. The
implementation schedules backend actions (apply personality, fetch voices)
onto the running LocalStream asyncio loop using the supplied get_loop
callable to avoid cross-thread issues.
"""

from __future__ import annotations
import os
import asyncio
import logging
from typing import Any, Callable, Optional
from pathlib import Path

from fastapi import FastAPI

from .config import config, reload_config
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


# Re-export for test patching
__all__ = [
    "mount_personality_routes",
    "CONFIG_VARS",
    "DEFAULT_OPTION",
    "_sanitize_name",
    "_write_profile",
    "list_personalities",
    "available_tools_for",
    "resolve_profile_dir",
    "read_instructions_for",
]


# Configuration variables that can be managed via the UI
# Format: (env_var_name, config_attr_name, is_secret, description)
CONFIG_VARS = [
    ("OPENAI_API_KEY", "OPENAI_API_KEY", True, "OpenAI API key (required for voice)"),
    ("MODEL_NAME", "MODEL_NAME", False, "OpenAI model name"),
    ("HF_TOKEN", "HF_TOKEN", True, "Hugging Face token (optional, for vision)"),
    ("HF_HOME", "HF_HOME", False, "Hugging Face cache directory"),
    ("LOCAL_VISION_MODEL", "LOCAL_VISION_MODEL", False, "Local vision model path"),
    ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", True, "Anthropic API key (for Linus profile)"),
    ("ANTHROPIC_MODEL", "ANTHROPIC_MODEL", False, "Anthropic model name"),
    ("GITHUB_TOKEN", "GITHUB_TOKEN", True, "GitHub token (for Linus profile)"),
    ("GITHUB_DEFAULT_OWNER", "GITHUB_DEFAULT_OWNER", False, "Default GitHub owner/org"),
    ("GITHUB_OWNER_EMAIL", "GITHUB_OWNER_EMAIL", False, "Email for git commits"),
    ("REACHY_MINI_CUSTOM_PROFILE", "REACHY_MINI_CUSTOM_PROFILE", False, "Custom profile name"),
]


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
        from fastapi import Body, Request
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
    def _list() -> dict[str, Any]:
        choices = [DEFAULT_OPTION, *list_personalities()]
        return {"choices": choices, "current": _current_choice(), "startup": _startup_choice()}

    @app.get("/personalities/load")
    def _load(name: str) -> dict[str, Any]:
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
    def _save(
        name: str = Body(..., embed=True),
        instructions: str = Body("", embed=True),
        tools_text: str = Body("", embed=True),
        voice: Optional[str] = Body("cedar", embed=True),
    ) -> dict[str, Any] | JSONResponse:
        name_s = _sanitize_name(name)
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)
        v = voice or "cedar"
        try:
            logger.info(
                "Headless save: name=%r voice=%r instr_len=%d tools_len=%d",
                name_s,
                v,
                len(instructions),
                len(tools_text),
            )
            _write_profile(name_s, instructions, tools_text, v)
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.post("/personalities/save_raw")
    def _save_raw(
        name: str = Body(..., embed=True),
        instructions: str = Body("", embed=True),
        tools_text: str = Body("", embed=True),
        voice: Optional[str] = Body("cedar", embed=True),
    ) -> dict[str, Any] | JSONResponse:
        name_s = _sanitize_name(name)
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)
        v = voice or "cedar"
        try:
            logger.info(
                "Headless save_raw: name=%r voice=%r instr_len=%d tools_len=%d", name_s, v, len(instructions), len(tools_text)
            )
            _write_profile(name_s, instructions, tools_text, v)
            value = f"user_personalities/{name_s}"
            choices = [DEFAULT_OPTION, *list_personalities()]
            return {"ok": True, "value": value, "choices": choices}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.get("/personalities/save_raw")
    async def _save_raw_get(
        name: str, instructions: str = "", tools_text: str = "", voice: str = "cedar"
    ) -> dict[str, Any] | JSONResponse:
        name_s = _sanitize_name(name)
        if not name_s:
            return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)
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
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    logger = logging.getLogger(__name__)

    @app.post("/personalities/apply")
    async def _apply(
        payload: ApplyPayload | None = None,
        name: str | None = None,
        persist: Optional[bool] = None,
        request: Optional[Request] = None,
    ) -> dict[str, Any] | JSONResponse:
        loop = get_loop()
        if loop is None:
            return JSONResponse({"ok": False, "error": "loop_unavailable"}, status_code=503)

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
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

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

    # ========== Configuration Management Endpoints ==========

    def _mask_secret(value: str | None, is_secret: bool) -> str | None:
        """Mask secret values for display."""
        if value is None or not value:
            return None
        if not is_secret:
            return value
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"

    def _get_env_file_path() -> Path | None:
        """Find the .env file path."""
        from dotenv import find_dotenv
        dotenv_path = find_dotenv(usecwd=True)
        return Path(dotenv_path) if dotenv_path else None

    def _update_env_file(key: str, value: str | None) -> bool:
        """Update or add a key in the .env file."""
        env_path = _get_env_file_path()
        if env_path is None:
            # Create a new .env in the current directory
            env_path = Path.cwd() / ".env"

        try:
            lines = []
            key_found = False

            if env_path.exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                            key_found = True
                            if value is not None and value != "":
                                lines.append(f"{key}={value}\n")
                            # If value is None or empty, we remove the line
                        else:
                            lines.append(line)

            if not key_found and value is not None and value != "":
                lines.append(f"{key}={value}\n")

            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            return True
        except Exception as e:
            logger.error(f"Failed to update .env file: {e}")
            return False

    def _update_runtime_config(key: str, value: str | None) -> None:
        """Update config and environment at runtime."""
        # Update environment variable
        if value is not None and value != "":
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

        # Update config object
        if hasattr(config, key):
            setattr(config, key, value if value else None)

    @app.get("/config")
    def _get_config() -> dict[str, Any]:
        """Get all configuration variables with masked secrets."""
        variables = []
        for env_key, config_attr, is_secret, description in CONFIG_VARS:
            value = getattr(config, config_attr, None)
            variables.append({
                "key": env_key,
                "value": _mask_secret(value, is_secret),
                "is_set": value is not None and value != "",
                "is_secret": is_secret,
                "description": description,
            })
        return {"variables": variables}

    @app.post("/config/reload")
    def _reload_config() -> dict[str, Any] | JSONResponse:
        """Reload configuration from .env file."""
        try:
            reload_config()
            # Return updated config
            variables = []
            for env_key, config_attr, is_secret, description in CONFIG_VARS:
                value = getattr(config, config_attr, None)
                variables.append({
                    "key": env_key,
                    "value": _mask_secret(value, is_secret),
                    "is_set": value is not None and value != "",
                    "is_secret": is_secret,
                    "description": description,
                })
            return {"ok": True, "message": "Configuration reloaded", "variables": variables}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.get("/config/{key}")
    def _get_config_key(key: str) -> dict[str, Any] | JSONResponse:
        """Get a specific configuration variable."""
        for env_key, config_attr, is_secret, description in CONFIG_VARS:
            if env_key == key:
                value = getattr(config, config_attr, None)
                return {
                    "key": env_key,
                    "value": _mask_secret(value, is_secret),
                    "is_set": value is not None and value != "",
                    "is_secret": is_secret,
                    "description": description,
                }
        return JSONResponse({"error": f"Unknown config key: {key}"}, status_code=404)

    @app.post("/config/{key}")
    def _set_config_key(
        key: str,
        value: Optional[str] = Body(None, embed=True),
        persist: bool = Body(True, embed=True),
    ) -> dict[str, Any] | JSONResponse:
        """Set a configuration variable."""
        # Find the config variable
        config_info = None
        for env_key, config_attr, is_secret, description in CONFIG_VARS:
            if env_key == key:
                config_info = (env_key, config_attr, is_secret, description)
                break

        if config_info is None:
            return JSONResponse({"error": f"Unknown config key: {key}"}, status_code=404)

        env_key, config_attr, is_secret, description = config_info

        # Update runtime config
        _update_runtime_config(env_key, value)

        # Update .env file for persistence
        persisted = False
        if persist:
            persisted = _update_env_file(env_key, value)

        current_value = getattr(config, config_attr, None)
        return {
            "ok": True,
            "key": env_key,
            "value": _mask_secret(current_value, is_secret),
            "is_set": current_value is not None and current_value != "",
            "persisted": persisted,
        }

    @app.delete("/config/{key}")
    def _delete_config_key(key: str, persist: bool = True) -> dict[str, Any] | JSONResponse:
        """Remove a configuration variable."""
        # Find the config variable
        config_info = None
        for env_key, config_attr, is_secret, description in CONFIG_VARS:
            if env_key == key:
                config_info = (env_key, config_attr, is_secret, description)
                break

        if config_info is None:
            return JSONResponse({"error": f"Unknown config key: {key}"}, status_code=404)

        env_key, config_attr, _, _ = config_info

        # Clear runtime config
        _update_runtime_config(env_key, None)

        # Update .env file
        persisted = False
        if persist:
            persisted = _update_env_file(env_key, None)

        return {
            "ok": True,
            "key": env_key,
            "cleared": True,
            "persisted": persisted,
        }
