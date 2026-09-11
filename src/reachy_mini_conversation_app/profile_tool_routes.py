"""JSON-RPC methods for per-personality tool access."""

import asyncio
import logging
from typing import Any
from pathlib import Path
from collections.abc import Callable

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import LOCKED_PROFILE, config
from reachy_mini_conversation_app.personality import AvailableTool, list_personalities, available_tool_catalog
from reachy_mini_conversation_app.profile_store import (
    DEFAULT_PROFILE_NAME,
    normalize_tool_names,
    canonical_profile_name,
    profile_directory_has_definition,
)
from reachy_mini_conversation_app.tool_settings import (
    RestartCallback,
    apply_tool_change,
    raise_tool_settings_error,
)
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_tool_override,
    clear_profile_tool_override,
    write_profile_tool_override,
    read_profile_default_tool_names,
)


logger = logging.getLogger(__name__)


def _known_profile_names() -> list[str]:
    profile_names = [canonical_profile_name(profile) for profile in list_personalities()]
    active_profile = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
    if active_profile not in profile_names and profile_directory_has_definition(
        config.resolve_profile_dir(active_profile)
    ):
        profile_names.append(active_profile)
    return sorted(set(profile_names), key=lambda profile: (profile != DEFAULT_PROFILE_NAME, profile))


def _validated_profile(profile: str | None, known_profile_names: list[str]) -> str:
    profile_name = canonical_profile_name(profile)
    if profile_name not in known_profile_names:
        raise ValueError(f"Unknown personality: {profile_name}")
    return profile_name


def _profile_param(params: dict[str, Any], *, required: bool = False) -> str | None:
    profile = params.get("profile")
    if profile is None and not required:
        return None
    if not isinstance(profile, str) or not profile.strip():
        raise_tool_settings_error("unknown_profile", "Choose an available personality.")
    return profile


def _profile_tool_payload(
    profile_name: str,
    known_profile_names: list[str],
    available_tools: list[AvailableTool],
    enabled_tools: list[str],
    *,
    overridden: bool,
) -> dict[str, object]:
    active_profile = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
    available_ids = {tool["id"] for tool in available_tools}
    return {
        "profile": profile_name,
        "is_active": profile_name == active_profile,
        "overridden": overridden,
        "editable": LOCKED_PROFILE is None,
        "profiles": [
            {"id": known_profile, "active": known_profile == active_profile} for known_profile in known_profile_names
        ],
        "enabled_tools": enabled_tools,
        "available_tools": available_tools,
        "unavailable_enabled_tools": [tool_id for tool_id in enabled_tools if tool_id not in available_ids],
    }


def register_profile_tool_methods(
    rpc: JsonRpcServer,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    restart_conversation: RestartCallback,
    *,
    instance_path: str | Path | None,
) -> None:
    """Register per-personality tool-access methods."""

    async def _get_profile_tools(params: dict[str, Any]) -> dict[str, object]:
        requested_profile = _profile_param(params)
        try:
            known_profile_names = await asyncio.to_thread(_known_profile_names)
            profile_name = _validated_profile(
                requested_profile or config.REACHY_MINI_CUSTOM_PROFILE,
                known_profile_names,
            )
            override = await asyncio.to_thread(read_profile_tool_override, profile_name, instance_path)
            enabled_tools = (
                list(override)
                if override is not None
                else await asyncio.to_thread(read_profile_default_tool_names, profile_name)
            )
            available_tools = await asyncio.to_thread(available_tool_catalog)
        except ValueError as exc:
            raise_tool_settings_error("unknown_profile", str(exc))
        except Exception as exc:
            logger.exception("Failed to read profile tools for %r", requested_profile)
            raise_tool_settings_error("profile_tools_unavailable", str(exc))
        return _profile_tool_payload(
            profile_name,
            known_profile_names,
            available_tools,
            enabled_tools,
            overridden=override is not None,
        )

    async def _save_profile_tools(params: dict[str, Any]) -> dict[str, object]:
        if LOCKED_PROFILE is not None:
            raise_tool_settings_error("profile_locked", "Personality tool editing is locked.")
        requested_profile = _profile_param(params, required=True)
        requested_tools = params.get("enabled_tools")
        if not isinstance(requested_tools, list) or not all(
            isinstance(tool_name, str) for tool_name in requested_tools
        ):
            raise_tool_settings_error("invalid_tool_selection", "Enabled tools must be a list of tool names.")
        enabled_tools = normalize_tool_names(requested_tools)
        try:
            known_profile_names = await asyncio.to_thread(_known_profile_names)
            profile_name = _validated_profile(requested_profile, known_profile_names)
            current_override = await asyncio.to_thread(read_profile_tool_override, profile_name, instance_path)
            default_tools = await asyncio.to_thread(read_profile_default_tool_names, profile_name)
            current_tools = list(current_override) if current_override is not None else list(default_tools)
            available_tools = await asyncio.to_thread(available_tool_catalog)
            available_ids = {tool["id"] for tool in available_tools}
            unknown_tools = sorted(set(enabled_tools) - available_ids - set(current_tools))
        except ValueError as exc:
            raise_tool_settings_error("unknown_profile", str(exc))
        except Exception as exc:
            logger.exception("Failed to save profile tools for %r", requested_profile)
            raise_tool_settings_error("profile_tools_save_failed", str(exc))

        if unknown_tools:
            raise_tool_settings_error(
                "invalid_tool_selection",
                f"Unknown tools for '{profile_name}': {', '.join(unknown_tools)}",
            )
        try:
            await asyncio.to_thread(write_profile_tool_override, profile_name, enabled_tools, instance_path)
            is_active = profile_name == canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
            apply_detail = (
                await asyncio.to_thread(
                    apply_tool_change,
                    instance_path,
                    get_loop,
                    restart_conversation,
                    "profile_tools_changed",
                )
                if is_active
                else "The tools will apply next time this personality is selected."
            )
        except Exception as exc:
            logger.exception("Failed to save profile tools for %r", requested_profile)
            raise_tool_settings_error("profile_tools_save_failed", str(exc))

        response = _profile_tool_payload(
            profile_name,
            known_profile_names,
            available_tools,
            enabled_tools,
            overridden=True,
        )
        response["message"] = f"Saved tools for {profile_name}. {apply_detail}"
        return response

    async def _reset_profile_tools(params: dict[str, Any]) -> dict[str, object]:
        if LOCKED_PROFILE is not None:
            raise_tool_settings_error("profile_locked", "Personality tool editing is locked.")
        requested_profile = _profile_param(params, required=True)
        try:
            known_profile_names = await asyncio.to_thread(_known_profile_names)
            profile_name = _validated_profile(requested_profile, known_profile_names)
            default_tools = await asyncio.to_thread(read_profile_default_tool_names, profile_name)
            available_tools = await asyncio.to_thread(available_tool_catalog)
            cleared = await asyncio.to_thread(clear_profile_tool_override, profile_name, instance_path)
            is_active = profile_name == canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
            apply_detail = (
                await asyncio.to_thread(
                    apply_tool_change,
                    instance_path,
                    get_loop,
                    restart_conversation,
                    "profile_tools_changed",
                )
                if cleared and is_active
                else "The tools will apply next time this personality is selected."
            )
        except ValueError as exc:
            raise_tool_settings_error("unknown_profile", str(exc))
        except Exception as exc:
            logger.exception("Failed to reset profile tools for %r", requested_profile)
            raise_tool_settings_error("profile_tools_reset_failed", str(exc))

        response = _profile_tool_payload(
            profile_name,
            known_profile_names,
            available_tools,
            default_tools,
            overridden=False,
        )
        response["message"] = f"Restored profile defaults for {profile_name}. {apply_detail}"
        return response

    rpc.register("profile_tools.get", _get_profile_tools)
    rpc.register("profile_tools.save", _save_profile_tools)
    rpc.register("profile_tools.reset", _reset_profile_tools)
