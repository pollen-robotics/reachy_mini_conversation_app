"""JSON-RPC methods for managing installed Hugging Face Space tools."""

import asyncio
import logging
from typing import Any
from pathlib import Path
from collections.abc import Callable

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import LOCKED_PROFILE, config
from reachy_mini_conversation_app.tool_spaces import (
    ToolSpaceNotInstalledError,
    InstalledToolSpacesManifest,
    ToolSpaceAliasConflictError,
    ToolSpaceProfileUpdateError,
    remove_tool_space,
    install_tool_space,
    read_installed_tool_spaces,
)
from reachy_mini_conversation_app.profile_store import canonical_profile_name
from reachy_mini_conversation_app.tool_settings import (
    RestartCallback,
    apply_tool_change,
    raise_tool_settings_error,
)
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_names


logger = logging.getLogger(__name__)


def _space_settings_payload(manifest: InstalledToolSpacesManifest) -> dict[str, object]:
    return {
        "spaces": [
            {
                "slug": space.slug,
                "private": space.private,
                "tool_count": len(space.tools),
            }
            for space in sorted(manifest.spaces, key=lambda installed_space: installed_space.slug)
        ],
        "editable": LOCKED_PROFILE is None,
    }


def _error_detail(error: BaseException) -> str:
    return str(error).strip() or type(error).__name__


def _required_slug(params: dict[str, Any]) -> str:
    slug = params.get("slug")
    if not isinstance(slug, str):
        raise_tool_settings_error("invalid_tool_space_slug", "Enter a Space in owner/name format.")
    return slug


def register_tool_space_methods(
    rpc: JsonRpcServer,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    restart_conversation: RestartCallback,
    *,
    instance_path: str | Path | None,
) -> None:
    """Register Hugging Face Space tool-management methods."""

    async def _list_tool_spaces(_params: dict[str, Any]) -> dict[str, object]:
        try:
            manifest = await asyncio.to_thread(read_installed_tool_spaces, instance_path)
        except Exception as exc:
            logger.exception("Failed to list installed tool Spaces")
            raise_tool_settings_error("tool_spaces_unavailable", _error_detail(exc))
        return _space_settings_payload(manifest)

    async def _add_tool_space(params: dict[str, Any]) -> dict[str, object]:
        if LOCKED_PROFILE is not None:
            raise_tool_settings_error("profile_locked", "Tool Space editing is locked.")
        slug = _required_slug(params)
        try:
            result = await asyncio.to_thread(
                install_tool_space,
                slug,
                instance_path,
                install_only=True,
            )
        except ToolSpaceAliasConflictError as exc:
            logger.warning("Tool Space alias conflict: %s", exc)
            raise_tool_settings_error("tool_space_alias_conflict", _error_detail(exc))
        except ValueError as exc:
            logger.warning("Invalid tool Space slug %r: %s", slug, exc)
            raise_tool_settings_error("invalid_tool_space_slug", _error_detail(exc))
        except RuntimeError as exc:
            logger.error("Failed to install tool Space %r: %s", slug, exc)
            raise_tool_settings_error("tool_space_install_failed", _error_detail(exc))
        except Exception as exc:
            logger.exception("Unexpected failure installing tool Space %r", slug)
            raise_tool_settings_error("tool_space_install_failed", _error_detail(exc))

        active_profile = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
        try:
            active_tools = await asyncio.to_thread(read_profile_tool_names, active_profile, instance_path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning("Installed Tool Space but could not inspect active profile %r: %s", active_profile, exc)
            active_tools = []
        apply_detail = (
            await asyncio.to_thread(
                apply_tool_change,
                instance_path,
                get_loop,
                restart_conversation,
                "tool_spaces_changed",
            )
            if any(tool_name.startswith(f"{result.resolved_space.alias}__") for tool_name in active_tools)
            else "The Space is ready to assign to personalities."
        )

        action = "Refreshed" if result.refreshed else "Installed"
        tool_count = len(result.resolved_space.tools)
        tool_label = "tool" if tool_count == 1 else "tools"
        message = (
            f"{action} {result.resolved_space.slug} with {tool_count} {tool_label}. "
            "Choose which personalities can use them in Tool access. "
            f"{apply_detail}"
        )
        return {
            **_space_settings_payload(result.manifest),
            "message": message,
        }

    async def _remove_tool_space(params: dict[str, Any]) -> dict[str, object]:
        if LOCKED_PROFILE is not None:
            raise_tool_settings_error("profile_locked", "Tool Space editing is locked.")
        slug = _required_slug(params)
        try:
            result = await asyncio.to_thread(remove_tool_space, slug, instance_path)
            disabled_profiles = result.disabled_profiles
        except ToolSpaceNotInstalledError as exc:
            logger.warning("Cannot remove tool Space %r: %s", slug, exc)
            raise_tool_settings_error("tool_space_not_installed", _error_detail(exc))
        except ToolSpaceProfileUpdateError as exc:
            logger.error("Failed to disable removed tool Space: %s", exc)
            raise_tool_settings_error("profile_disable_failed", _error_detail(exc))
        except ValueError as exc:
            logger.warning("Invalid tool Space slug %r: %s", slug, exc)
            raise_tool_settings_error("invalid_tool_space_slug", _error_detail(exc))
        except RuntimeError as exc:
            logger.error("Failed to remove tool Space %r: %s", slug, exc)
            raise_tool_settings_error("tool_space_remove_failed", _error_detail(exc))
        except Exception as exc:
            logger.exception("Unexpected failure removing tool Space %r", slug)
            raise_tool_settings_error("tool_space_remove_failed", _error_detail(exc))

        active_profile = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
        apply_detail = (
            await asyncio.to_thread(
                apply_tool_change,
                instance_path,
                get_loop,
                restart_conversation,
                "tool_spaces_changed",
            )
            if any(profile_name == active_profile for profile_name, _ in disabled_profiles)
            else "No active conversation restart is needed."
        )

        disabled_tool_count = sum(len(tool_ids) for _, tool_ids in disabled_profiles)
        tool_label = "tool" if disabled_tool_count == 1 else "tools"
        message = (
            f"Removed {result.removed_space.slug}. Disabled {disabled_tool_count} {tool_label} across personalities. "
            f"{apply_detail}"
        )
        return {
            **_space_settings_payload(result.manifest),
            "message": message,
        }

    rpc.register("tool_spaces.list", _list_tool_spaces)
    rpc.register("tool_spaces.add", _add_tool_space)
    rpc.register("tool_spaces.remove", _remove_tool_space)
