"""Personality profile data layer."""

import re
import shutil
import logging
from typing import Literal, TypedDict
from pathlib import Path
from collections.abc import Iterable

from reachy_mini_conversation_app.config import (
    USER_PERSONALITIES_DIRNAME,
    config,
    get_default_voice,
    list_tool_module_names,
)
from reachy_mini_conversation_app.tool_spaces import read_installed_tool_spaces
from reachy_mini_conversation_app.profile_store import (
    DEFAULT_PROFILE_NAME,
    ProfileFormatError,
    write_profile,
    list_profile_names,
    read_profile_from_directory,
    read_packaged_default_profile,
)
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_toolsets,
    write_profile_toolsets,
    get_profile_toolsets_path,
    clear_profile_tool_override,
    profile_toolsets_transaction,
)
from reachy_mini_conversation_app.tools.tool_constants import SystemTool


logger = logging.getLogger(__name__)


class AvailableTool(TypedDict):
    """Tool metadata used by personality configuration surfaces."""

    id: str
    kind: Literal["shared", "external", "tool_space"]
    source: str
    description: str


def _visible_profile_names(profiles_root: Path, prefix: str = "") -> list[str]:
    visible: list[str] = []
    for profile_name in list_profile_names(profiles_root):
        try:
            profile = read_profile_from_directory(profile_name, profiles_root / profile_name)
        except (FileNotFoundError, ProfileFormatError) as exc:
            logger.warning("Skipping invalid profile %r: %s", profile_name, exc)
            continue
        if not profile.hidden:
            visible.append(f"{prefix}{profile_name}")
    return visible


def list_personalities() -> list[str]:
    """List available visible personality profile names."""
    names = [DEFAULT_PROFILE_NAME]
    names.extend(
        profile_name
        for profile_name in _visible_profile_names(config.PROFILES_DIRECTORY)
        if profile_name != DEFAULT_PROFILE_NAME
    )
    user_root = config.user_personalities_root()
    if user_root != config.PROFILES_DIRECTORY:
        names.extend(_visible_profile_names(user_root, f"{USER_PERSONALITIES_DIRNAME}/"))
    return names


def available_tool_catalog() -> list[AvailableTool]:
    """List configurable tools and their source."""
    catalog: dict[str, AvailableTool] = {}
    excluded_modules = {"__init__", "core_tools", "background_tool_manager", "tool_constants"}
    excluded_modules.update(tool.value for tool in SystemTool)
    for tool_name in list_tool_module_names(Path(__file__).parent / "tools"):
        if tool_name in excluded_modules:
            continue
        catalog[tool_name] = {
            "id": tool_name,
            "kind": "shared",
            "source": "Built-in",
            "description": "",
        }

    for tool_name in list_tool_module_names(config.TOOLS_DIRECTORY):
        catalog[tool_name] = {
            "id": tool_name,
            "kind": "external",
            "source": "External",
            "description": "",
        }

    try:
        for space in read_installed_tool_spaces(config.INSTANCE_PATH).spaces:
            for tool in space.tools:
                catalog[tool.local_name] = {
                    "id": tool.local_name,
                    "kind": "tool_space",
                    "source": space.slug,
                    "description": tool.description,
                }
    except (RuntimeError, ValueError) as exc:
        logger.warning("Failed to list installed Tool Space tools: %s", exc)
    return [catalog[tool_id] for tool_id in sorted(catalog)]


def delete_personality(name: str) -> bool:
    """Delete a user-created personality without touching bundled profiles."""
    target = config.resolve_profile_dir(name).resolve()
    user_root = config.user_personalities_root().resolve()
    if user_root not in target.parents:
        return False
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    try:
        clear_profile_tool_override(name, config.INSTANCE_PATH)
    except (OSError, RuntimeError) as exc:
        logger.warning("Deleted personality %r but could not remove its tool override: %s", name, exc)
    return True


def save_user_personality(
    name: str,
    instructions: str,
    voice: str | None = None,
    greeting: str | None = None,
    *,
    overwrite: bool = False,
    default_tools: Iterable[str] | None = None,
) -> str:
    """Save a custom personality with optional authored tool defaults."""
    profile_name = name.strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]+", profile_name) is None:
        raise ValueError("Profile names may contain only letters, numbers, dashes, and underscores.")
    if not instructions.strip():
        raise ValueError(f"Profile {profile_name!r} must have non-empty instructions.")

    profile_directory = config.user_personalities_root() / profile_name
    selection = f"{USER_PERSONALITIES_DIRNAME}/{profile_name}"
    selected_voice = voice or get_default_voice()
    authored_tools = tuple(default_tools) if default_tools is not None else None
    with profile_toolsets_transaction():
        if profile_directory.exists() and not overwrite:
            raise FileExistsError(f"Personality {profile_name!r} already exists.")
        try:
            previous_profile = read_profile_from_directory(profile_name, profile_directory)
            profile_tools = previous_profile.default_tools
            hidden = previous_profile.hidden
        except FileNotFoundError:
            profile_tools = read_packaged_default_profile().default_tools
            hidden = False
        if authored_tools is not None:
            profile_tools = authored_tools

        toolsets_path = get_profile_toolsets_path(config.INSTANCE_PATH)
        toolsets_existed = toolsets_path.is_file()
        previous_toolsets = read_profile_toolsets(config.INSTANCE_PATH) if authored_tools is not None else None

        if authored_tools is not None:
            clear_profile_tool_override(selection, config.INSTANCE_PATH)

        try:
            write_profile(
                profile_name,
                profile_directory,
                instructions,
                profile_tools,
                voice=selected_voice,
                greeting=greeting,
                hidden=hidden,
                overwrite=overwrite,
            )
        except OSError:
            try:
                if toolsets_existed and previous_toolsets is not None:
                    write_profile_toolsets(config.INSTANCE_PATH, previous_toolsets)
                elif authored_tools is not None:
                    toolsets_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to restore profile toolsets after saving profile %r: %s", profile_name, exc)
            raise
    return selection
