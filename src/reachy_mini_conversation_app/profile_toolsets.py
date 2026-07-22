"""Persist instance-local tool selections for personality profiles."""

import os
import json
import logging
import threading
from pathlib import Path
from contextlib import contextmanager
from dataclasses import field, dataclass
from collections.abc import Iterable, Iterator

from reachy_mini_conversation_app.profile_store import (
    read_profile,
    normalize_tool_names,
    canonical_profile_name,
)


logger = logging.getLogger(__name__)

PROFILE_TOOLSETS_FILENAME = "profile_toolsets.json"
PROFILE_TOOLSETS_VERSION = 1
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")
_STORE_LOCK = threading.RLock()


@dataclass(frozen=True)
class ProfileToolsets:
    """Instance-local tool overrides keyed by canonical profile name."""

    profiles: dict[str, list[str]] = field(default_factory=dict)


@contextmanager
def profile_toolsets_transaction() -> Iterator[None]:
    """Serialize related profile and toolset persistence operations."""
    with _STORE_LOCK:
        yield


def get_profile_toolsets_path(instance_path: str | Path | None) -> Path:
    """Return the profile-toolset settings path for the current mode."""
    if instance_path is not None:
        return Path(instance_path) / PROFILE_TOOLSETS_FILENAME
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / PROFILE_TOOLSETS_FILENAME


def read_profile_toolsets(instance_path: str | Path | None) -> ProfileToolsets:
    """Read instance-local profile tool overrides."""
    with _STORE_LOCK:
        settings_path = get_profile_toolsets_path(instance_path)
        if not settings_path.exists():
            return ProfileToolsets()

        try:
            payload: object = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to read profile toolsets from {settings_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid profile toolsets payload in {settings_path}: expected a JSON object.")
        version = payload.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version != PROFILE_TOOLSETS_VERSION:
            raise RuntimeError(
                f"Unsupported profile toolsets version in {settings_path}: expected {PROFILE_TOOLSETS_VERSION}."
            )
        raw_profiles = payload.get("profiles", {})
        if not isinstance(raw_profiles, dict):
            raise RuntimeError(f"Invalid profile toolsets payload in {settings_path}: 'profiles' must be an object.")

        profiles: dict[str, list[str]] = {}
        for raw_profile, raw_tool_names in raw_profiles.items():
            if (
                not isinstance(raw_profile, str)
                or not isinstance(raw_tool_names, list)
                or not all(isinstance(tool_name, str) for tool_name in raw_tool_names)
            ):
                raise RuntimeError(
                    f"Invalid profile toolsets entry in {settings_path}: profile names and tool lists must be strings."
                )
            profiles[canonical_profile_name(raw_profile)] = normalize_tool_names(raw_tool_names)
        return ProfileToolsets(profiles=profiles)


def write_profile_toolsets(
    instance_path: str | Path | None,
    toolsets: ProfileToolsets,
) -> Path:
    """Persist instance-local profile tool overrides."""
    with _STORE_LOCK:
        settings_path = get_profile_toolsets_path(instance_path)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": PROFILE_TOOLSETS_VERSION,
            "profiles": {profile: list(tool_names) for profile, tool_names in sorted(toolsets.profiles.items())},
        }
        temporary_path = settings_path.with_name(f".{settings_path.name}.{os.getpid()}.tmp")
        try:
            temporary_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
            temporary_path.replace(settings_path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove temporary profile toolsets file %s: %s", temporary_path, exc)
        return settings_path


def read_profile_tool_override(
    profile: str | None,
    instance_path: str | Path | None,
) -> list[str] | None:
    """Return a profile override, distinguishing an empty override from none."""
    tool_names = read_profile_toolsets(instance_path).profiles.get(canonical_profile_name(profile))
    return list(tool_names) if tool_names is not None else None


def read_profile_default_tool_names(profile: str | None) -> list[str]:
    """Read a profile's authored tool defaults without applying an override."""
    return list(read_profile(profile).default_tools)


def read_profile_tool_names(
    profile: str | None,
    instance_path: str | Path | None,
) -> list[str]:
    """Read the effective enabled tools for a profile."""
    override = read_profile_tool_override(profile, instance_path)
    if override is not None:
        return override
    return read_profile_default_tool_names(profile)


def write_profile_tool_override(
    profile: str | None,
    tool_names: Iterable[str],
    instance_path: str | Path | None,
) -> Path:
    """Write a complete enabled-tool override for one profile."""
    with _STORE_LOCK:
        toolsets = read_profile_toolsets(instance_path)
        profiles = dict(toolsets.profiles)
        profiles[canonical_profile_name(profile)] = normalize_tool_names(tool_names)
        return write_profile_toolsets(instance_path, ProfileToolsets(profiles=profiles))


def clear_profile_tool_override(
    profile: str | None,
    instance_path: str | Path | None,
) -> bool:
    """Clear one profile override and restore its on-disk defaults."""
    with _STORE_LOCK:
        toolsets = read_profile_toolsets(instance_path)
        profile_name = canonical_profile_name(profile)
        if profile_name not in toolsets.profiles:
            return False

        profiles = dict(toolsets.profiles)
        del profiles[profile_name]
        settings_path = get_profile_toolsets_path(instance_path)
        if profiles:
            write_profile_toolsets(instance_path, ProfileToolsets(profiles=profiles))
        elif settings_path.exists():
            settings_path.unlink()
        return True


def enable_profile_tools(
    profile: str | None,
    tool_names: Iterable[str],
    instance_path: str | Path | None,
) -> list[str]:
    """Enable additional tools for one profile and return newly enabled IDs."""
    with profile_toolsets_transaction():
        current = read_profile_tool_names(profile, instance_path)
        additions = [tool_name for tool_name in normalize_tool_names(tool_names) if tool_name not in current]
        if additions:
            write_profile_tool_override(profile, [*current, *additions], instance_path)
        return additions


def disable_profile_tools_by_prefix(
    profile_names: Iterable[str],
    prefix: str,
    instance_path: str | Path | None,
) -> list[tuple[str, list[str]]]:
    """Disable matching tools per profile while preserving explicit tombstones."""
    with _STORE_LOCK:
        toolsets = read_profile_toolsets(instance_path)
        profiles = dict(toolsets.profiles)
        disabled: list[tuple[str, list[str]]] = []
        seen_profiles: set[str] = set()
        for profile in profile_names:
            profile_name = canonical_profile_name(profile)
            if profile_name in seen_profiles:
                continue
            seen_profiles.add(profile_name)
            current = profiles.get(profile_name)
            if current is None:
                current = read_profile_default_tool_names(profile_name)
            removed = [tool_name for tool_name in current if tool_name.startswith(prefix)]
            if not removed:
                continue
            profiles[profile_name] = [tool_name for tool_name in current if not tool_name.startswith(prefix)]
            disabled.append((profile_name, removed))

        if disabled:
            write_profile_toolsets(instance_path, ProfileToolsets(profiles=profiles))
        return disabled
