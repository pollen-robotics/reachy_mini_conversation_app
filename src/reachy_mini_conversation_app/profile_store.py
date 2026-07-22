"""Load and write declarative personality profiles."""

import os
import json
import logging
import tomllib
import threading
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Iterable

from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config


logger = logging.getLogger(__name__)

PROFILE_FILENAME = "profile.md"
PROFILE_SCHEMA_VERSION = 1
DEFAULT_PROFILE_NAME = "default"
_FRONT_MATTER_DELIMITER = "+++"
_PROFILE_METADATA_FIELDS = {"schema_version", "default_tools", "voice", "greeting", "hidden"}
_STORE_LOCK = threading.Lock()


class ProfileFormatError(ValueError):
    """Raised when a profile definition is malformed."""


@dataclass(frozen=True)
class ProfileDefinition:
    """Validated profile content and authored tool defaults."""

    instructions: str
    default_tools: tuple[str, ...]
    voice: str | None
    greeting: str | None
    hidden: bool


def canonical_profile_name(profile: str | None) -> str:
    """Return the stable profile name used by storage and runtime."""
    candidate = (profile or "").strip()
    if candidate in {"", DEFAULT_PROFILE_NAME}:
        return DEFAULT_PROFILE_NAME
    return candidate


def normalize_tool_names(tool_names: Iterable[str]) -> list[str]:
    """Normalize an ordered tool selection."""
    normalized: list[str] = []
    seen: set[str] = set()
    for tool_name in tool_names:
        candidate = tool_name.strip()
        if not candidate or candidate.startswith("#") or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def _optional_string(metadata: dict[str, object], field_name: str, source_path: Path) -> str | None:
    value = metadata.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProfileFormatError(f"Invalid {field_name!r} in {source_path}: expected a string.")
    return value.strip() or None


def _parse_profile_document(source_path: Path) -> ProfileDefinition:
    try:
        content = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProfileFormatError(f"Failed to read profile from {source_path}: {exc}") from exc

    lines = content.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIMITER:
        raise ProfileFormatError(f"Invalid profile {source_path}: expected TOML front matter starting with '+++'.")
    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == _FRONT_MATTER_DELIMITER
        )
    except StopIteration as exc:
        raise ProfileFormatError(
            f"Invalid profile {source_path}: missing closing '+++' front matter delimiter."
        ) from exc

    try:
        raw_metadata: object = tomllib.loads("\n".join(lines[1:closing_index]))
    except tomllib.TOMLDecodeError as exc:
        raise ProfileFormatError(f"Invalid TOML front matter in {source_path}: {exc}") from exc
    if not isinstance(raw_metadata, dict):
        raise ProfileFormatError(f"Invalid profile metadata in {source_path}: expected a TOML table.")
    metadata: dict[str, object] = raw_metadata
    unknown_fields = sorted(set(metadata) - _PROFILE_METADATA_FIELDS)
    if unknown_fields:
        raise ProfileFormatError(f"Unknown profile metadata in {source_path}: {', '.join(unknown_fields)}.")

    schema_version = metadata.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ProfileFormatError(f"Invalid schema_version in {source_path}: expected an integer.")
    if schema_version != PROFILE_SCHEMA_VERSION:
        raise ProfileFormatError(
            f"Unsupported profile schema in {source_path}: expected version {PROFILE_SCHEMA_VERSION}."
        )

    raw_tools = metadata.get("default_tools")
    if not isinstance(raw_tools, list) or not all(isinstance(tool_name, str) for tool_name in raw_tools):
        raise ProfileFormatError(f"Invalid default_tools in {source_path}: expected a list of strings.")
    hidden = metadata.get("hidden", False)
    if not isinstance(hidden, bool):
        raise ProfileFormatError(f"Invalid hidden flag in {source_path}: expected a boolean.")

    instructions = "\n".join(lines[closing_index + 1 :]).strip()
    if not instructions:
        raise ProfileFormatError(f"Profile {source_path} has an empty instruction body.")
    return ProfileDefinition(
        instructions=instructions,
        default_tools=tuple(normalize_tool_names(raw_tools)),
        voice=_optional_string(metadata, "voice", source_path),
        greeting=_optional_string(metadata, "greeting", source_path),
        hidden=hidden,
    )


def read_profile_from_directory(profile_name: str, profile_directory: Path) -> ProfileDefinition:
    """Read a profile document from its directory."""
    profile_path = profile_directory / PROFILE_FILENAME
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile {profile_name!r} has no {PROFILE_FILENAME} at {profile_directory}")
    return _parse_profile_document(profile_path)


def read_profile(profile: str | None) -> ProfileDefinition:
    """Read a profile selected through the runtime configuration."""
    profile_name = canonical_profile_name(profile)
    if profile_name == DEFAULT_PROFILE_NAME:
        return read_packaged_default_profile()
    return read_profile_from_directory(profile_name, config.resolve_profile_dir(profile_name))


def read_packaged_default_profile() -> ProfileDefinition:
    """Read the immutable bundled default profile."""
    return read_profile_from_directory(
        DEFAULT_PROFILE_NAME,
        DEFAULT_PROFILES_DIRECTORY / DEFAULT_PROFILE_NAME,
    )


def profile_directory_has_definition(profile_directory: Path) -> bool:
    """Return whether a directory contains a profile definition."""
    return (profile_directory / PROFILE_FILENAME).is_file()


def list_profile_names(profiles_root: Path) -> list[str]:
    """List profile directory names containing a supported definition."""
    if not profiles_root.is_dir():
        return []
    return sorted(
        profile_directory.name
        for profile_directory in profiles_root.iterdir()
        if profile_directory.is_dir() and profile_directory_has_definition(profile_directory)
    )


def write_profile(
    profile_name: str,
    profile_directory: Path,
    instructions: str,
    default_tools: Iterable[str],
    *,
    voice: str | None = None,
    greeting: str | None = None,
    hidden: bool = False,
    overwrite: bool = True,
) -> Path:
    """Atomically write one declarative profile document."""
    if not instructions.strip():
        raise ValueError(f"Profile {profile_name!r} must have non-empty instructions.")
    lines = [
        _FRONT_MATTER_DELIMITER,
        f"schema_version = {PROFILE_SCHEMA_VERSION}",
    ]
    if hidden:
        lines.append("hidden = true")
    if voice and voice.strip():
        lines.append(f"voice = {json.dumps(voice.strip(), ensure_ascii=False)}")
    if greeting and greeting.strip():
        lines.append(f"greeting = {json.dumps(greeting.strip(), ensure_ascii=False)}")
    lines.extend(
        ["default_tools = [", *[f"  {json.dumps(name)}," for name in normalize_tool_names(default_tools)], "]"]
    )
    lines.extend([_FRONT_MATTER_DELIMITER, "", instructions.strip(), ""])
    with _STORE_LOCK:
        profile_directory.mkdir(parents=True, exist_ok=True)
        profile_path = profile_directory / PROFILE_FILENAME
        if profile_path.exists() and not overwrite:
            raise FileExistsError(f"Profile {profile_name!r} already exists.")
        temporary_path = profile_path.with_name(f".{profile_path.name}.{os.getpid()}.tmp")
        try:
            temporary_path.write_text(
                "\n".join(lines),
                encoding="utf-8",
            )
            temporary_path.replace(profile_path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove temporary profile file %s: %s", temporary_path, exc)
        return profile_path
