"""Helpers for persisting the UI-selected startup personality."""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass

from reachy_mini_conversation_app.profile_voices import write_profile_voice_override


logger = logging.getLogger(__name__)

STARTUP_SETTINGS_FILENAME = "startup_settings.json"


@dataclass(frozen=True)
class StartupSettings:
    """Instance-local startup profile selected from the UI."""

    profile: str | None = None


def _normalize_optional_text(value: object) -> str | None:
    """Return a stripped string or None for empty/non-string values."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _startup_settings_path(instance_path: str | Path | None) -> Path | None:
    """Return the startup settings JSON path for an instance directory."""
    if instance_path is None:
        return None
    return Path(instance_path) / STARTUP_SETTINGS_FILENAME


def read_startup_settings(instance_path: str | Path | None) -> StartupSettings:
    """Read startup settings from an instance-local JSON file."""
    settings_path = _startup_settings_path(instance_path)
    if settings_path is None or not settings_path.exists():
        return StartupSettings()

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read startup settings from %s: %s", settings_path, exc)
        return StartupSettings()

    if not isinstance(payload, dict):
        logger.warning("Ignoring invalid startup settings payload from %s: %r", settings_path, payload)
        return StartupSettings()

    return StartupSettings(profile=_normalize_optional_text(payload.get("profile")))


def write_startup_settings(instance_path: str | Path | None, *, profile: str | None) -> None:
    """Persist startup settings in an instance-local JSON file."""
    settings_path = _startup_settings_path(instance_path)
    if settings_path is None:
        return

    settings = StartupSettings(profile=_normalize_optional_text(profile))
    if settings.profile is None:
        try:
            settings_path.unlink()
        except FileNotFoundError:
            return
        return

    payload = {"profile": settings.profile}
    settings_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


def _migrate_startup_voice(instance_path: str | Path | None, profile: str | None) -> None:
    """Move a voice saved by an older version onto the personality it was picked for.

    That voice was pinned to startup rather than to a personality, so swapping
    personalities and coming back spoke with the wrong one. It now belongs in
    :mod:`profile_voices`, where every personality carries its own.
    """
    settings_path = _startup_settings_path(instance_path)
    if settings_path is None or not settings_path.exists():
        return

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s while migrating the startup voice: %s", settings_path, exc)
        return
    voice = _normalize_optional_text(payload.get("voice")) if isinstance(payload, dict) else None
    if voice is None:
        return

    try:
        write_profile_voice_override(profile, voice, instance_path)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Failed to give personality %r its saved voice %r: %s", profile, voice, exc)
        return
    write_startup_settings(instance_path, profile=profile)
    logger.info("Moved the saved startup voice %r onto personality %r", voice, profile)


def load_startup_settings_into_runtime(instance_path: str | Path | None) -> StartupSettings:
    """Load instance-local startup settings when no explicit profile override is set."""
    from reachy_mini_conversation_app.config import LOCKED_PROFILE, set_custom_profile

    if LOCKED_PROFILE is not None:
        return StartupSettings()

    settings_path = _startup_settings_path(instance_path)
    settings = read_startup_settings(instance_path)
    _migrate_startup_voice(instance_path, settings.profile)
    if settings_path is None or not settings_path.exists():
        if os.getenv("REACHY_MINI_CUSTOM_PROFILE"):
            return StartupSettings()

    set_custom_profile(settings.profile)
    return settings
