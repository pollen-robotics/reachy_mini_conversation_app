"""Helpers for persisting UI-selected startup profile and voice settings."""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass


logger = logging.getLogger(__name__)

STARTUP_SETTINGS_FILENAME = "startup_settings.json"


@dataclass(frozen=True)
class StartupSettings:
    """Instance-local startup profile/voice settings selected from the UI."""

    profile: str | None = None
    voice: str | None = None
    prompt_language: str | None = None


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

    return StartupSettings(
        profile=_normalize_optional_text(payload.get("profile")),
        voice=_normalize_optional_text(payload.get("voice")),
        prompt_language=_normalize_optional_text(payload.get("prompt_language")),
    )


def write_startup_settings(
    instance_path: str | Path | None,
    *,
    profile: str | None,
    voice: str | None,
    prompt_language: str | None = None,
) -> None:
    """Persist startup settings in an instance-local JSON file."""
    settings_path = _startup_settings_path(instance_path)
    if settings_path is None:
        return

    settings = StartupSettings(
        profile=_normalize_optional_text(profile),
        voice=_normalize_optional_text(voice),
        prompt_language=_normalize_optional_text(prompt_language),
    )
    if settings.profile is None and settings.voice is None and settings.prompt_language is None:
        try:
            settings_path.unlink()
        except FileNotFoundError:
            return
        return

    payload: dict[str, str] = {}
    if settings.profile is not None:
        payload["profile"] = settings.profile
    if settings.voice is not None:
        payload["voice"] = settings.voice
    if settings.prompt_language is not None:
        payload["prompt_language"] = settings.prompt_language

    settings_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


def load_startup_settings_into_runtime(instance_path: str | Path | None) -> StartupSettings:
    """Load instance-local startup settings when no explicit profile override is set."""
    from reachy_mini_conversation_app.config import LOCKED_PROFILE, set_custom_profile, set_prompt_language

    if LOCKED_PROFILE is not None:
        return StartupSettings()

    settings_path = _startup_settings_path(instance_path)
    settings = read_startup_settings(instance_path)
    if settings_path is None or not settings_path.exists():
        if os.getenv("REACHY_MINI_CUSTOM_PROFILE"):
            if settings.prompt_language:
                set_prompt_language(settings.prompt_language)
            return StartupSettings(voice=settings.voice, prompt_language=settings.prompt_language)

    set_custom_profile(settings.profile)
    if settings.prompt_language:
        set_prompt_language(settings.prompt_language)
    return settings
