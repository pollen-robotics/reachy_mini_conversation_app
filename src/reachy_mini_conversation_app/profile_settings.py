"""Helpers to load per-profile settings such as voice enablement."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from reachy_mini_conversation_app.config import config

logger = logging.getLogger(__name__)

PROFILES_DIRECTORY = Path(__file__).parent / "profiles"
SETTINGS_FILENAME = "profile.json"


@dataclass(frozen=True)
class ProfileSettings:
    """Profile-level knobs that adjust runtime behavior."""

    enable_voice: bool = True
    enable_idle_behaviors: bool = True
    enable_flute_audio: bool = False


_PROFILE_SETTINGS: ProfileSettings | None = None


def get_profile_settings() -> ProfileSettings:
    """Return the cached profile settings, loading them once per run."""
    global _PROFILE_SETTINGS
    if _PROFILE_SETTINGS is None:
        _PROFILE_SETTINGS = _load_profile_settings()
    return _PROFILE_SETTINGS


def _load_profile_settings() -> ProfileSettings:
    profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"
    settings_path = PROFILES_DIRECTORY / profile / SETTINGS_FILENAME

    if not settings_path.exists():
        logger.debug(f"Profile '{profile}' has no {SETTINGS_FILENAME}; using defaults.")
        return ProfileSettings()

    try:
        raw = settings_path.read_text(encoding="utf-8").strip() or "{}"
    except OSError as exc:
        logger.warning(f"Failed to read {settings_path}: {exc}")
        return ProfileSettings()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(f"Invalid JSON in {settings_path}: {exc}")
        return ProfileSettings()

    enable_voice = bool(data.get("enable_voice", True))
    enable_idle_behaviors = bool(data.get("enable_idle_behaviors", True))
    enable_flute_audio = bool(data.get("enable_flute_audio", False))
    settings = ProfileSettings(
        enable_voice=enable_voice,
        enable_idle_behaviors=enable_idle_behaviors,
        enable_flute_audio=enable_flute_audio,
    )
    logger.info(f"Profile '{profile}' settings loaded: {settings}")
    return settings


def update_profile_settings(
    *,
    enable_voice: bool | None = None,
    enable_idle_behaviors: bool | None = None,
    enable_flute_audio: bool | None = None,
) -> ProfileSettings:
    """Update the cached profile settings and return the new values."""

    global _PROFILE_SETTINGS
    current = get_profile_settings()
    new_settings = ProfileSettings(
        enable_voice=current.enable_voice if enable_voice is None else bool(enable_voice),
        enable_idle_behaviors=(
            current.enable_idle_behaviors
            if enable_idle_behaviors is None
            else bool(enable_idle_behaviors)
        ),
        enable_flute_audio=(
            current.enable_flute_audio
            if enable_flute_audio is None
            else bool(enable_flute_audio)
        ),
    )
    _PROFILE_SETTINGS = new_settings
    logger.info(f"Profile settings updated: {new_settings}")
    return new_settings
