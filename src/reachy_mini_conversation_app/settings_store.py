"""Instance-local settings the UI and remote clients write.

The app used to persist these into the instance `.env`. That file is still read
at startup, so hand-written configuration and deployment environment variables
keep working, but they are only the starting value: what a user changes through
the UI lands here and wins. Writing them back as env assignments could not
represent a boolean, dropped empty values, shared a namespace with the process
environment, and produced #532 (an unquoted value with a newline wrote an
unrelated key).
"""

import json
import logging
import threading
from typing import Any
from pathlib import Path
from dataclasses import fields, replace, dataclass

from reachy_mini_conversation_app.config import (
    config,
    normalize_hf_connection_mode,
    normalize_transcription_language,
)


logger = logging.getLogger(__name__)

SETTINGS_FILENAME = "settings.json"
SETTINGS_VERSION = 1

_STORE_LOCK = threading.Lock()


@dataclass(frozen=True)
class AppSettings:
    """Settings chosen through the UI. ``None`` means "not set here"."""

    hf_connection_mode: str | None = None
    hf_ws_url: str | None = None
    language: str | None = None
    memory_enabled: bool | None = None
    camera_enabled: bool | None = None


def settings_path_for_instance(instance_path: str | Path | None) -> Path | None:
    """Return the settings file for an instance directory, if there is one."""
    if instance_path is None:
        return None
    return Path(instance_path) / SETTINGS_FILENAME


def _text(payload: dict[str, Any], name: str) -> str | None:
    """Return a non-empty stored string, or None when absent or unusable."""
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _flag(payload: dict[str, Any], name: str) -> bool | None:
    """Return a stored boolean, or None when absent or unusable."""
    value = payload.get(name)
    return value if isinstance(value, bool) else None


def _read_unlocked(path: Path | None) -> AppSettings:
    if path is None or not path.exists():
        return AppSettings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable settings file %s: %s", path, exc)
        return AppSettings()
    if not isinstance(payload, dict):
        logger.warning("Ignoring malformed settings file %s", path)
        return AppSettings()

    return AppSettings(
        hf_connection_mode=_text(payload, "hf_connection_mode"),
        hf_ws_url=_text(payload, "hf_ws_url"),
        language=_text(payload, "language"),
        memory_enabled=_flag(payload, "memory_enabled"),
        camera_enabled=_flag(payload, "camera_enabled"),
    )


def read_settings(instance_path: str | Path | None) -> AppSettings:
    """Read the instance settings, ignoring anything unreadable or malformed."""
    with _STORE_LOCK:
        return _read_unlocked(settings_path_for_instance(instance_path))


def update_settings(instance_path: str | Path | None, changes: AppSettings) -> AppSettings:
    """Merge non-None fields of ``changes`` into the stored settings."""
    path = settings_path_for_instance(instance_path)
    if path is None:
        return changes

    with _STORE_LOCK:
        current = _read_unlocked(path)
        updates = {
            field.name: getattr(changes, field.name)
            for field in fields(AppSettings)
            if getattr(changes, field.name) is not None
        }
        merged = replace(current, **updates)

        payload: dict[str, Any] = {"version": SETTINGS_VERSION}
        payload.update(
            {
                field.name: getattr(merged, field.name)
                for field in fields(AppSettings)
                if getattr(merged, field.name) is not None
            }
        )
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
        temporary.replace(path)
        logger.info("Persisted %s to %s", ", ".join(sorted(updates)), path)
        return merged


def apply_settings_to_runtime(settings: AppSettings) -> None:
    """Apply stored settings over the environment-derived runtime config."""
    if settings.hf_connection_mode is not None:
        mode = normalize_hf_connection_mode(settings.hf_connection_mode)
        if mode is not None:
            config.HF_REALTIME_CONNECTION_MODE = mode
    if settings.hf_ws_url is not None:
        config.HF_REALTIME_WS_URL = settings.hf_ws_url
    if settings.language is not None:
        config.REALTIME_TRANSCRIPTION_LANGUAGE = normalize_transcription_language(settings.language)
    if settings.memory_enabled is not None:
        config.MEMORY_ENABLED = settings.memory_enabled
    if settings.camera_enabled is not None:
        config.CAMERA_ENABLED = settings.camera_enabled


def load_settings_into_runtime(instance_path: str | Path | None) -> AppSettings:
    """Read the instance settings and apply them to the runtime config."""
    settings = read_settings(instance_path)
    apply_settings_to_runtime(settings)
    return settings
