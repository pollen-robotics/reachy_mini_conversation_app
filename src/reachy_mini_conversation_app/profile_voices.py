"""Persist instance-local voice selections for personality profiles.

A profile document authors a voice; this store holds the per-instance override on
top of it, the same way :mod:`profile_toolsets` overrides authored tool defaults.
Packaged profiles are read-only, so the override is the only way to give one of
them a voice of its own — and it survives an app update, which editing the
packaged ``profile.md`` would not.
"""

import os
import json
import logging
import threading
from pathlib import Path
from dataclasses import field, dataclass

from reachy_mini_conversation_app.profile_store import read_profile, canonical_profile_name


logger = logging.getLogger(__name__)

PROFILE_VOICES_FILENAME = "profile_voices.json"
PROFILE_VOICES_VERSION = 1
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")
_STORE_LOCK = threading.RLock()


@dataclass(frozen=True)
class ProfileVoices:
    """Instance-local voice overrides keyed by canonical profile name."""

    profiles: dict[str, str] = field(default_factory=dict)


def get_profile_voices_path(instance_path: str | Path | None) -> Path:
    """Return the profile-voice settings path for the current mode."""
    if instance_path is not None:
        return Path(instance_path) / PROFILE_VOICES_FILENAME
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / PROFILE_VOICES_FILENAME


def read_profile_voices(instance_path: str | Path | None) -> ProfileVoices:
    """Read instance-local profile voice overrides."""
    with _STORE_LOCK:
        settings_path = get_profile_voices_path(instance_path)
        if not settings_path.exists():
            return ProfileVoices()

        try:
            payload: object = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to read profile voices from {settings_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid profile voices payload in {settings_path}: expected a JSON object.")
        version = payload.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version != PROFILE_VOICES_VERSION:
            raise RuntimeError(
                f"Unsupported profile voices version in {settings_path}: expected {PROFILE_VOICES_VERSION}."
            )
        raw_profiles = payload.get("profiles", {})
        if not isinstance(raw_profiles, dict):
            raise RuntimeError(f"Invalid profile voices payload in {settings_path}: 'profiles' must be an object.")

        profiles: dict[str, str] = {}
        for raw_profile, raw_voice in raw_profiles.items():
            if not isinstance(raw_profile, str) or not isinstance(raw_voice, str):
                raise RuntimeError(
                    f"Invalid profile voices entry in {settings_path}: profile names and voices must be strings."
                )
            voice = raw_voice.strip()
            if voice:
                profiles[canonical_profile_name(raw_profile)] = voice
        return ProfileVoices(profiles=profiles)


def write_profile_voices(
    instance_path: str | Path | None,
    voices: ProfileVoices,
) -> Path:
    """Persist instance-local profile voice overrides."""
    with _STORE_LOCK:
        settings_path = get_profile_voices_path(instance_path)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": PROFILE_VOICES_VERSION,
            "profiles": dict(sorted(voices.profiles.items())),
        }
        temporary_path = settings_path.with_name(f".{settings_path.name}.{os.getpid()}.tmp")
        try:
            temporary_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
            temporary_path.replace(settings_path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove temporary profile voices file %s: %s", temporary_path, exc)
        return settings_path


def read_profile_voice_override(
    profile: str | None,
    instance_path: str | Path | None,
) -> str | None:
    """Return one profile's voice override, or None when it has none."""
    return read_profile_voices(instance_path).profiles.get(canonical_profile_name(profile))


def read_profile_voice(
    profile: str | None,
    instance_path: str | Path | None,
) -> str | None:
    """Return the effective voice for a profile, or None to use the backend default."""
    return read_profile_voice_override(profile, instance_path) or read_profile(profile).voice


def write_profile_voice_override(
    profile: str | None,
    voice: str,
    instance_path: str | Path | None,
) -> Path:
    """Write a voice override for one profile."""
    voice_name = voice.strip()
    if not voice_name:
        raise ValueError("A voice override cannot be empty; clear the override instead.")
    with _STORE_LOCK:
        voices = read_profile_voices(instance_path)
        profiles = dict(voices.profiles)
        profiles[canonical_profile_name(profile)] = voice_name
        return write_profile_voices(instance_path, ProfileVoices(profiles=profiles))


def clear_profile_voice_override(
    profile: str | None,
    instance_path: str | Path | None,
) -> bool:
    """Clear one profile's voice override and restore its authored voice."""
    with _STORE_LOCK:
        voices = read_profile_voices(instance_path)
        profile_name = canonical_profile_name(profile)
        if profile_name not in voices.profiles:
            return False

        profiles = dict(voices.profiles)
        del profiles[profile_name]
        settings_path = get_profile_voices_path(instance_path)
        if profiles:
            write_profile_voices(instance_path, ProfileVoices(profiles=profiles))
        elif settings_path.exists():
            settings_path.unlink()
        return True
