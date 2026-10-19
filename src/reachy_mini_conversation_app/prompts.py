"""Resolve active profile prompts, voice and language settings."""

import logging
from pathlib import Path

from reachy_mini_conversation_app.config import HF_SPEECH_LANGUAGES, config, get_default_voice
from reachy_mini_conversation_app.memory import format_memory_for_prompt
from reachy_mini_conversation_app.profile_store import (
    DEFAULT_PROFILE_NAME,
    ProfileDefinition,
    ProfileFormatError,
    read_profile,
    canonical_profile_name,
    read_packaged_default_profile,
)


logger = logging.getLogger(__name__)

DEFAULT_GREETING_PROMPT = (
    "Start the conversation now with a brief, spontaneous greeting in character. "
    "Keep it to one sentence, invite the user in naturally, and vary the wording each time."
)


def _active_profile() -> ProfileDefinition:
    return read_profile(config.REACHY_MINI_CUSTOM_PROFILE)


def get_session_language() -> str:
    """Return the language saved for the active profile, else the one saved for all profiles."""
    profile_name = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
    return config.PROFILE_LANGUAGES.get(profile_name, config.REALTIME_TRANSCRIPTION_LANGUAGE)


def get_session_instructions(instance_path: str | Path | None = None) -> str:
    """Return instructions for the active profile with memory context and its language."""
    selected_profile = config.REACHY_MINI_CUSTOM_PROFILE
    profile_name = selected_profile or DEFAULT_PROFILE_NAME
    try:
        profile = _active_profile()
        instructions = profile.instructions.strip()
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load profile %r: %s", profile_name, exc)
        instructions = ""

    if not instructions and selected_profile and selected_profile != DEFAULT_PROFILE_NAME:
        logger.warning("Using bundled default instructions because profile %r is incomplete", selected_profile)
        try:
            instructions = read_packaged_default_profile().instructions.strip()
        except (FileNotFoundError, ProfileFormatError) as exc:
            raise RuntimeError("Default profile has no usable instructions") from exc
    if not instructions:
        raise RuntimeError("Default profile has no usable instructions")

    # The backend forces speech recognition and synthesis to this language, but not the model's replies.
    language = get_session_language()
    if language != "auto":
        language_name = next(
            (choice.name for choice in HF_SPEECH_LANGUAGES if choice.code == language),
            f"the language with ISO 639-1 code {language!r}",
        )
        instructions = (
            f"{instructions}\n\nSpeak only {language_name}, starting with your first greeting, "
            "whatever language these instructions are written in."
        )

    memory_prompt = format_memory_for_prompt(instance_path)
    if memory_prompt:
        return f"{memory_prompt}\n\n{instructions}"
    return instructions


def get_session_voice(default: str | None = None) -> str:
    """Return the active profile voice or the backend default."""
    fallback = get_default_voice() if default is None else default
    try:
        return _active_profile().voice or fallback
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile voice: %s", exc)
        return fallback


def get_session_greeting_prompt() -> str:
    """Return the active profile greeting prompt or the app default."""
    try:
        return _active_profile().greeting or DEFAULT_GREETING_PROMPT
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile greeting: %s", exc)
        return DEFAULT_GREETING_PROMPT
