import re
import sys
import logging
from pathlib import Path

from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config, get_default_voice_for_backend
from reachy_mini_conversation_app.memory import format_memory_for_prompt


logger = logging.getLogger(__name__)


INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"
GREETING_FILENAME = "greeting.txt"
DEFAULT_PROFILE_NAME = "default"
DEFAULT_PROMPT_INCLUDE = "default_prompt"

DEFAULT_GREETING_PROMPT = (
    "Start the conversation now with a brief, spontaneous greeting in character. "
    "Keep it to one sentence, invite the user in naturally, and vary the wording each time."
)


def _default_instructions_file() -> Path:
    return DEFAULT_PROFILES_DIRECTORY / DEFAULT_PROFILE_NAME / INSTRUCTIONS_FILENAME


def _expand_prompt_includes(content: str) -> str:
    """Expand supported prompt placeholders."""
    pattern = re.compile(r"^\[([a-zA-Z0-9_-]+)\]$")

    lines = content.split("\n")
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if not match:
            expanded_lines.append(line)
            continue

        template_name = match.group(1)
        if template_name != DEFAULT_PROMPT_INCLUDE:
            logger.warning("Prompt include not found: [%s], keeping placeholder", template_name)
            expanded_lines.append(line)
            continue

        try:
            template_content = _default_instructions_file().read_text(encoding="utf-8").rstrip()
        except OSError as e:
            logger.warning("Failed to read prompt include '%s': %s, keeping placeholder", template_name, e)
            expanded_lines.append(line)
            continue

        expanded_lines.append(template_content)
        logger.debug("Expanded template: [%s]", template_name)

    return "\n".join(expanded_lines)


def get_session_instructions(instance_path: str | Path | None = None) -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    profile_name = profile or DEFAULT_PROFILE_NAME
    if not profile:
        instructions_file = _default_instructions_file()
        logger.info("Loading default prompt from %s", instructions_file)
    else:
        if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.info(
                "Loading prompt from external profile '%s' (root=%s)",
                profile,
                config.PROFILES_DIRECTORY,
            )
        else:
            logger.info("Loading prompt from profile '%s'", profile)
        instructions_file = config.resolve_profile_dir(profile) / INSTRUCTIONS_FILENAME

    try:
        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                expanded_instructions = _expand_prompt_includes(instructions)
                memory_prompt = format_memory_for_prompt(instance_path)
                if memory_prompt:
                    return f"{memory_prompt}\n\n{expanded_instructions}"
                return expanded_instructions
            logger.error("Profile '%s' has empty %s", profile_name, INSTRUCTIONS_FILENAME)
            sys.exit(1)
        logger.error("Profile '%s' has no %s", profile_name, INSTRUCTIONS_FILENAME)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load instructions from profile '%s': %s", profile_name, e)
        sys.exit(1)


def get_session_voice(default: str | None = None) -> str:
    """Resolve the voice to use for the session.

    If a custom profile is selected and contains a voice.txt, return its
    trimmed content; otherwise return the provided default or the active
    backend default voice.
    """
    fallback = get_default_voice_for_backend() if default is None else default
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return fallback
    try:
        voice_file = config.resolve_profile_dir(profile) / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or fallback
    except Exception:
        pass
    return fallback


def get_session_greeting_prompt() -> str:
    """Resolve the startup greeting prompt for the selected profile."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return DEFAULT_GREETING_PROMPT

    try:
        greeting_file = config.resolve_profile_dir(profile) / GREETING_FILENAME
        if greeting_file.exists():
            greeting = greeting_file.read_text(encoding="utf-8").strip()
            if greeting:
                return _expand_prompt_includes(greeting)
    except Exception as e:
        logger.warning("Failed to load greeting prompt from profile %r: %s", profile, e)
    return DEFAULT_GREETING_PROMPT
