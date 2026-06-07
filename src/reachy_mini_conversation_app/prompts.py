import re
import sys
import logging
from pathlib import Path

from reachy_mini_conversation_app.config import (
    DEFAULT_PROFILES_DIRECTORY,
    PROMPT_LANGUAGE_ENV,
    config,
    get_default_voice_for_backend,
    normalize_prompt_language,
)


logger = logging.getLogger(__name__)


PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"
DEFAULT_PROMPT_FILENAME = "default_prompt.txt"
PROMPT_LANGUAGE_FILENAME_BY_LANGUAGE = {
    "zh": "default_prompt.zh.txt",
    "en": "default_prompt.en.txt",
}


def get_prompt_language() -> str:
    """Return the selected prompt language, resolving auto from runtime config."""
    selected = normalize_prompt_language(getattr(config, "PROMPT_LANGUAGE", None))
    if selected != "auto":
        return selected

    transcription_language = getattr(config, "INPUT_TRANSCRIPTION_LANGUAGE", None)
    normalized_transcription_language = normalize_prompt_language(transcription_language)
    if normalized_transcription_language != "auto":
        return normalized_transcription_language

    return "en"


def _language_specific_file(base_file: Path, language: str) -> Path:
    """Return a language-specific sibling when it exists."""
    if language not in {"zh", "en"}:
        return base_file

    if base_file.name == DEFAULT_PROMPT_FILENAME:
        candidate = base_file.with_name(PROMPT_LANGUAGE_FILENAME_BY_LANGUAGE[language])
    else:
        candidate = base_file.with_name(f"{base_file.stem}.{language}{base_file.suffix}")

    return candidate if candidate.exists() else base_file


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    # pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')
    # Allow slashes for subdirectories
    pattern = re.compile(r"^\[([a-zA-Z0-9/_-]+)\]$")

    lines = content.split("\n")
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return "\n".join(expanded_lines)


def get_session_instructions(language: str | None = None) -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    prompt_language = get_prompt_language() if language is None else normalize_prompt_language(language)
    if prompt_language == "auto":
        prompt_language = get_prompt_language()

    if not profile:
        base_file = PROMPTS_LIBRARY_DIRECTORY / DEFAULT_PROMPT_FILENAME
        instructions_file = _language_specific_file(base_file, prompt_language)
        logger.info(
            "Loading default prompt from %s (language=%s via %s)",
            instructions_file,
            prompt_language,
            PROMPT_LANGUAGE_ENV,
        )
    else:
        if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.info(
                "Loading prompt from external profile '%s' (root=%s)",
                profile,
                config.PROFILES_DIRECTORY,
            )
        else:
            logger.info(f"Loading prompt from profile '{profile}'")
        base_file = config.PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME
        instructions_file = _language_specific_file(base_file, prompt_language)

    try:
        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                # Expand [<name>] placeholders with content from prompts library
                expanded_instructions = _expand_prompt_includes(instructions)
                return expanded_instructions
            logger.error(f"Profile '{profile}' has empty {INSTRUCTIONS_FILENAME}")
            sys.exit(1)
        logger.error(f"Profile {profile} has no {INSTRUCTIONS_FILENAME}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load instructions from profile '{profile}': {e}")
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
        voice_file = config.PROFILES_DIRECTORY / profile / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or fallback
    except Exception:
        pass
    return fallback
