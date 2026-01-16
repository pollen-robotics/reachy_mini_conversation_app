import re
import sys
import asyncio
import logging
from pathlib import Path

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


PROFILES_DIRECTORY = Path(__file__).parent / "profiles"
PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"


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
    pattern = re.compile(r'^\[([a-zA-Z0-9/_-]+)\]$')

    lines = content.split('\n')
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

    return '\n'.join(expanded_lines)


def _inject_memory_context(instructions: str) -> str:
    """Inject memory instructions and context into the prompt.

    Args:
        instructions: Base instructions

    Returns:
        Instructions with memory context appended (if enabled)
    """
    try:
        from reachy_mini_conversation_app.memory import get_memory_module

        memory_module = get_memory_module()

        if not memory_module.is_enabled():
            logger.debug("Memory system disabled, skipping injection")
            return instructions

        # Load memory instructions
        memory_instructions = memory_module.get_memory_instructions()

        # Get memory context (async operation, run in event loop)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, we can't use asyncio.run()
                # Schedule as a task and wait for it
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    memory_context = loop.run_in_executor(
                        pool, lambda: asyncio.run(memory_module.get_memory_context())
                    )
                    memory_context = asyncio.get_event_loop().run_until_complete(memory_context)
            else:
                # No event loop running, safe to use asyncio.run()
                memory_context = asyncio.run(memory_module.get_memory_context())
        except RuntimeError:
            # Fallback: create new event loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                memory_context = new_loop.run_until_complete(memory_module.get_memory_context())
            finally:
                new_loop.close()

        # Append memory instructions and context
        parts = [instructions]

        if memory_instructions:
            parts.append("\n\n" + memory_instructions)

        if memory_context:
            parts.append("\n\n" + memory_context)

        return "".join(parts)

    except Exception as e:
        logger.warning("Failed to inject memory context: %s", e)
        return instructions


def get_session_instructions() -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.info(f"Loading default prompt from {PROMPTS_LIBRARY_DIRECTORY / 'default_prompt.txt'}")
        instructions_file = PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
    else:
        logger.info(f"Loading prompt from profile '{profile}'")
        instructions_file = PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME

    try:
        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                # Expand [<name>] placeholders with content from prompts library
                expanded_instructions = _expand_prompt_includes(instructions)

                # Inject memory instructions and context if enabled
                expanded_instructions = _inject_memory_context(expanded_instructions)

                return expanded_instructions
            logger.error(f"Profile '{profile}' has empty {INSTRUCTIONS_FILENAME}")
            sys.exit(1)
        logger.error(f"Profile {profile} has no {INSTRUCTIONS_FILENAME}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load instructions from profile '{profile}': {e}")
        sys.exit(1)


def get_session_voice(default: str = "cedar") -> str:
    """Resolve the voice to use for the session.

    If a custom profile is selected and contains a voice.txt, return its
    trimmed content; otherwise return the provided default ("cedar").
    """
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return default
    try:
        voice_file = PROFILES_DIRECTORY / profile / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or default
    except Exception:
        pass
    return default
