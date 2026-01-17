"""Memory storage utilities for reading and writing memory files."""

import logging
from pathlib import Path

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


def load_memories() -> str:
    """Load existing memory summary from file.

    Returns:
        The memory summary as a string, or empty string if no file exists.

    """
    path = Path(config.MEMORY_FILE_PATH)
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        logger.debug(f"Loaded memories from {path} ({len(content)} chars)")
        return content
    logger.debug(f"No memory file found at {path}")
    return ""


def save_memories(summary: str) -> None:
    """Save memory summary to file, creating parent directories if needed.

    Args:
        summary: The memory summary to save.

    """
    path = Path(config.MEMORY_FILE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")
    logger.info(f"Memory saved to {path} ({len(summary)} chars)")


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic of ~4 characters per token, which is a reasonable
    approximation for English and many other languages.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated number of tokens.

    """
    return len(text) // 4
