"""Memory module facade for integration with the conversation app."""

import logging
from pathlib import Path
from typing import Optional

from reachy_mini_conversation_app.memory.memory_manager import get_memory_manager


logger = logging.getLogger(__name__)


class MemoryModule:
    """Facade for memory system integration with prompt injection."""

    _instance: Optional["MemoryModule"] = None

    def __new__(cls) -> "MemoryModule":
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize memory module."""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            logger.info("MemoryModule initialized")

    def is_enabled(self) -> bool:
        """Check if memory system is enabled via config.

        Returns:
            True if MEMORY_ENABLED is true in config
        """
        try:
            from reachy_mini_conversation_app.config import config
            return getattr(config, "MEMORY_ENABLED", True)
        except Exception as e:
            logger.warning("Failed to check MEMORY_ENABLED config: %s", e)
            return True  # Default to enabled

    async def get_memory_context(self) -> str:
        """Compile memory blocks into XML format for prompt injection.

        Returns:
            XML-formatted string containing memory context, or empty string if disabled or no memory
        """
        if not self.is_enabled():
            return ""

        try:
            manager = get_memory_manager()
            await manager.initialize()

            all_blocks = await manager.get_all_blocks()

            if not all_blocks:
                return ""

            # Build XML structure
            lines = ["<Memory>"]

            # Add user block if exists
            user_blocks = [b for b in all_blocks if b.block_type == "user"]
            if user_blocks:
                lines.append("<user>")
                for block in user_blocks:
                    lines.append(block.content)
                lines.append("</user>")

            # Add facts block if exists
            facts_blocks = [b for b in all_blocks if b.block_type == "facts"]
            if facts_blocks:
                lines.append("<facts>")
                for block in facts_blocks:
                    lines.append(block.content)
                lines.append("</facts>")

            # Add robot blocks if exist
            robot_blocks = [b for b in all_blocks if b.block_type == "robot"]
            if robot_blocks:
                lines.append("<robot_blocks>")
                for block in robot_blocks:
                    # Extract custom label from "robot:<label>"
                    custom_label = block.label.split(":", 1)[1] if ":" in block.label else block.label
                    lines.append(f'<robot name="{custom_label}">')
                    lines.append(block.content)
                    lines.append("</robot>")
                lines.append("</robot_blocks>")

            lines.append("</Memory>")

            return "\n".join(lines)

        except Exception as e:
            logger.error("Failed to generate memory context: %s", e)
            return ""

    def get_memory_instructions(self) -> str:
        """Load memory instructions from prompts/memory_instructions.txt.

        Returns:
            Memory instructions text, or empty string if not found or disabled
        """
        if not self.is_enabled():
            return ""

        try:
            # Look for memory_instructions.txt in prompts directory
            prompts_dir = Path(__file__).parent.parent / "prompts"
            instructions_file = prompts_dir / "memory_instructions.txt"

            if instructions_file.exists():
                return instructions_file.read_text(encoding="utf-8").strip()
            else:
                logger.warning("memory_instructions.txt not found at %s", instructions_file)
                return ""

        except Exception as e:
            logger.error("Failed to load memory instructions: %s", e)
            return ""


# Singleton instance accessor
_module_instance: Optional[MemoryModule] = None


def get_memory_module() -> MemoryModule:
    """Get or create the global MemoryModule instance.

    Returns:
        MemoryModule singleton instance
    """
    global _module_instance
    if _module_instance is None:
        _module_instance = MemoryModule()
    return _module_instance
