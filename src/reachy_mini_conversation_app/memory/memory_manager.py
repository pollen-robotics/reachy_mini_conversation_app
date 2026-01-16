"""Memory manager singleton wrapping storage operations."""

import logging
from pathlib import Path
from typing import Optional

from reachy_mini_conversation_app.memory.storage import MemoryStorage
from reachy_mini_conversation_app.memory.types import MAX_ROBOT_BLOCKS, BlockType, MemoryBlock, MemorySearchResult


logger = logging.getLogger(__name__)


class MemoryManager:
    """Singleton manager for memory operations with validation and business logic."""

    _instance: Optional["MemoryManager"] = None
    _initialized: bool = False

    def __new__(cls, db_path: Optional[str | Path] = None) -> "MemoryManager":
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        """Initialize memory manager (only once).

        Args:
            db_path: Path to SQLite database file
        """
        # Only initialize once
        if self._initialized:
            return

        if db_path is None:
            # Default path
            db_path = Path.home() / ".reachy_mini" / "memory.db"

        self.storage = MemoryStorage(db_path)
        self._initialized = True
        logger.info("MemoryManager initialized with database at %s", db_path)

    async def initialize(self) -> None:
        """Initialize the underlying storage."""
        await self.storage.initialize()

    async def get_block(self, block_type: BlockType, label: Optional[str] = None) -> Optional[MemoryBlock]:
        """Get a memory block.

        Args:
            block_type: Type of block (user, facts, robot)
            label: For robot blocks, the custom label. For user/facts, can be None.

        Returns:
            MemoryBlock if found, None otherwise
        """
        if block_type == "robot":
            if not label:
                raise ValueError("Robot blocks require a label")
            full_label = f"robot:{label}"
        else:
            full_label = block_type

        return await self.storage.get_block(full_label)

    async def update_block(
        self,
        block_type: BlockType,
        content: str,
        label: Optional[str] = None,
    ) -> MemoryBlock:
        """Update or create a memory block with validation.

        Args:
            block_type: Type of block (user, facts, robot)
            content: Content to store
            label: For robot blocks, the custom label. For user/facts, can be None.

        Returns:
            The updated MemoryBlock

        Raises:
            ValueError: If validation fails (char limit, max robot blocks, etc.)
        """
        # Construct the full label
        if block_type == "robot":
            if not label:
                raise ValueError("Robot blocks require a label")
            full_label = f"robot:{label}"
        else:
            full_label = block_type

        # Get char limit for this block type
        from reachy_mini_conversation_app.memory.types import DEFAULT_LIMITS
        char_limit = DEFAULT_LIMITS.get(block_type, 10000)

        # Validate content length
        if len(content) > char_limit:
            raise ValueError(
                f"Content exceeds character limit of {char_limit} for {block_type} blocks "
                f"(got {len(content)} characters)"
            )

        # For robot blocks, enforce max count
        if block_type == "robot":
            existing_block = await self.storage.get_block(full_label)
            if not existing_block:
                # Check if we're at the limit
                all_blocks = await self.storage.get_all_blocks()
                robot_blocks = [b for b in all_blocks if b.block_type == "robot"]
                if len(robot_blocks) >= MAX_ROBOT_BLOCKS:
                    raise ValueError(
                        f"Maximum of {MAX_ROBOT_BLOCKS} robot blocks allowed. "
                        f"Delete an existing robot block before creating a new one."
                    )

        return await self.storage.upsert_block(block_type, full_label, content, char_limit)

    async def delete_block(self, block_type: BlockType, label: Optional[str] = None) -> bool:
        """Delete a memory block.

        Args:
            block_type: Type of block (user, facts, robot)
            label: For robot blocks, the custom label. For user/facts, can be None.

        Returns:
            True if deleted, False if not found
        """
        if block_type == "robot":
            if not label:
                raise ValueError("Robot blocks require a label")
            full_label = f"robot:{label}"
        else:
            full_label = block_type

        return await self.storage.delete_block(full_label)

    async def get_all_blocks(self) -> list[MemoryBlock]:
        """Get all memory blocks.

        Returns:
            List of all MemoryBlock objects
        """
        return await self.storage.get_all_blocks()

    async def search_blocks(self, query: str) -> list[MemorySearchResult]:
        """Search memory blocks using full-text search.

        Args:
            query: Search query string

        Returns:
            List of MemorySearchResult objects
        """
        return await self.storage.search(query)


# Singleton instance accessor
_manager_instance: Optional[MemoryManager] = None


def get_memory_manager(db_path: Optional[str | Path] = None) -> MemoryManager:
    """Get or create the global MemoryManager instance.

    Args:
        db_path: Path to database (only used on first call)

    Returns:
        MemoryManager singleton instance
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = MemoryManager(db_path)
    return _manager_instance
