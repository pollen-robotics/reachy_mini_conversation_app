"""Memory system type definitions."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


BlockType = Literal["user", "facts", "robot"]


@dataclass
class MemoryBlock:
    """A memory block containing information of a specific type."""

    id: int
    block_type: BlockType
    label: str  # 'user', 'facts', or 'robot:<name>'
    content: str
    char_limit: int
    created_at: datetime
    updated_at: datetime


@dataclass
class MemorySearchResult:
    """Search result containing a memory block and matching text snippets."""

    block: MemoryBlock
    matches: list[str]


DEFAULT_LIMITS = {"user": 10000, "facts": 10000, "robot": 10000}
MAX_ROBOT_BLOCKS = 4
