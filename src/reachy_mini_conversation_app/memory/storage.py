"""SQLite storage backend for memory system with FTS5 full-text search."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from reachy_mini_conversation_app.memory.types import DEFAULT_LIMITS, BlockType, MemoryBlock, MemorySearchResult


logger = logging.getLogger(__name__)


class MemoryStorage:
    """Async SQLite storage with FTS5 full-text search for memory blocks."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize storage with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path).expanduser()
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Create the directory for the database if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Memory database directory: {self.db_path.parent}")

    async def initialize(self) -> None:
        """Initialize database schema and FTS5 virtual table."""
        async with aiosqlite.connect(self.db_path) as db:
            # Create main memory_blocks table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_type TEXT NOT NULL,
                    label TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    char_limit INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # Create FTS5 virtual table for full-text search
            await db.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    label,
                    content,
                    content='memory_blocks',
                    content_rowid='id'
                )
                """
            )

            # Create triggers to keep FTS5 in sync with main table
            await db.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memory_fts_insert AFTER INSERT ON memory_blocks BEGIN
                    INSERT INTO memory_fts(rowid, label, content)
                    VALUES (new.id, new.label, new.content);
                END
                """
            )

            await db.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memory_fts_update AFTER UPDATE ON memory_blocks BEGIN
                    UPDATE memory_fts SET label = new.label, content = new.content
                    WHERE rowid = new.id;
                END
                """
            )

            await db.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memory_fts_delete AFTER DELETE ON memory_blocks BEGIN
                    DELETE FROM memory_fts WHERE rowid = old.id;
                END
                """
            )

            await db.commit()
            logger.info(f"Memory database initialized at {self.db_path}")

    async def upsert_block(
        self,
        block_type: BlockType,
        label: str,
        content: str,
        char_limit: Optional[int] = None,
    ) -> MemoryBlock:
        """Insert or update a memory block.

        Args:
            block_type: Type of block (user, facts, robot)
            label: Unique label for the block
            content: Text content
            char_limit: Character limit (defaults to DEFAULT_LIMITS)

        Returns:
            The created or updated MemoryBlock
        """
        if char_limit is None:
            char_limit = DEFAULT_LIMITS.get(block_type, 10000)

        now = datetime.now().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Check if block exists
            cursor = await db.execute(
                "SELECT id, created_at FROM memory_blocks WHERE label = ?",
                (label,),
            )
            row = await cursor.fetchone()

            if row:
                # Update existing block
                block_id, created_at = row
                await db.execute(
                    """
                    UPDATE memory_blocks
                    SET block_type = ?, content = ?, char_limit = ?, updated_at = ?
                    WHERE label = ?
                    """,
                    (block_type, content, char_limit, now, label),
                )
            else:
                # Insert new block
                cursor = await db.execute(
                    """
                    INSERT INTO memory_blocks (block_type, label, content, char_limit, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (block_type, label, content, char_limit, now, now),
                )
                block_id = cursor.lastrowid
                created_at = now

            await db.commit()

            return MemoryBlock(
                id=block_id,
                block_type=block_type,
                label=label,
                content=content,
                char_limit=char_limit,
                created_at=datetime.fromisoformat(created_at),
                updated_at=datetime.fromisoformat(now),
            )

    async def get_block(self, label: str) -> Optional[MemoryBlock]:
        """Retrieve a memory block by label.

        Args:
            label: Unique label of the block

        Returns:
            MemoryBlock if found, None otherwise
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM memory_blocks WHERE label = ?",
                (label,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return MemoryBlock(
                id=row["id"],
                block_type=row["block_type"],
                label=row["label"],
                content=row["content"],
                char_limit=row["char_limit"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    async def get_all_blocks(self) -> list[MemoryBlock]:
        """Retrieve all memory blocks.

        Returns:
            List of all MemoryBlock objects
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM memory_blocks ORDER BY block_type, label"
            )
            rows = await cursor.fetchall()

            return [
                MemoryBlock(
                    id=row["id"],
                    block_type=row["block_type"],
                    label=row["label"],
                    content=row["content"],
                    char_limit=row["char_limit"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                for row in rows
            ]

    async def delete_block(self, label: str) -> bool:
        """Delete a memory block by label.

        Args:
            label: Unique label of the block

        Returns:
            True if deleted, False if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM memory_blocks WHERE label = ?",
                (label,),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def search(self, query: str) -> list[MemorySearchResult]:
        """Full-text search across all memory blocks using FTS5.

        Args:
            query: Search query string

        Returns:
            List of MemorySearchResult objects with matching blocks and snippets
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Use FTS5 match query with snippet extraction
            cursor = await db.execute(
                """
                SELECT
                    m.*,
                    snippet(memory_fts, 1, '<b>', '</b>', '...', 32) as snippet
                FROM memory_blocks m
                JOIN memory_fts ON m.id = memory_fts.rowid
                WHERE memory_fts MATCH ?
                ORDER BY rank
                """,
                (query,),
            )
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                block = MemoryBlock(
                    id=row["id"],
                    block_type=row["block_type"],
                    label=row["label"],
                    content=row["content"],
                    char_limit=row["char_limit"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )

                # Extract matches from snippet (text between <b> and </b>)
                snippet_text = row["snippet"]
                matches = []
                if snippet_text:
                    # Simple extraction - could be enhanced
                    matches = [snippet_text]

                results.append(MemorySearchResult(block=block, matches=matches))

            return results
