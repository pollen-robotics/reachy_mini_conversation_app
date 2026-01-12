"""Tests for memory store module."""

from pathlib import Path
from unittest.mock import patch


class TestLoadMemories:
    """Tests for load_memories function."""

    def test_load_memories_file_exists(self, tmp_path: Path) -> None:
        """Test loading memories when file exists."""
        memory_file = tmp_path / "memory.txt"
        memory_file.write_text("User is Alice, a developer.", encoding="utf-8")

        with patch("reachy_mini_conversation_app.memory.store.config") as mock_config:
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            from reachy_mini_conversation_app.memory.store import load_memories

            result = load_memories()
            assert result == "User is Alice, a developer."

    def test_load_memories_file_not_exists(self, tmp_path: Path) -> None:
        """Test loading memories when file does not exist."""
        memory_file = tmp_path / "nonexistent.txt"

        with patch("reachy_mini_conversation_app.memory.store.config") as mock_config:
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            from reachy_mini_conversation_app.memory.store import load_memories

            result = load_memories()
            assert result == ""

    def test_load_memories_strips_whitespace(self, tmp_path: Path) -> None:
        """Test that loaded memories are stripped of whitespace."""
        memory_file = tmp_path / "memory.txt"
        memory_file.write_text("  User prefers Python.  \n\n", encoding="utf-8")

        with patch("reachy_mini_conversation_app.memory.store.config") as mock_config:
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            from reachy_mini_conversation_app.memory.store import load_memories

            result = load_memories()
            assert result == "User prefers Python."


class TestSaveMemories:
    """Tests for save_memories function."""

    def test_save_memories_creates_file(self, tmp_path: Path) -> None:
        """Test saving memories creates the file."""
        memory_file = tmp_path / "memory.txt"

        with patch("reachy_mini_conversation_app.memory.store.config") as mock_config:
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            from reachy_mini_conversation_app.memory.store import save_memories

            save_memories("User is Bob.")

            assert memory_file.exists()
            assert memory_file.read_text(encoding="utf-8") == "User is Bob."

    def test_save_memories_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test saving memories creates parent directories if needed."""
        memory_file = tmp_path / "subdir" / "nested" / "memory.txt"

        with patch("reachy_mini_conversation_app.memory.store.config") as mock_config:
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            from reachy_mini_conversation_app.memory.store import save_memories

            save_memories("User likes robotics.")

            assert memory_file.exists()
            assert memory_file.read_text(encoding="utf-8") == "User likes robotics."

    def test_save_memories_overwrites_existing(self, tmp_path: Path) -> None:
        """Test saving memories overwrites existing content."""
        memory_file = tmp_path / "memory.txt"
        memory_file.write_text("Old content", encoding="utf-8")

        with patch("reachy_mini_conversation_app.memory.store.config") as mock_config:
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            from reachy_mini_conversation_app.memory.store import save_memories

            save_memories("New content")

            assert memory_file.read_text(encoding="utf-8") == "New content"


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_estimate_tokens_empty_string(self) -> None:
        """Test token estimation for empty string."""
        from reachy_mini_conversation_app.memory.store import estimate_tokens

        assert estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self) -> None:
        """Test token estimation for short text."""
        from reachy_mini_conversation_app.memory.store import estimate_tokens

        # 12 characters -> 3 tokens
        assert estimate_tokens("Hello World!") == 3

    def test_estimate_tokens_longer_text(self) -> None:
        """Test token estimation for longer text."""
        from reachy_mini_conversation_app.memory.store import estimate_tokens

        # 100 characters -> 25 tokens
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_estimate_tokens_approximation(self) -> None:
        """Test that token estimation is approximately 4 chars per token."""
        from reachy_mini_conversation_app.memory.store import estimate_tokens

        text = "This is a sample text for testing token estimation."
        estimated = estimate_tokens(text)
        # Should be roughly len(text) / 4
        assert estimated == len(text) // 4
