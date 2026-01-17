"""Tests for memory synthesizer module."""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSynthesizeConversation:
    """Tests for synthesize_conversation function."""

    def test_empty_transcript_returns_existing(self, tmp_path: Path) -> None:
        """Test that empty transcript returns existing memories without API call."""
        memory_file = tmp_path / "memory.txt"
        memory_file.write_text("Existing memories", encoding="utf-8")

        with (
            patch("reachy_mini_conversation_app.memory.synthesizer.config") as mock_config,
            patch("reachy_mini_conversation_app.memory.store.config") as mock_store_config,
        ):
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            mock_config.MEMORY_MAX_TOKENS = 2000
            mock_store_config.MEMORY_FILE_PATH = str(memory_file)

            from reachy_mini_conversation_app.memory.synthesizer import (
                synthesize_conversation,
            )

            result = synthesize_conversation("")

            assert result == "Existing memories"

    def test_empty_transcript_whitespace_only(self, tmp_path: Path) -> None:
        """Test that whitespace-only transcript is treated as empty."""
        memory_file = tmp_path / "memory.txt"
        memory_file.write_text("Existing memories", encoding="utf-8")

        with (
            patch("reachy_mini_conversation_app.memory.synthesizer.config") as mock_config,
            patch("reachy_mini_conversation_app.memory.store.config") as mock_store_config,
        ):
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            mock_config.MEMORY_MAX_TOKENS = 2000
            mock_store_config.MEMORY_FILE_PATH = str(memory_file)

            from reachy_mini_conversation_app.memory.synthesizer import (
                synthesize_conversation,
            )

            result = synthesize_conversation("   \n\t  ")

            assert result == "Existing memories"

    def test_synthesize_calls_openai(self, tmp_path: Path) -> None:
        """Test that synthesize_conversation calls OpenAI API."""
        memory_file = tmp_path / "memory.txt"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Updated memory summary"

        with (
            patch("reachy_mini_conversation_app.memory.synthesizer.config") as mock_config,
            patch("reachy_mini_conversation_app.memory.synthesizer.OpenAI") as mock_openai_class,
            patch("reachy_mini_conversation_app.memory.store.config") as mock_store_config,
        ):
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            mock_config.MEMORY_MAX_TOKENS = 2000
            mock_store_config.MEMORY_FILE_PATH = str(memory_file)

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client

            from reachy_mini_conversation_app.memory.synthesizer import (
                synthesize_conversation,
            )

            synthesize_conversation("User: Hello\nAssistant: Hi there!")

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o-mini"
            assert "Hello" in call_kwargs["messages"][0]["content"]

    def test_synthesize_saves_result(self, tmp_path: Path) -> None:
        """Test that synthesize_conversation saves the result to file."""
        memory_file = tmp_path / "memory.txt"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "New memory content"

        with (
            patch("reachy_mini_conversation_app.memory.synthesizer.config") as mock_config,
            patch("reachy_mini_conversation_app.memory.synthesizer.OpenAI") as mock_openai_class,
            patch("reachy_mini_conversation_app.memory.store.config") as mock_store_config,
        ):
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            mock_config.MEMORY_MAX_TOKENS = 2000
            mock_store_config.MEMORY_FILE_PATH = str(memory_file)

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client

            from reachy_mini_conversation_app.memory.synthesizer import (
                synthesize_conversation,
            )

            synthesize_conversation("User: My name is Alice")

            assert memory_file.exists()
            assert memory_file.read_text(encoding="utf-8") == "New memory content"

    def test_synthesize_empty_response_keeps_existing(self, tmp_path: Path) -> None:
        """Test that empty LLM response preserves existing memory."""
        memory_file = tmp_path / "memory.txt"
        memory_file.write_text("Existing content", encoding="utf-8")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        with (
            patch("reachy_mini_conversation_app.memory.synthesizer.config") as mock_config,
            patch("reachy_mini_conversation_app.memory.synthesizer.OpenAI") as mock_openai_class,
            patch("reachy_mini_conversation_app.memory.store.config") as mock_store_config,
        ):
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            mock_config.MEMORY_MAX_TOKENS = 2000
            mock_store_config.MEMORY_FILE_PATH = str(memory_file)

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client

            from reachy_mini_conversation_app.memory.synthesizer import (
                synthesize_conversation,
            )

            result = synthesize_conversation("Some conversation")

            assert result == "Existing content"

    def test_synthesize_none_response_keeps_existing(self, tmp_path: Path) -> None:
        """Test that None LLM response preserves existing memory."""
        memory_file = tmp_path / "memory.txt"
        memory_file.write_text("Existing content", encoding="utf-8")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with (
            patch("reachy_mini_conversation_app.memory.synthesizer.config") as mock_config,
            patch("reachy_mini_conversation_app.memory.synthesizer.OpenAI") as mock_openai_class,
            patch("reachy_mini_conversation_app.memory.store.config") as mock_store_config,
        ):
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            mock_config.MEMORY_MAX_TOKENS = 2000
            mock_store_config.MEMORY_FILE_PATH = str(memory_file)

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client

            from reachy_mini_conversation_app.memory.synthesizer import (
                synthesize_conversation,
            )

            result = synthesize_conversation("Some conversation")

            assert result == "Existing content"


class TestCondenseMemory:
    """Tests for _condense_memory function."""

    def test_condense_memory_calls_api(self) -> None:
        """Test that _condense_memory calls OpenAI API."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Condensed memory"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("reachy_mini_conversation_app.memory.synthesizer.config") as mock_config:
            mock_config.MEMORY_MAX_TOKENS = 500

            from reachy_mini_conversation_app.memory.synthesizer import _condense_memory

            result = _condense_memory(mock_client, "Very long memory text")

            mock_client.chat.completions.create.assert_called_once()
            assert result == "Condensed memory"

    def test_condense_memory_returns_original_on_none(self) -> None:
        """Test that _condense_memory returns original if API returns None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("reachy_mini_conversation_app.memory.synthesizer.config") as mock_config:
            mock_config.MEMORY_MAX_TOKENS = 500

            from reachy_mini_conversation_app.memory.synthesizer import _condense_memory

            result = _condense_memory(mock_client, "Original memory")

            assert result == "Original memory"


class TestTokenLimitHandling:
    """Tests for token limit handling in synthesize_conversation."""

    def test_exceeds_token_limit_triggers_condensation(self, tmp_path: Path) -> None:
        """Test that exceeding token limit triggers condensation."""
        memory_file = tmp_path / "memory.txt"

        # Response that exceeds token limit (8004 chars = 2001 tokens > 2000)
        long_response = "x" * 8004

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = long_response

        condense_response = MagicMock()
        condense_response.choices = [MagicMock()]
        condense_response.choices[0].message.content = "Condensed"

        with (
            patch("reachy_mini_conversation_app.memory.synthesizer.config") as mock_config,
            patch("reachy_mini_conversation_app.memory.synthesizer.OpenAI") as mock_openai_class,
            patch("reachy_mini_conversation_app.memory.store.config") as mock_store_config,
        ):
            mock_config.MEMORY_FILE_PATH = str(memory_file)
            mock_config.MEMORY_MAX_TOKENS = 2000
            mock_store_config.MEMORY_FILE_PATH = str(memory_file)

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [mock_response, condense_response]
            mock_openai_class.return_value = mock_client

            from reachy_mini_conversation_app.memory.synthesizer import (
                synthesize_conversation,
            )

            result = synthesize_conversation("Test conversation")

            # Should have called API twice (synthesize + condense)
            assert mock_client.chat.completions.create.call_count == 2
            assert result == "Condensed"
