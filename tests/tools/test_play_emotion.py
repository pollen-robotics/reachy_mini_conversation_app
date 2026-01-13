"""Unit tests for the play_emotion tool."""

import builtins
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestPlayEmotionToolAttributes:
    """Tests for PlayEmotion tool attributes."""

    def test_play_emotion_has_correct_name(self) -> None:
        """Test PlayEmotion tool has correct name."""
        from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion

        tool = PlayEmotion()
        assert tool.name == "play_emotion"

    def test_play_emotion_has_description(self) -> None:
        """Test PlayEmotion tool has description."""
        from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion

        tool = PlayEmotion()
        assert "emotion" in tool.description.lower()

    def test_play_emotion_has_parameters_schema(self) -> None:
        """Test PlayEmotion tool has correct parameters schema."""
        from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion

        tool = PlayEmotion()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "emotion" in schema["properties"]
        assert schema["properties"]["emotion"]["type"] == "string"
        assert "emotion" in schema["required"]

    def test_play_emotion_spec(self) -> None:
        """Test PlayEmotion tool spec generation."""
        from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion

        tool = PlayEmotion()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "play_emotion"


class TestPlayEmotionToolExecution:
    """Tests for PlayEmotion tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_play_emotion_not_available_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test play_emotion returns error when emotion library not available."""
        from reachy_mini_conversation_app.tools import play_emotion

        # Temporarily set EMOTION_AVAILABLE to False
        original = play_emotion.EMOTION_AVAILABLE
        play_emotion.EMOTION_AVAILABLE = False

        try:
            tool = play_emotion.PlayEmotion()
            result = await tool(mock_deps, emotion="happy")

            assert "error" in result
            assert "not available" in result["error"]
        finally:
            play_emotion.EMOTION_AVAILABLE = original

    @pytest.mark.asyncio
    async def test_play_emotion_missing_emotion_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test play_emotion returns error when emotion name is missing."""
        from reachy_mini_conversation_app.tools import play_emotion

        if not play_emotion.EMOTION_AVAILABLE:
            pytest.skip("Emotion library not available")

        tool = play_emotion.PlayEmotion()
        result = await tool(mock_deps)

        assert "error" in result
        assert "required" in result["error"]

    @pytest.mark.asyncio
    async def test_play_emotion_empty_emotion_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test play_emotion returns error when emotion name is empty."""
        from reachy_mini_conversation_app.tools import play_emotion

        if not play_emotion.EMOTION_AVAILABLE:
            pytest.skip("Emotion library not available")

        tool = play_emotion.PlayEmotion()
        result = await tool(mock_deps, emotion="")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_play_emotion_unknown_emotion_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test play_emotion returns error for unknown emotion."""
        from reachy_mini_conversation_app.tools import play_emotion

        if not play_emotion.EMOTION_AVAILABLE:
            pytest.skip("Emotion library not available")

        tool = play_emotion.PlayEmotion()
        result = await tool(mock_deps, emotion="nonexistent_emotion_xyz")

        assert "error" in result
        assert "Unknown emotion" in result["error"]

    @pytest.mark.asyncio
    async def test_play_emotion_queues_move(self, mock_deps: ToolDependencies) -> None:
        """Test play_emotion queues the emotion move."""
        from reachy_mini_conversation_app.tools import play_emotion

        if not play_emotion.EMOTION_AVAILABLE or play_emotion.RECORDED_MOVES is None:
            pytest.skip("Emotion library not available")

        # Get a valid emotion name
        available_emotions = play_emotion.RECORDED_MOVES.list_moves()
        if not available_emotions:
            pytest.skip("No emotions available")

        emotion_name = available_emotions[0]

        tool = play_emotion.PlayEmotion()
        result = await tool(mock_deps, emotion=emotion_name)

        assert result["status"] == "queued"
        assert result["emotion"] == emotion_name
        mock_deps.movement_manager.queue_move.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_emotion_handles_exception(self, mock_deps: ToolDependencies) -> None:
        """Test play_emotion handles exceptions gracefully."""
        from reachy_mini_conversation_app.tools import play_emotion

        if not play_emotion.EMOTION_AVAILABLE or play_emotion.RECORDED_MOVES is None:
            pytest.skip("Emotion library not available")

        # Get a valid emotion name
        available_emotions = play_emotion.RECORDED_MOVES.list_moves()
        if not available_emotions:
            pytest.skip("No emotions available")

        emotion_name = available_emotions[0]

        # Make queue_move raise an exception
        mock_deps.movement_manager.queue_move.side_effect = RuntimeError("Test error")

        tool = play_emotion.PlayEmotion()
        result = await tool(mock_deps, emotion=emotion_name)

        assert "error" in result
        assert "Failed to play emotion" in result["error"]


class TestEmotionAvailability:
    """Tests for emotion library availability detection."""

    def test_emotion_available_flag_exists(self) -> None:
        """Test EMOTION_AVAILABLE flag exists."""
        from reachy_mini_conversation_app.tools import play_emotion

        assert hasattr(play_emotion, "EMOTION_AVAILABLE")
        assert isinstance(play_emotion.EMOTION_AVAILABLE, bool)

    def test_recorded_moves_exists(self) -> None:
        """Test RECORDED_MOVES exists."""
        from reachy_mini_conversation_app.tools import play_emotion

        assert hasattr(play_emotion, "RECORDED_MOVES")
        # Could be RecordedMoves object or None depending on library availability


class TestPlayEmotionImportFailure:
    """Tests for import failure handling."""

    def test_import_failure_sets_emotion_not_available(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test ImportError during emotion library import sets EMOTION_AVAILABLE to False."""
        import sys
        import importlib

        # Save original modules to restore later
        modules_to_remove = [
            "reachy_mini_conversation_app.tools.play_emotion",
            "reachy_mini.motion.recorded_move",
            "reachy_mini.motion",
            "reachy_mini",
            "reachy_mini_conversation_app.dance_emotion_moves",
        ]
        saved_modules = {name: sys.modules.get(name) for name in modules_to_remove}

        try:
            # Remove the modules so they can be reimported
            for name in modules_to_remove:
                if name in sys.modules:
                    del sys.modules[name]

            # Mock the import to fail
            original_import: Callable[..., Any] = builtins.__import__

            def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if "recorded_move" in name or "dance_emotion_moves" in name:
                    raise ImportError(f"Mocked import error for {name}")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = mock_import

            try:
                # Reimport the module
                play_emotion_module = importlib.import_module(
                    "reachy_mini_conversation_app.tools.play_emotion"
                )

                # Check that EMOTION_AVAILABLE is False
                assert play_emotion_module.EMOTION_AVAILABLE is False
                assert play_emotion_module.RECORDED_MOVES is None
                assert "not available" in caplog.text.lower()
            finally:
                builtins.__import__ = original_import
        finally:
            # Restore original modules
            for name, module in saved_modules.items():
                if module is not None:
                    sys.modules[name] = module
                elif name in sys.modules:
                    del sys.modules[name]

            # Force reimport to restore proper state
            if "reachy_mini_conversation_app.tools.play_emotion" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.tools.play_emotion"]


class TestGetAvailableEmotionsAndDescriptions:
    """Tests for get_available_emotions_and_descriptions function."""

    def test_returns_string(self) -> None:
        """Test function returns a string."""
        from reachy_mini_conversation_app.tools.play_emotion import get_available_emotions_and_descriptions

        result = get_available_emotions_and_descriptions()
        assert isinstance(result, str)

    def test_not_available_message(self) -> None:
        """Test function returns appropriate message when not available."""
        from reachy_mini_conversation_app.tools import play_emotion

        original = play_emotion.EMOTION_AVAILABLE
        play_emotion.EMOTION_AVAILABLE = False

        try:
            result = play_emotion.get_available_emotions_and_descriptions()
            assert "not available" in result.lower()
        finally:
            play_emotion.EMOTION_AVAILABLE = original

    def test_available_lists_emotions(self) -> None:
        """Test function lists emotions when available."""
        from reachy_mini_conversation_app.tools import play_emotion

        if not play_emotion.EMOTION_AVAILABLE:
            pytest.skip("Emotion library not available")

        result = play_emotion.get_available_emotions_and_descriptions()
        assert "Available emotions" in result or "Error" in result

    def test_handles_exception_in_list_moves(self) -> None:
        """Test function handles exceptions gracefully."""
        from reachy_mini_conversation_app.tools import play_emotion

        if not play_emotion.EMOTION_AVAILABLE:
            pytest.skip("Emotion library not available")

        original_recorded = play_emotion.RECORDED_MOVES

        # Mock RECORDED_MOVES to raise exception
        mock_moves = MagicMock()
        mock_moves.list_moves.side_effect = RuntimeError("Test error")
        play_emotion.RECORDED_MOVES = mock_moves

        try:
            result = play_emotion.get_available_emotions_and_descriptions()
            assert "Error" in result
        finally:
            play_emotion.RECORDED_MOVES = original_recorded
