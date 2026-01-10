"""Unit tests for the dance tool."""

from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestDanceToolAttributes:
    """Tests for Dance tool attributes."""

    def test_dance_has_correct_name(self) -> None:
        """Test Dance tool has correct name."""
        from reachy_mini_conversation_app.tools.dance import Dance

        tool = Dance()
        assert tool.name == "dance"

    def test_dance_has_description(self) -> None:
        """Test Dance tool has description."""
        from reachy_mini_conversation_app.tools.dance import Dance

        tool = Dance()
        assert "dance" in tool.description.lower()

    def test_dance_has_parameters_schema(self) -> None:
        """Test Dance tool has correct parameters schema."""
        from reachy_mini_conversation_app.tools.dance import Dance

        tool = Dance()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "move" in schema["properties"]
        assert "repeat" in schema["properties"]
        assert schema["properties"]["move"]["type"] == "string"
        assert schema["properties"]["repeat"]["type"] == "integer"

    def test_dance_spec(self) -> None:
        """Test Dance tool spec generation."""
        from reachy_mini_conversation_app.tools.dance import Dance

        tool = Dance()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "dance"


class TestDanceToolExecution:
    """Tests for Dance tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_dance_not_available_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test dance returns error when dance library not available."""
        from reachy_mini_conversation_app.tools import dance

        # Temporarily set DANCE_AVAILABLE to False
        original = dance.DANCE_AVAILABLE
        dance.DANCE_AVAILABLE = False

        try:
            tool = dance.Dance()
            result = await tool(mock_deps)

            assert "error" in result
            assert "not available" in result["error"]
        finally:
            dance.DANCE_AVAILABLE = original

    @pytest.mark.asyncio
    async def test_dance_unknown_move_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test dance returns error for unknown move."""
        from reachy_mini_conversation_app.tools import dance

        if not dance.DANCE_AVAILABLE:
            pytest.skip("Dance library not available")

        tool = dance.Dance()
        result = await tool(mock_deps, move="nonexistent_move")

        assert "error" in result
        assert "Unknown dance move" in result["error"]

    @pytest.mark.asyncio
    async def test_dance_queues_move(self, mock_deps: ToolDependencies) -> None:
        """Test dance queues the move."""
        from reachy_mini_conversation_app.tools import dance

        if not dance.DANCE_AVAILABLE:
            pytest.skip("Dance library not available")

        # Get a valid move name
        available_moves = list(dance.AVAILABLE_MOVES.keys())
        if not available_moves:
            pytest.skip("No moves available")

        move_name = available_moves[0]

        tool = dance.Dance()
        result = await tool(mock_deps, move=move_name)

        assert result["status"] == "queued"
        assert result["move"] == move_name
        assert result["repeat"] == 1
        mock_deps.movement_manager.queue_move.assert_called_once()

    @pytest.mark.asyncio
    async def test_dance_repeat_queues_multiple(self, mock_deps: ToolDependencies) -> None:
        """Test dance queues multiple moves with repeat."""
        from reachy_mini_conversation_app.tools import dance

        if not dance.DANCE_AVAILABLE:
            pytest.skip("Dance library not available")

        available_moves = list(dance.AVAILABLE_MOVES.keys())
        if not available_moves:
            pytest.skip("No moves available")

        move_name = available_moves[0]

        tool = dance.Dance()
        result = await tool(mock_deps, move=move_name, repeat=3)

        assert result["status"] == "queued"
        assert result["repeat"] == 3
        assert mock_deps.movement_manager.queue_move.call_count == 3

    @pytest.mark.asyncio
    async def test_dance_random_picks_move(self, mock_deps: ToolDependencies) -> None:
        """Test dance picks random move when move is 'random'."""
        from reachy_mini_conversation_app.tools import dance

        if not dance.DANCE_AVAILABLE:
            pytest.skip("Dance library not available")

        if not dance.AVAILABLE_MOVES:
            pytest.skip("No moves available")

        tool = dance.Dance()

        with patch("random.choice") as mock_choice:
            mock_choice.return_value = list(dance.AVAILABLE_MOVES.keys())[0]
            result = await tool(mock_deps, move="random")

        assert result["status"] == "queued"
        mock_choice.assert_called_once()

    @pytest.mark.asyncio
    async def test_dance_no_move_picks_random(self, mock_deps: ToolDependencies) -> None:
        """Test dance picks random move when no move specified."""
        from reachy_mini_conversation_app.tools import dance

        if not dance.DANCE_AVAILABLE:
            pytest.skip("Dance library not available")

        if not dance.AVAILABLE_MOVES:
            pytest.skip("No moves available")

        tool = dance.Dance()

        with patch("random.choice") as mock_choice:
            mock_choice.return_value = list(dance.AVAILABLE_MOVES.keys())[0]
            result = await tool(mock_deps)  # No move parameter

        assert result["status"] == "queued"
        mock_choice.assert_called_once()

    @pytest.mark.asyncio
    async def test_dance_default_repeat_is_one(self, mock_deps: ToolDependencies) -> None:
        """Test dance defaults to repeat=1."""
        from reachy_mini_conversation_app.tools import dance

        if not dance.DANCE_AVAILABLE:
            pytest.skip("Dance library not available")

        available_moves = list(dance.AVAILABLE_MOVES.keys())
        if not available_moves:
            pytest.skip("No moves available")

        tool = dance.Dance()
        result = await tool(mock_deps, move=available_moves[0])

        assert result["repeat"] == 1


class TestDanceAvailability:
    """Tests for dance library availability detection."""

    def test_dance_available_flag_exists(self) -> None:
        """Test DANCE_AVAILABLE flag exists."""
        from reachy_mini_conversation_app.tools import dance

        assert hasattr(dance, "DANCE_AVAILABLE")
        assert isinstance(dance.DANCE_AVAILABLE, bool)

    def test_available_moves_exists(self) -> None:
        """Test AVAILABLE_MOVES exists."""
        from reachy_mini_conversation_app.tools import dance

        assert hasattr(dance, "AVAILABLE_MOVES")
        # Could be dict or empty dict depending on library availability
        assert isinstance(dance.AVAILABLE_MOVES, dict)


class TestDanceImportFailure:
    """Tests for dance library import failure handling."""

    def test_import_failure_sets_dance_not_available(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test ImportError during dance library import sets DANCE_AVAILABLE to False."""
        import importlib
        import sys

        # Save original modules
        modules_to_remove = [
            "reachy_mini_conversation_app.tools.dance",
            "reachy_mini_dances_library.collection.dance",
            "reachy_mini_dances_library.collection",
            "reachy_mini_dances_library",
        ]
        saved_modules = {}
        for mod in modules_to_remove:
            if mod in sys.modules:
                saved_modules[mod] = sys.modules.pop(mod)

        try:
            # Create a mock that raises ImportError for dance library
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if "reachy_mini_dances_library" in name:
                    raise ImportError("Test import error for dance library")
                return original_import(name, *args, **kwargs)

            __builtins__["__import__"] = mock_import

            # Import the dance module fresh - this should trigger the except branch
            with caplog.at_level("WARNING"):
                dance_module = importlib.import_module("reachy_mini_conversation_app.tools.dance")

            # Verify the module handled the import error
            assert dance_module.DANCE_AVAILABLE is False
            assert dance_module.AVAILABLE_MOVES == {}
            assert "Dance library not available" in caplog.text

        finally:
            # Restore original import
            __builtins__["__import__"] = original_import

            # Restore original modules
            for mod in modules_to_remove:
                if mod in sys.modules:
                    del sys.modules[mod]
            for mod, module in saved_modules.items():
                sys.modules[mod] = module
