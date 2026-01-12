"""Pytest fixtures specific to tool testing."""

from typing import Any, Dict, Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tool Registry Isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_tool_registry(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Isolate tool registry to prevent cross-test contamination.

    This fixture resets the tool registry before and after each test,
    ensuring tools loaded in one test don't affect others.
    """
    from reachy_mini_conversation_app.tools import core_tools

    # Save original state
    original_tools = core_tools.ALL_TOOLS.copy()
    original_specs = core_tools.ALL_TOOL_SPECS.copy()
    original_initialized = core_tools._TOOLS_INITIALIZED

    # Reset for test
    core_tools.ALL_TOOLS = {}
    core_tools.ALL_TOOL_SPECS = []
    core_tools._TOOLS_INITIALIZED = False

    yield

    # Restore original state
    core_tools.ALL_TOOLS = original_tools
    core_tools.ALL_TOOL_SPECS = original_specs
    core_tools._TOOLS_INITIALIZED = original_initialized


# ---------------------------------------------------------------------------
# Tool Dependencies with All Components
# ---------------------------------------------------------------------------


@pytest.fixture
def full_mock_tool_deps(
    mock_reachy_mini: MagicMock,
    mock_movement_manager: MagicMock,
    mock_camera_worker: MagicMock,
    mock_head_wobbler: MagicMock,
) -> Any:
    """Create fully mocked ToolDependencies with all components."""
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

    # Mock vision manager
    mock_vision_manager = MagicMock()
    mock_vision_manager.get_description = MagicMock(return_value="A test scene description")

    deps = ToolDependencies(
        reachy_mini=mock_reachy_mini,
        movement_manager=mock_movement_manager,
        camera_worker=mock_camera_worker,
        vision_manager=mock_vision_manager,
        head_wobbler=mock_head_wobbler,
        motion_duration_s=1.0,
    )
    return deps


# ---------------------------------------------------------------------------
# Dance Library Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dance_library(monkeypatch: pytest.MonkeyPatch) -> Dict[str, MagicMock]:
    """Mock the dance library for testing dance tools."""
    available_moves = {
        "simple_nod": MagicMock(),
        "head_tilt_roll": MagicMock(),
        "side_to_side_sway": MagicMock(),
        "dizzy_spin": MagicMock(),
    }

    # Patch at module level
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tools.dance.AVAILABLE_MOVES",
        available_moves,
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tools.dance.DANCE_AVAILABLE",
        True,
    )

    return available_moves


@pytest.fixture
def mock_dance_queue_move(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock DanceQueueMove class."""
    mock_class = MagicMock()
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tools.dance.DanceQueueMove",
        mock_class,
    )
    return mock_class


# ---------------------------------------------------------------------------
# Emotion Library Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_emotion_library(monkeypatch: pytest.MonkeyPatch) -> Dict[str, MagicMock]:
    """Mock the emotion library for testing emotion tools."""
    available_emotions = {
        "happy": MagicMock(),
        "sad": MagicMock(),
        "surprised": MagicMock(),
        "thinking": MagicMock(),
    }

    # These would be patched in play_emotion.py
    return available_emotions


# ---------------------------------------------------------------------------
# Camera Tool Specific Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cv2_imencode(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock cv2.imencode for camera tests."""
    mock_encode = MagicMock()
    # Return success and fake encoded buffer
    mock_encode.return_value = (True, np.array([1, 2, 3, 4], dtype=np.uint8))
    monkeypatch.setattr("cv2.imencode", mock_encode)
    return mock_encode


# ---------------------------------------------------------------------------
# Tool Instance Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def do_nothing_tool() -> Any:
    """Create a DoNothing tool instance."""
    from reachy_mini_conversation_app.tools.do_nothing import DoNothing

    return DoNothing()


@pytest.fixture
def dance_tool() -> Any:
    """Create a Dance tool instance."""
    from reachy_mini_conversation_app.tools.dance import Dance

    return Dance()


@pytest.fixture
def stop_dance_tool() -> Any:
    """Create a StopDance tool instance."""
    from reachy_mini_conversation_app.tools.stop_dance import StopDance

    return StopDance()


@pytest.fixture
def move_head_tool() -> Any:
    """Create a MoveHead tool instance."""
    from reachy_mini_conversation_app.tools.move_head import MoveHead

    return MoveHead()


@pytest.fixture
def camera_tool() -> Any:
    """Create a Camera tool instance."""
    from reachy_mini_conversation_app.tools.camera import Camera

    return Camera()


@pytest.fixture
def head_tracking_tool() -> Any:
    """Create a HeadTracking tool instance."""
    from reachy_mini_conversation_app.tools.head_tracking import HeadTracking

    return HeadTracking()
