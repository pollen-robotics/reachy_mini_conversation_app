"""Pytest fixtures specific to tool testing."""

from typing import Any, Dict
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Tool Registry Isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_tool_registry(monkeypatch: pytest.MonkeyPatch) -> None:
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
    mock_background_task_manager: MagicMock,
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
        background_task_manager=mock_background_task_manager,
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
# Code Generation Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_for_code(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock Anthropic client for code generation tests."""
    mock_client = MagicMock()

    # Mock successful code generation response
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            text="""Here's the Python code:

```python
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
```

This function prints a greeting message."""
        )
    ]
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create = MagicMock(return_value=mock_response)

    # Patch the Anthropic client
    with patch("anthropic.Anthropic", return_value=mock_client):
        yield mock_client


@pytest.fixture
def temp_reachy_code_dir(tmp_path: Path) -> Path:
    """Create temporary directory for generated code."""
    code_dir = tmp_path / "reachy_code"
    code_dir.mkdir(parents=True)
    return code_dir


# ---------------------------------------------------------------------------
# GitHub Operation Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_subprocess_for_git(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock subprocess for git operations."""
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="Success",
        stderr="",
    )
    monkeypatch.setattr("subprocess.run", mock_run)
    return mock_run


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary directory that simulates a git repo."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir(parents=True)

    # Create .git directory to simulate git repo
    git_dir = repo_dir / ".git"
    git_dir.mkdir()

    # Create some test files
    (repo_dir / "README.md").write_text("# Test Repository")
    (repo_dir / "main.py").write_text("print('hello')")

    return repo_dir


@pytest.fixture
def mock_github_api(monkeypatch: pytest.MonkeyPatch, mock_github_client: MagicMock) -> MagicMock:
    """Mock PyGithub for API operations."""
    with patch("github.Github", return_value=mock_github_client):
        yield mock_github_client


# ---------------------------------------------------------------------------
# Code Execution Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_subprocess_for_code_exec(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock subprocess for code execution tests."""
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="Code executed successfully\nOutput: 42",
        stderr="",
    )
    monkeypatch.setattr("subprocess.run", mock_run)
    return mock_run


@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    """Create a sample Python file for execution tests."""
    code_file = tmp_path / "test_script.py"
    code_file.write_text(
        """#!/usr/bin/env python3
import sys

def main():
    print("Hello from test script!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
    )
    return code_file


# ---------------------------------------------------------------------------
# Background Task Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_asyncio_task() -> MagicMock:
    """Create a mock asyncio.Task."""
    mock_task = MagicMock()
    mock_task.done = MagicMock(return_value=False)
    mock_task.cancel = MagicMock()
    mock_task.cancelled = MagicMock(return_value=False)
    return mock_task


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
