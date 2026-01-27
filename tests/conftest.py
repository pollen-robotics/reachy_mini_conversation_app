"""Pytest configuration and shared fixtures for all tests."""

import sys
import asyncio
from typing import Any, Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


# Ensure src is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ---------------------------------------------------------------------------
# ReachyMini Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_reachy_mini() -> MagicMock:
    """Create a mock ReachyMini robot instance."""
    mock = MagicMock()

    # Mock head component
    mock.head = MagicMock()
    mock.head.forward_kinematics = MagicMock(return_value=np.eye(4))
    mock.head.joints = MagicMock()
    mock.head.joints.positions = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    # Mock mobile base
    mock.mobile_base = MagicMock()
    mock.mobile_base.is_moving = False

    # Mock media for camera
    mock.media = MagicMock()
    # Create a dummy BGR frame (480x640x3)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock.media.get_frame = MagicMock(return_value=dummy_frame)

    # Mock look_at_image method
    mock.look_at_image = MagicMock(return_value=np.eye(4))

    # Mock send_head_pose
    mock.send_head_pose = MagicMock()

    return mock


@pytest.fixture
def mock_movement_manager() -> MagicMock:
    """Create a mock MovementManager."""
    mock = MagicMock()
    mock.set_base_pose = MagicMock()
    mock.get_current_pose = MagicMock(return_value=np.eye(4))
    mock.is_dancing = False
    mock.is_emotion_playing = False
    mock.stop_dance = MagicMock()
    mock.stop_emotion = MagicMock()
    mock.play_dance = AsyncMock()
    mock.play_emotion = AsyncMock()
    mock.set_speech_offsets = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Tool Dependencies Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tool_deps(mock_reachy_mini: MagicMock, mock_movement_manager: MagicMock) -> Any:
    """Create mock ToolDependencies for testing tools."""
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

    deps = ToolDependencies(
        reachy_mini=mock_reachy_mini,
        movement_manager=mock_movement_manager,
        camera_worker=None,
        vision_manager=None,
        head_wobbler=None,
        motion_duration_s=1.0,
        background_task_manager=None,
    )
    return deps


@pytest.fixture
def mock_tool_deps_with_camera(
    mock_reachy_mini: MagicMock,
    mock_movement_manager: MagicMock,
    mock_camera_worker: MagicMock,
) -> Any:
    """Create mock ToolDependencies with camera worker."""
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

    deps = ToolDependencies(
        reachy_mini=mock_reachy_mini,
        movement_manager=mock_movement_manager,
        camera_worker=mock_camera_worker,
        vision_manager=None,
        head_wobbler=None,
        motion_duration_s=1.0,
        background_task_manager=None,
    )
    return deps


# ---------------------------------------------------------------------------
# Camera Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_camera_worker() -> MagicMock:
    """Create a mock CameraWorker."""
    mock = MagicMock()

    # Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock.get_latest_frame = MagicMock(return_value=dummy_frame)
    mock.get_face_tracking_offsets = MagicMock(return_value=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    mock.set_head_tracking_enabled = MagicMock()
    mock.is_head_tracking_enabled = True
    mock.start = MagicMock()
    mock.stop = MagicMock()

    return mock


@pytest.fixture
def mock_head_wobbler() -> MagicMock:
    """Create a mock HeadWobbler."""
    mock = MagicMock()
    mock.feed = MagicMock()
    mock.start = MagicMock()
    mock.stop = MagicMock()
    mock.reset = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Background Task Manager Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_background_task_manager() -> MagicMock:
    """Create a mock BackgroundTaskManager."""
    mock = MagicMock()
    mock.start_task = AsyncMock()
    mock.cancel_task = AsyncMock(return_value=True)
    mock.get_task = MagicMock(return_value=None)
    mock.get_running_tasks = MagicMock(return_value=[])
    mock.get_all_tasks = MagicMock(return_value=[])
    mock.get_status_summary = MagicMock(
        return_value={
            "total": 0,
            "counts": {"pending": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0},
            "running": [],
        }
    )
    return mock


@pytest.fixture
def background_task_manager() -> Generator[Any, None, None]:
    """Create a real BackgroundTaskManager instance (reset after test)."""
    from reachy_mini_conversation_app.background_tasks import BackgroundTaskManager

    # Reset singleton before test
    BackgroundTaskManager.reset_instance()
    manager = BackgroundTaskManager.get_instance()
    yield manager
    # Reset singleton after test
    BackgroundTaskManager.reset_instance()


# ---------------------------------------------------------------------------
# Async Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
