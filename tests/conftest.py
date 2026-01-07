"""Pytest configuration and shared fixtures for all tests."""

import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, AsyncMock

import pytest
import numpy as np
from numpy.typing import NDArray


# Ensure src is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ---------------------------------------------------------------------------
# Environment Variables Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Provide a clean environment without app-specific env vars."""
    env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_DEFAULT_OWNER",
        "GITHUB_OWNER_EMAIL",
        "MODEL_NAME",
        "HF_HOME",
        "HF_TOKEN",
        "LOCAL_VISION_MODEL",
        "REACHY_MINI_CUSTOM_PROFILE",
        "ANTHROPIC_MODEL",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    """Provide mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "sk-test-openai-key",
        "ANTHROPIC_API_KEY": "sk-ant-test-key",
        "GITHUB_TOKEN": "ghp_test_token",
        "GITHUB_DEFAULT_OWNER": "test-owner",
        "GITHUB_OWNER_EMAIL": "test@example.com",
        "MODEL_NAME": "gpt-realtime-test",
        "HF_HOME": "/tmp/hf_cache",
        "HF_TOKEN": "hf_test_token",
        "REACHY_MINI_CUSTOM_PROFILE": "default",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


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
def sample_frame() -> NDArray[np.uint8]:
    """Create a sample BGR frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_base64(sample_frame: NDArray[np.uint8]) -> str:
    """Create a base64 encoded sample frame."""
    import base64
    import cv2

    _, buffer = cv2.imencode(".jpg", sample_frame)
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# Audio Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audio_pcm16() -> NDArray[np.int16]:
    """Create sample PCM16 audio data (1 second at 24kHz)."""
    sample_rate = 24000
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Generate 440Hz sine wave
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    return (wave * 32767).astype(np.int16)


@pytest.fixture
def sample_audio_base64(sample_audio_pcm16: NDArray[np.int16]) -> str:
    """Create base64 encoded audio data."""
    import base64

    return base64.b64encode(sample_audio_pcm16.tobytes()).decode("ascii")


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
# OpenAI Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock AsyncOpenAI client."""
    mock = MagicMock()

    # Mock realtime connection
    mock_connection = AsyncMock()
    mock_connection.session = AsyncMock()
    mock_connection.session.update = AsyncMock()
    mock_connection.input_audio_buffer = AsyncMock()
    mock_connection.input_audio_buffer.append = AsyncMock()
    mock_connection.conversation = MagicMock()
    mock_connection.conversation.item = MagicMock()
    mock_connection.conversation.item.create = AsyncMock()
    mock_connection.response = AsyncMock()
    mock_connection.response.create = AsyncMock()
    mock_connection.response.cancel = AsyncMock()
    mock_connection.close = AsyncMock()

    # Make connection usable as async context manager
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock(return_value=False)

    # Make connection iterable (no events)
    mock_connection.__aiter__ = MagicMock(return_value=iter([]))

    mock.realtime = MagicMock()
    mock.realtime.connect = MagicMock(return_value=mock_connection)

    return mock


# ---------------------------------------------------------------------------
# Anthropic Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Create a mock Anthropic client."""
    mock = MagicMock()

    # Mock messages.create response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Generated code response")]
    mock_response.stop_reason = "end_turn"

    mock.messages = MagicMock()
    mock.messages.create = MagicMock(return_value=mock_response)

    return mock


# ---------------------------------------------------------------------------
# GitHub Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_github_client() -> MagicMock:
    """Create a mock PyGithub client."""
    mock = MagicMock()

    # Mock user
    mock_user = MagicMock()
    mock_user.login = "test-user"
    mock_user.name = "Test User"
    mock.get_user = MagicMock(return_value=mock_user)

    # Mock repo
    mock_repo = MagicMock()
    mock_repo.full_name = "test-owner/test-repo"
    mock_repo.default_branch = "main"
    mock_repo.clone_url = "https://github.com/test-owner/test-repo.git"
    mock.get_repo = MagicMock(return_value=mock_repo)

    # Mock issue
    mock_issue = MagicMock()
    mock_issue.number = 1
    mock_issue.title = "Test Issue"
    mock_issue.html_url = "https://github.com/test-owner/test-repo/issues/1"
    mock_repo.create_issue = MagicMock(return_value=mock_issue)
    mock_repo.get_issue = MagicMock(return_value=mock_issue)

    # Mock PR
    mock_pr = MagicMock()
    mock_pr.number = 42
    mock_pr.title = "Test PR"
    mock_pr.html_url = "https://github.com/test-owner/test-repo/pull/42"
    mock_repo.create_pull = MagicMock(return_value=mock_pr)
    mock_repo.get_pull = MagicMock(return_value=mock_pr)

    return mock


@pytest.fixture
def mock_git_repo(tmp_path: Path) -> MagicMock:
    """Create a mock Git repository."""
    mock = MagicMock()
    mock.working_dir = str(tmp_path)
    mock.git = MagicMock()
    mock.git.status = MagicMock(return_value="On branch main\nnothing to commit")
    mock.git.diff = MagicMock(return_value="")
    mock.git.log = MagicMock(return_value="abc1234 Initial commit")
    mock.git.add = MagicMock()
    mock.git.commit = MagicMock()
    mock.git.push = MagicMock()
    mock.git.pull = MagicMock()
    mock.git.clone = MagicMock()
    mock.remotes = MagicMock()
    mock.remotes.origin = MagicMock()
    mock.remotes.origin.url = "https://github.com/test-owner/test-repo.git"
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
# Temporary File/Directory Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_profile_dir(tmp_path: Path) -> Path:
    """Create a temporary profile directory structure."""
    profile_dir = tmp_path / "profiles" / "test_profile"
    profile_dir.mkdir(parents=True)

    # Create minimal profile files
    (profile_dir / "instructions.txt").write_text("Test instructions for profile.")
    (profile_dir / "tools.txt").write_text("do_nothing\n")
    (profile_dir / "voice.txt").write_text("ash\n")

    return profile_dir


@pytest.fixture
def temp_prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts library directory."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    # Create sample prompt files
    (prompts_dir / "greeting.txt").write_text("Hello, I am Reachy!")
    (prompts_dir / "farewell.txt").write_text("Goodbye!")

    return prompts_dir


@pytest.fixture
def temp_code_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for generated code."""
    code_dir = tmp_path / "reachy_code"
    code_dir.mkdir(parents=True)
    return code_dir


# ---------------------------------------------------------------------------
# Async Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
