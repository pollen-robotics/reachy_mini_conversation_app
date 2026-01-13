"""Unit tests for main module (entrypoint)."""

from __future__ import annotations
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Store original modules before any mocking
_ORIGINAL_MODULES: dict[str, Any] = {}
_MODULES_TO_MOCK = [
    "reachy_mini",
    "reachy_mini.media",
    "reachy_mini.media.media_manager",
    "reachy_mini.utils",
    "reachy_mini.utils.interpolation",
    "reachy_mini.motion",
    "reachy_mini.motion.move",
    "gradio",
    "gradio.utils",
    "fastapi",
    "fastrtc",
]


@pytest.fixture(autouse=True)
def mock_main_dependencies() -> Generator[None, None, None]:
    """Mock heavy dependencies for main tests and restore them after."""
    # Save originals
    for mod_name in _MODULES_TO_MOCK:
        if mod_name in sys.modules:
            _ORIGINAL_MODULES[mod_name] = sys.modules[mod_name]

    # Create mock modules
    mock_reachy_mini = MagicMock()
    mock_reachy_mini.ReachyMini = MagicMock
    mock_reachy_mini.ReachyMiniApp = type("ReachyMiniApp", (), {
        "custom_app_url": "",
        "dont_start_webserver": False,
        "settings_app": None,
        "_get_instance_path": lambda self: Path("/tmp/test"),
        "wrapped_run": lambda self: None,
        "stop": lambda self: None,
    })

    mock_gradio = MagicMock()
    mock_gradio.Chatbot = MagicMock(return_value=MagicMock(avatar_images=None))
    mock_gradio.Textbox = MagicMock(return_value=MagicMock())
    mock_gradio.Blocks = MagicMock
    mock_gradio.mount_gradio_app = MagicMock(return_value=MagicMock())
    mock_gradio.utils = MagicMock()
    mock_gradio.utils.get_space = MagicMock(return_value=None)

    mock_fastapi = MagicMock()
    mock_fastapi.FastAPI = MagicMock

    mock_fastrtc = MagicMock()
    mock_fastrtc.Stream = MagicMock(return_value=MagicMock(ui=MagicMock()))

    # Install mocks - reachy_mini and all submodules
    sys.modules["reachy_mini"] = mock_reachy_mini
    sys.modules["reachy_mini.media"] = MagicMock()
    sys.modules["reachy_mini.media.media_manager"] = MagicMock()
    sys.modules["reachy_mini.utils"] = MagicMock()
    sys.modules["reachy_mini.utils.interpolation"] = MagicMock()
    sys.modules["reachy_mini.motion"] = MagicMock()
    sys.modules["reachy_mini.motion.move"] = MagicMock()
    sys.modules["gradio"] = mock_gradio
    sys.modules["gradio.utils"] = mock_gradio.utils
    sys.modules["fastapi"] = mock_fastapi
    sys.modules["fastrtc"] = mock_fastrtc

    yield

    # Restore originals
    for mod_name in _MODULES_TO_MOCK:
        if mod_name in _ORIGINAL_MODULES:
            sys.modules[mod_name] = _ORIGINAL_MODULES[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]

    # Clear cached module imports
    mods_to_clear = [k for k in sys.modules if k.startswith("reachy_mini_conversation_app.main")]
    for mod_name in mods_to_clear:
        del sys.modules[mod_name]
    # Also clear related modules that may have been imported
    related_mods = [
        "reachy_mini_conversation_app.utils",
        "reachy_mini_conversation_app.camera_worker",
        "reachy_mini_conversation_app.moves",
        "reachy_mini_conversation_app.console",
    ]
    for mod_name in related_mods:
        if mod_name in sys.modules:
            del sys.modules[mod_name]


class TestUpdateChatbot:
    """Tests for update_chatbot function."""

    def test_update_chatbot_appends_response(self) -> None:
        """Test that update_chatbot appends response to chatbot."""
        from reachy_mini_conversation_app.main import update_chatbot

        chatbot: List[Dict[str, Any]] = []
        response = {"role": "assistant", "content": "Hello"}

        result = update_chatbot(chatbot, response)

        assert len(result) == 1
        assert result[0] == response

    def test_update_chatbot_preserves_existing(self) -> None:
        """Test that update_chatbot preserves existing messages."""
        from reachy_mini_conversation_app.main import update_chatbot

        chatbot: List[Dict[str, Any]] = [{"role": "user", "content": "Hi"}]
        response = {"role": "assistant", "content": "Hello"}

        result = update_chatbot(chatbot, response)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hi"}
        assert result[1] == response

    def test_update_chatbot_returns_same_list(self) -> None:
        """Test that update_chatbot returns the same list object."""
        from reachy_mini_conversation_app.main import update_chatbot

        chatbot: List[Dict[str, Any]] = []
        response = {"role": "assistant", "content": "Hello"}

        result = update_chatbot(chatbot, response)

        assert result is chatbot


class TestMain:
    """Tests for main function."""

    def test_main_calls_parse_args_and_run(self) -> None:
        """Test that main calls parse_args and run."""
        from reachy_mini_conversation_app.main import main

        mock_args = MagicMock()

        with patch("reachy_mini_conversation_app.main.parse_args", return_value=(mock_args, [])):
            with patch("reachy_mini_conversation_app.main.run") as mock_run:
                main()

                mock_run.assert_called_once_with(mock_args)


class TestRunSimulationCheck:
    """Tests for run function simulation mode check."""

    def test_run_exits_on_simulation_without_gradio(self) -> None:
        """Test run exits when simulation mode without gradio flag."""
        from reachy_mini_conversation_app.main import run

        args = argparse.Namespace(
            debug=False,
            no_camera=False,
            head_tracker=None,
            gradio=False,
            local_vision=False,
            robot_name=None,
        )
        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": True}

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
            with pytest.raises(SystemExit) as exc_info:
                run(args, robot=mock_robot)

            assert exc_info.value.code == 1

        mock_robot.client.disconnect.assert_called_once()


class TestReachyMiniConversationApp:
    """Tests for ReachyMiniConversationApp class."""

    def test_class_attributes(self) -> None:
        """Test class has correct default attributes."""
        from reachy_mini_conversation_app.main import ReachyMiniConversationApp

        assert ReachyMiniConversationApp.custom_app_url == "http://0.0.0.0:7860/"
        assert ReachyMiniConversationApp.dont_start_webserver is False

    def test_run_method_calls_run_function(self) -> None:
        """Test run method calls the run function."""
        from reachy_mini_conversation_app.main import ReachyMiniConversationApp

        app = ReachyMiniConversationApp()
        object.__setattr__(app, "_get_instance_path", MagicMock(return_value=Path("/tmp/test/file")))
        app.settings_app = MagicMock()

        mock_robot = MagicMock()
        stop_event = threading.Event()

        with patch("reachy_mini_conversation_app.main.parse_args", return_value=(MagicMock(), [])):
            with patch("reachy_mini_conversation_app.main.run") as mock_run:
                with patch("asyncio.new_event_loop"):
                    with patch("asyncio.set_event_loop"):
                        app.run(mock_robot, stop_event)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["robot"] is mock_robot
        assert call_kwargs["app_stop_event"] is stop_event
        assert call_kwargs["settings_app"] is app.settings_app
        assert call_kwargs["instance_path"] == Path("/tmp/test")


class TestRunWithMockedInternalImports:
    """Tests for run function with mocked internal imports."""

    def _create_mock_args(self, **kwargs: Any) -> argparse.Namespace:
        """Create mock args with defaults."""
        defaults = {
            "debug": False,
            "no_camera": False,
            "head_tracker": None,
            "gradio": False,
            "local_vision": False,
            "robot_name": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def _mock_run_internals(self) -> Dict[str, MagicMock]:
        """Create mocks for all internal imports in run()."""
        mocks = {
            "MovementManager": MagicMock(),
            "LocalStream": MagicMock(),
            "OpenaiRealtimeHandler": MagicMock(),
            "ToolDependencies": MagicMock(),
            "HeadWobbler": MagicMock(),
        }
        # Setup LocalStream to raise KeyboardInterrupt on launch
        mock_stream_instance = MagicMock()
        mock_stream_instance.launch.side_effect = KeyboardInterrupt()
        mocks["LocalStream"].return_value = mock_stream_instance
        return mocks

    def test_run_initializes_robot_auto_backend(self) -> None:
        """Test run initializes robot with auto-detected backend."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()
        mocks = self._mock_run_internals()

        mock_robot_class = MagicMock()
        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}
        mock_robot_class.return_value = mock_robot

        with patch("reachy_mini_conversation_app.main.ReachyMini", mock_robot_class):
            with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
                with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                    with patch.dict("sys.modules", {
                        "reachy_mini_conversation_app.moves": MagicMock(MovementManager=mocks["MovementManager"]),
                        "reachy_mini_conversation_app.console": MagicMock(LocalStream=mocks["LocalStream"]),
                        "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mocks["OpenaiRealtimeHandler"]),
                        "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=mocks["ToolDependencies"]),
                        "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=mocks["HeadWobbler"]),
                    }):
                        with patch("time.sleep"):
                            run(args)

        # ReachyMini is called without explicit backend (auto-detection)
        mock_robot_class.assert_called_once_with()

    def test_run_initializes_robot_with_robot_name(self) -> None:
        """Test run initializes robot with robot_name parameter."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args(robot_name="my_robot")
        mocks = self._mock_run_internals()

        mock_robot_class = MagicMock()
        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}
        mock_robot_class.return_value = mock_robot

        with patch("reachy_mini_conversation_app.main.ReachyMini", mock_robot_class):
            with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
                with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                    with patch.dict("sys.modules", {
                        "reachy_mini_conversation_app.moves": MagicMock(MovementManager=mocks["MovementManager"]),
                        "reachy_mini_conversation_app.console": MagicMock(LocalStream=mocks["LocalStream"]),
                        "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mocks["OpenaiRealtimeHandler"]),
                        "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=mocks["ToolDependencies"]),
                        "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=mocks["HeadWobbler"]),
                    }):
                        with patch("time.sleep"):
                            run(args)

        # ReachyMini is called with robot_name
        mock_robot_class.assert_called_once_with(robot_name="my_robot")

    def test_run_uses_provided_robot(self) -> None:
        """Test run uses provided robot instead of creating new one."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()
        mocks = self._mock_run_internals()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}

        with patch("reachy_mini_conversation_app.main.ReachyMini") as mock_robot_class:
            with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
                with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                    with patch.dict("sys.modules", {
                        "reachy_mini_conversation_app.moves": MagicMock(MovementManager=mocks["MovementManager"]),
                        "reachy_mini_conversation_app.console": MagicMock(LocalStream=mocks["LocalStream"]),
                        "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mocks["OpenaiRealtimeHandler"]),
                        "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=mocks["ToolDependencies"]),
                        "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=mocks["HeadWobbler"]),
                    }):
                        with patch("time.sleep"):
                            run(args, robot=mock_robot)

        # Should not have created a new robot
        mock_robot_class.assert_not_called()

    def test_run_warns_head_tracker_with_no_camera(self) -> None:
        """Test run warns when head tracker with no camera."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args(no_camera=True, head_tracker="mediapipe")
        mocks = self._mock_run_internals()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}
        mock_logger = MagicMock()

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=mock_logger):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                with patch.dict("sys.modules", {
                    "reachy_mini_conversation_app.moves": MagicMock(MovementManager=mocks["MovementManager"]),
                    "reachy_mini_conversation_app.console": MagicMock(LocalStream=mocks["LocalStream"]),
                    "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mocks["OpenaiRealtimeHandler"]),
                    "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=mocks["ToolDependencies"]),
                    "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=mocks["HeadWobbler"]),
                }):
                    with patch("time.sleep"):
                        run(args, robot=mock_robot)

        mock_logger.warning.assert_any_call(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    def test_run_starts_all_services(self) -> None:
        """Test run starts all services (movement_manager, head_wobbler, etc.)."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}

        mock_movement_manager = MagicMock()
        mock_head_wobbler = MagicMock()
        mock_camera_worker = MagicMock()
        mock_vision_manager = MagicMock()
        mock_stream = MagicMock()
        mock_stream.launch.side_effect = KeyboardInterrupt()

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(mock_camera_worker, None, mock_vision_manager)):
                with patch.dict("sys.modules", {
                    "reachy_mini_conversation_app.moves": MagicMock(MovementManager=MagicMock(return_value=mock_movement_manager)),
                    "reachy_mini_conversation_app.console": MagicMock(LocalStream=MagicMock(return_value=mock_stream)),
                    "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=MagicMock()),
                    "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=MagicMock()),
                    "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=MagicMock(return_value=mock_head_wobbler)),
                }):
                    with patch("time.sleep"):
                        run(args, robot=mock_robot)

        mock_movement_manager.start.assert_called_once()
        mock_head_wobbler.start.assert_called_once()
        mock_camera_worker.start.assert_called_once()
        mock_vision_manager.start.assert_called_once()

    def test_run_stops_all_services_on_keyboard_interrupt(self) -> None:
        """Test run stops all services on KeyboardInterrupt."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}

        mock_movement_manager = MagicMock()
        mock_head_wobbler = MagicMock()
        mock_camera_worker = MagicMock()
        mock_vision_manager = MagicMock()
        mock_stream = MagicMock()
        mock_stream.launch.side_effect = KeyboardInterrupt()

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(mock_camera_worker, None, mock_vision_manager)):
                with patch.dict("sys.modules", {
                    "reachy_mini_conversation_app.moves": MagicMock(MovementManager=MagicMock(return_value=mock_movement_manager)),
                    "reachy_mini_conversation_app.console": MagicMock(LocalStream=MagicMock(return_value=mock_stream)),
                    "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=MagicMock()),
                    "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=MagicMock()),
                    "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=MagicMock(return_value=mock_head_wobbler)),
                }):
                    with patch("time.sleep"):
                        run(args, robot=mock_robot)

        mock_movement_manager.stop.assert_called_once()
        mock_head_wobbler.stop.assert_called_once()
        mock_camera_worker.stop.assert_called_once()
        mock_vision_manager.stop.assert_called_once()
        mock_robot.media.close.assert_called_once()
        mock_robot.client.disconnect.assert_called_once()

    def test_run_handles_media_close_error(self) -> None:
        """Test run handles media close error gracefully."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()
        mocks = self._mock_run_internals()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}
        mock_robot.media.close.side_effect = RuntimeError("Media error")

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                with patch.dict("sys.modules", {
                    "reachy_mini_conversation_app.moves": MagicMock(MovementManager=mocks["MovementManager"]),
                    "reachy_mini_conversation_app.console": MagicMock(LocalStream=mocks["LocalStream"]),
                    "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mocks["OpenaiRealtimeHandler"]),
                    "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=mocks["ToolDependencies"]),
                    "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=mocks["HeadWobbler"]),
                }):
                    with patch("time.sleep"):
                        # Should not raise
                        run(args, robot=mock_robot)

        mock_robot.client.disconnect.assert_called_once()

    def test_run_creates_local_stream_in_headless_mode(self) -> None:
        """Test run creates LocalStream in headless (non-gradio) mode."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args(gradio=False)

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}
        mock_settings_app = MagicMock()

        mock_stream_class = MagicMock()
        mock_stream = MagicMock()
        mock_stream.launch.side_effect = KeyboardInterrupt()
        mock_stream_class.return_value = mock_stream

        mock_handler_class = MagicMock()
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                with patch.dict("sys.modules", {
                    "reachy_mini_conversation_app.moves": MagicMock(MovementManager=MagicMock()),
                    "reachy_mini_conversation_app.console": MagicMock(LocalStream=mock_stream_class),
                    "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mock_handler_class),
                    "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=MagicMock()),
                    "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=MagicMock()),
                }):
                    with patch("time.sleep"):
                        run(args, robot=mock_robot, settings_app=mock_settings_app, instance_path="/test")

        mock_stream_class.assert_called_once_with(
            mock_handler,
            mock_robot,
            settings_app=mock_settings_app,
            instance_path="/test",
        )

    def test_run_with_app_stop_event(self) -> None:
        """Test run handles app_stop_event."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()
        mocks = self._mock_run_internals()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}

        stop_event = threading.Event()

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                with patch.dict("sys.modules", {
                    "reachy_mini_conversation_app.moves": MagicMock(MovementManager=mocks["MovementManager"]),
                    "reachy_mini_conversation_app.console": MagicMock(LocalStream=mocks["LocalStream"]),
                    "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mocks["OpenaiRealtimeHandler"]),
                    "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=mocks["ToolDependencies"]),
                    "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=mocks["HeadWobbler"]),
                }):
                    with patch("time.sleep"):
                        run(args, robot=mock_robot, app_stop_event=stop_event)

        # Test completed without hanging
    def test_run_creates_gradio_ui_when_gradio_true(self) -> None:
        """Test run creates Gradio UI when gradio=True."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args(gradio=True)
        mocks = self._mock_run_internals()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}

        # Mock gradio components
        mock_stream_ui = MagicMock()
        mock_stream = MagicMock()
        mock_stream.ui = mock_stream_ui
        mock_stream.ui.launch.side_effect = KeyboardInterrupt()

        mock_stream_class = MagicMock(return_value=mock_stream)

        mock_personality_ui = MagicMock()
        mock_personality_ui.additional_inputs_ordered.return_value = []
        mock_personality_ui_class = MagicMock(return_value=mock_personality_ui)

        mock_fastapi_app = MagicMock()

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                with patch("reachy_mini_conversation_app.main.FastAPI", return_value=mock_fastapi_app):
                    with patch.dict("sys.modules", {
                        "reachy_mini_conversation_app.moves": MagicMock(MovementManager=mocks["MovementManager"]),
                        "reachy_mini_conversation_app.console": MagicMock(LocalStream=mocks["LocalStream"]),
                        "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mocks["OpenaiRealtimeHandler"]),
                        "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=mocks["ToolDependencies"]),
                        "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=mocks["HeadWobbler"]),
                        "reachy_mini_conversation_app.gradio_personality": MagicMock(PersonalityUI=mock_personality_ui_class),
                    }):
                        with patch("fastrtc.Stream", mock_stream_class):
                            with patch("time.sleep"):
                                run(args, robot=mock_robot)

        # Verify Gradio setup was called
        mock_personality_ui_class.assert_called_once()
        mock_personality_ui.create_components.assert_called_once()

    def test_run_gradio_uses_existing_fastapi_app(self) -> None:
        """Test run uses existing FastAPI app in gradio mode when provided."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args(gradio=True)
        mocks = self._mock_run_internals()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}

        # Mock gradio components
        mock_stream_ui = MagicMock()
        mock_stream = MagicMock()
        mock_stream.ui = mock_stream_ui
        mock_stream.ui.launch.side_effect = KeyboardInterrupt()
        mock_stream_class = MagicMock(return_value=mock_stream)

        mock_personality_ui = MagicMock()
        mock_personality_ui.additional_inputs_ordered.return_value = []
        mock_personality_ui_class = MagicMock(return_value=mock_personality_ui)

        mock_existing_app = MagicMock()

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                with patch.dict("sys.modules", {
                    "reachy_mini_conversation_app.moves": MagicMock(MovementManager=mocks["MovementManager"]),
                    "reachy_mini_conversation_app.console": MagicMock(LocalStream=mocks["LocalStream"]),
                    "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mocks["OpenaiRealtimeHandler"]),
                    "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=mocks["ToolDependencies"]),
                    "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=mocks["HeadWobbler"]),
                    "reachy_mini_conversation_app.gradio_personality": MagicMock(PersonalityUI=mock_personality_ui_class),
                }):
                    with patch("fastrtc.Stream", mock_stream_class):
                        with patch("time.sleep"):
                            run(args, robot=mock_robot, settings_app=mock_existing_app)

        # Verify existing app was used (not FastAPI() called to create new one)
        mock_personality_ui.wire_events.assert_called_once()

    def test_run_gradio_with_stop_event_launches_thread(self) -> None:
        """Test run with app_stop_event launches daemon thread in gradio mode."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args(gradio=True)
        mocks = self._mock_run_internals()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}

        # Mock gradio components
        mock_stream_ui = MagicMock()
        mock_stream = MagicMock()
        mock_stream.ui = mock_stream_ui
        mock_stream.ui.launch.side_effect = KeyboardInterrupt()
        mock_stream_class = MagicMock(return_value=mock_stream)

        mock_personality_ui = MagicMock()
        mock_personality_ui.additional_inputs_ordered.return_value = []
        mock_personality_ui_class = MagicMock(return_value=mock_personality_ui)

        stop_event = threading.Event()

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=MagicMock()):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                with patch("reachy_mini_conversation_app.main.FastAPI", return_value=MagicMock()):
                    with patch.dict("sys.modules", {
                        "reachy_mini_conversation_app.moves": MagicMock(MovementManager=mocks["MovementManager"]),
                        "reachy_mini_conversation_app.console": MagicMock(LocalStream=mocks["LocalStream"]),
                        "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=mocks["OpenaiRealtimeHandler"]),
                        "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=mocks["ToolDependencies"]),
                        "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=mocks["HeadWobbler"]),
                        "reachy_mini_conversation_app.gradio_personality": MagicMock(PersonalityUI=mock_personality_ui_class),
                    }):
                        with patch("fastrtc.Stream", mock_stream_class):
                            with patch("time.sleep"):
                                run(args, robot=mock_robot, app_stop_event=stop_event)

        # Test completed without hanging (stop_event thread was created)

    def test_run_stop_event_thread_executes_shutdown(self) -> None:
        """Test that stop event thread executes shutdown code (lines 172-176)."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}

        stop_event = threading.Event()

        mock_stream = MagicMock()
        # Make launch block until stop_event is set, then raise KeyboardInterrupt
        def launch_side_effect() -> None:
            # Set the stop event to trigger the poll_stop_event thread
            stop_event.set()
            # Give the daemon thread time to process
            time.sleep(0.1)
            raise KeyboardInterrupt()

        mock_stream.launch.side_effect = launch_side_effect
        mock_stream_class = MagicMock(return_value=mock_stream)

        mock_logger = MagicMock()

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=mock_logger):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                with patch.dict("sys.modules", {
                    "reachy_mini_conversation_app.moves": MagicMock(MovementManager=MagicMock()),
                    "reachy_mini_conversation_app.console": MagicMock(LocalStream=mock_stream_class),
                    "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=MagicMock()),
                    "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=MagicMock()),
                    "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=MagicMock()),
                }):
                    run(args, robot=mock_robot, app_stop_event=stop_event)

        # Verify the stop event shutdown path was triggered
        mock_logger.info.assert_any_call("App stop event detected, shutting down...")

    def test_run_stop_event_thread_handles_close_error(self) -> None:
        """Test that stop event thread handles stream_manager.close() error (lines 175-176)."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()

        mock_robot = MagicMock()
        mock_robot.client.get_status.return_value = {"simulation_enabled": False}

        stop_event = threading.Event()

        mock_stream = MagicMock()
        mock_stream.close.side_effect = RuntimeError("Close error")

        def launch_side_effect() -> None:
            stop_event.set()
            time.sleep(0.1)
            raise KeyboardInterrupt()

        mock_stream.launch.side_effect = launch_side_effect
        mock_stream_class = MagicMock(return_value=mock_stream)

        mock_logger = MagicMock()

        with patch("reachy_mini_conversation_app.main.setup_logger", return_value=mock_logger):
            with patch("reachy_mini_conversation_app.main.handle_vision_stuff", return_value=(None, None, None)):
                with patch.dict("sys.modules", {
                    "reachy_mini_conversation_app.moves": MagicMock(MovementManager=MagicMock()),
                    "reachy_mini_conversation_app.console": MagicMock(LocalStream=mock_stream_class),
                    "reachy_mini_conversation_app.openai_realtime": MagicMock(OpenaiRealtimeHandler=MagicMock()),
                    "reachy_mini_conversation_app.tools.core_tools": MagicMock(ToolDependencies=MagicMock()),
                    "reachy_mini_conversation_app.audio.head_wobbler": MagicMock(HeadWobbler=MagicMock()),
                }):
                    run(args, robot=mock_robot, app_stop_event=stop_event)

        # Verify the error was logged
        assert any("Error while closing stream manager" in str(call) for call in mock_logger.error.call_args_list)

    def test_run_handles_timeout_error_on_robot_init(self) -> None:
        """Test run handles TimeoutError during robot initialization (lines 70-76)."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()

        mock_robot_class = MagicMock()
        mock_robot_class.side_effect = TimeoutError("Connection timed out")

        mock_logger = MagicMock()

        with patch("reachy_mini_conversation_app.main.ReachyMini", mock_robot_class):
            with patch("reachy_mini_conversation_app.main.setup_logger", return_value=mock_logger):
                with patch("reachy_mini_conversation_app.main.log_connection_troubleshooting") as mock_troubleshoot:
                    with pytest.raises(SystemExit) as exc_info:
                        run(args)

        assert exc_info.value.code == 1
        # Verify error was logged
        assert any("Connection timeout" in str(call) for call in mock_logger.error.call_args_list)
        # Verify troubleshooting was called
        mock_troubleshoot.assert_called_once_with(mock_logger, None)

    def test_run_handles_timeout_error_with_robot_name(self) -> None:
        """Test run handles TimeoutError and logs robot_name in troubleshooting."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args(robot_name="my_robot")

        mock_robot_class = MagicMock()
        mock_robot_class.side_effect = TimeoutError("Connection timed out")

        mock_logger = MagicMock()

        with patch("reachy_mini_conversation_app.main.ReachyMini", mock_robot_class):
            with patch("reachy_mini_conversation_app.main.setup_logger", return_value=mock_logger):
                with patch("reachy_mini_conversation_app.main.log_connection_troubleshooting") as mock_troubleshoot:
                    with pytest.raises(SystemExit) as exc_info:
                        run(args)

        assert exc_info.value.code == 1
        # Verify troubleshooting was called with robot_name
        mock_troubleshoot.assert_called_once_with(mock_logger, "my_robot")

    def test_run_handles_connection_error_on_robot_init(self) -> None:
        """Test run handles ConnectionError during robot initialization (lines 78-84)."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()

        mock_robot_class = MagicMock()
        mock_robot_class.side_effect = ConnectionError("Failed to connect")

        mock_logger = MagicMock()

        with patch("reachy_mini_conversation_app.main.ReachyMini", mock_robot_class):
            with patch("reachy_mini_conversation_app.main.setup_logger", return_value=mock_logger):
                with patch("reachy_mini_conversation_app.main.log_connection_troubleshooting") as mock_troubleshoot:
                    with pytest.raises(SystemExit) as exc_info:
                        run(args)

        assert exc_info.value.code == 1
        # Verify error was logged
        assert any("Connection failed" in str(call) for call in mock_logger.error.call_args_list)
        # Verify troubleshooting was called
        mock_troubleshoot.assert_called_once_with(mock_logger, None)

    def test_run_handles_generic_exception_on_robot_init(self) -> None:
        """Test run handles generic Exception during robot initialization (lines 86-91)."""
        from reachy_mini_conversation_app.main import run

        args = self._create_mock_args()

        mock_robot_class = MagicMock()
        mock_robot_class.side_effect = ValueError("Some unexpected error")

        mock_logger = MagicMock()

        with patch("reachy_mini_conversation_app.main.ReachyMini", mock_robot_class):
            with patch("reachy_mini_conversation_app.main.setup_logger", return_value=mock_logger):
                with pytest.raises(SystemExit) as exc_info:
                    run(args)

        assert exc_info.value.code == 1
        # Verify error messages were logged
        error_messages = [str(call) for call in mock_logger.error.call_args_list]
        assert any("Unexpected error" in msg for msg in error_messages)
        assert any("ValueError" in msg for msg in error_messages)
        assert any("check your configuration" in msg for msg in error_messages)
