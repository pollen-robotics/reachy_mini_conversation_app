"""Unit tests for the utils module."""

import logging
import argparse
from unittest.mock import MagicMock, patch

import pytest


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_default_values(self) -> None:
        """Test parse_args returns correct defaults with no arguments."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog"]):
            args, unknown = parse_args()

        assert args.head_tracker is None
        assert args.no_camera is False
        assert args.local_vision is False
        assert args.gradio is False
        assert args.debug is False
        assert args.robot_name is None
        assert unknown == []

    def test_parse_args_head_tracker_yolo(self) -> None:
        """Test parse_args with --head-tracker yolo."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog", "--head-tracker", "yolo"]):
            args, _ = parse_args()

        assert args.head_tracker == "yolo"

    def test_parse_args_head_tracker_mediapipe(self) -> None:
        """Test parse_args with --head-tracker mediapipe."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog", "--head-tracker", "mediapipe"]):
            args, _ = parse_args()

        assert args.head_tracker == "mediapipe"

    def test_parse_args_no_camera(self) -> None:
        """Test parse_args with --no-camera flag."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog", "--no-camera"]):
            args, _ = parse_args()

        assert args.no_camera is True

    def test_parse_args_local_vision(self) -> None:
        """Test parse_args with --local-vision flag."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog", "--local-vision"]):
            args, _ = parse_args()

        assert args.local_vision is True

    def test_parse_args_gradio(self) -> None:
        """Test parse_args with --gradio flag."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog", "--gradio"]):
            args, _ = parse_args()

        assert args.gradio is True

    def test_parse_args_debug(self) -> None:
        """Test parse_args with --debug flag."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog", "--debug"]):
            args, _ = parse_args()

        assert args.debug is True

    def test_parse_args_robot_name(self) -> None:
        """Test parse_args with --robot-name flag."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog", "--robot-name", "my_robot"]):
            args, _ = parse_args()

        assert args.robot_name == "my_robot"

    def test_parse_args_multiple_flags(self) -> None:
        """Test parse_args with multiple flags combined."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog", "--debug", "--gradio", "--no-camera"]):
            args, _ = parse_args()

        assert args.debug is True
        assert args.gradio is True
        assert args.no_camera is True

    def test_parse_args_unknown_args(self) -> None:
        """Test parse_args returns unknown arguments."""
        from reachy_mini_conversation_app.utils import parse_args

        with patch("sys.argv", ["prog", "--debug", "--unknown-flag", "value"]):
            args, unknown = parse_args()

        assert args.debug is True
        assert "--unknown-flag" in unknown
        assert "value" in unknown


class TestHandleVisionStuff:
    """Tests for handle_vision_stuff function."""

    def test_handle_vision_stuff_no_camera(self) -> None:
        """Test handle_vision_stuff with --no-camera flag."""
        from reachy_mini_conversation_app.utils import handle_vision_stuff

        args = argparse.Namespace(
            no_camera=True,
            head_tracker=None,
            local_vision=False,
        )
        mock_robot = MagicMock()

        camera_worker, head_tracker, vision_manager = handle_vision_stuff(args, mock_robot)

        assert camera_worker is None
        assert head_tracker is None
        assert vision_manager is None

    @patch("reachy_mini_conversation_app.utils.CameraWorker")
    def test_handle_vision_stuff_camera_enabled_no_tracker(self, mock_camera_worker_cls: MagicMock) -> None:
        """Test handle_vision_stuff with camera but no head tracker."""
        from reachy_mini_conversation_app.utils import handle_vision_stuff

        args = argparse.Namespace(
            no_camera=False,
            head_tracker=None,
            local_vision=False,
        )
        mock_robot = MagicMock()
        mock_camera_worker = MagicMock()
        mock_camera_worker_cls.return_value = mock_camera_worker

        camera_worker, head_tracker, vision_manager = handle_vision_stuff(args, mock_robot)

        assert camera_worker is mock_camera_worker
        assert head_tracker is None
        assert vision_manager is None
        mock_camera_worker_cls.assert_called_once_with(mock_robot, None)

    @patch("reachy_mini_conversation_app.utils.CameraWorker")
    def test_handle_vision_stuff_yolo_tracker(self, mock_camera_worker_cls: MagicMock) -> None:
        """Test handle_vision_stuff with YOLO head tracker."""
        from reachy_mini_conversation_app.utils import handle_vision_stuff

        args = argparse.Namespace(
            no_camera=False,
            head_tracker="yolo",
            local_vision=False,
        )
        mock_robot = MagicMock()
        mock_camera_worker = MagicMock()
        mock_camera_worker_cls.return_value = mock_camera_worker

        # Mock the yolo head tracker module
        mock_yolo_tracker = MagicMock()
        mock_yolo_module = MagicMock()
        mock_yolo_module.HeadTracker.return_value = mock_yolo_tracker

        with patch.dict(
            "sys.modules",
            {"reachy_mini_conversation_app.vision.yolo_head_tracker": mock_yolo_module},
        ):
            camera_worker, head_tracker, vision_manager = handle_vision_stuff(args, mock_robot)

        assert camera_worker is mock_camera_worker
        assert head_tracker is mock_yolo_tracker
        assert vision_manager is None
        mock_camera_worker_cls.assert_called_once_with(mock_robot, mock_yolo_tracker)

    def test_handle_vision_stuff_mediapipe_tracker(self) -> None:
        """Test handle_vision_stuff with MediaPipe head tracker."""
        args = argparse.Namespace(
            no_camera=False,
            head_tracker="mediapipe",
            local_vision=False,
        )
        mock_robot = MagicMock()
        mock_camera_worker = MagicMock()

        # Mock the mediapipe head tracker module before importing
        mock_mediapipe_tracker = MagicMock()
        mock_mediapipe_module = MagicMock()
        mock_mediapipe_module.HeadTracker.return_value = mock_mediapipe_tracker

        # Need to set up mocks before the import happens
        import sys

        sys.modules["reachy_mini_toolbox"] = MagicMock()
        sys.modules["reachy_mini_toolbox.vision"] = mock_mediapipe_module

        try:
            # Re-import the function to trigger the mediapipe branch
            from importlib import reload

            import reachy_mini_conversation_app.utils as utils_module

            reload(utils_module)

            # Patch CameraWorker after reload
            with patch.object(utils_module, "CameraWorker", return_value=mock_camera_worker):
                camera_worker, head_tracker, vision_manager = utils_module.handle_vision_stuff(args, mock_robot)

            assert camera_worker is mock_camera_worker
            assert head_tracker is mock_mediapipe_tracker
            assert vision_manager is None
        finally:
            # Clean up sys.modules
            sys.modules.pop("reachy_mini_toolbox", None)
            sys.modules.pop("reachy_mini_toolbox.vision", None)
            # Reload to restore original state
            from importlib import reload

            import reachy_mini_conversation_app.utils as utils_module

            reload(utils_module)

    @patch("reachy_mini_conversation_app.utils.CameraWorker")
    def test_handle_vision_stuff_local_vision_enabled(self, mock_camera_worker_cls: MagicMock) -> None:
        """Test handle_vision_stuff with local vision enabled."""
        from reachy_mini_conversation_app.utils import handle_vision_stuff

        args = argparse.Namespace(
            no_camera=False,
            head_tracker=None,
            local_vision=True,
        )
        mock_robot = MagicMock()
        mock_camera_worker = MagicMock()
        mock_camera_worker_cls.return_value = mock_camera_worker

        mock_vision_manager = MagicMock()
        mock_processors_module = MagicMock()
        mock_processors_module.initialize_vision_manager.return_value = mock_vision_manager

        with patch.dict(
            "sys.modules",
            {"reachy_mini_conversation_app.vision.processors": mock_processors_module},
        ):
            camera_worker, head_tracker, vision_manager = handle_vision_stuff(args, mock_robot)

        assert camera_worker is mock_camera_worker
        assert head_tracker is None
        assert vision_manager is mock_vision_manager
        mock_processors_module.initialize_vision_manager.assert_called_once_with(mock_camera_worker)

    @patch("reachy_mini_conversation_app.utils.CameraWorker")
    def test_handle_vision_stuff_local_vision_import_error(self, mock_camera_worker_cls: MagicMock) -> None:
        """Test handle_vision_stuff raises ImportError when local vision deps missing."""
        from reachy_mini_conversation_app.utils import handle_vision_stuff

        args = argparse.Namespace(
            no_camera=False,
            head_tracker=None,
            local_vision=True,
        )
        mock_robot = MagicMock()
        mock_camera_worker = MagicMock()
        mock_camera_worker_cls.return_value = mock_camera_worker

        # Make the import fail
        with patch.dict("sys.modules", {"reachy_mini_conversation_app.vision.processors": None}):
            # Force ImportError by making the module unavailable
            import sys

            original_modules = sys.modules.copy()

            def mock_import(name: str, *args: object, **kwargs: object) -> object:
                if "processors" in name:
                    raise ImportError("Missing dependencies")
                return original_modules.get(name)

            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError) as exc_info:
                    handle_vision_stuff(args, mock_robot)

                assert "local_vision" in str(exc_info.value)

    @patch("reachy_mini_conversation_app.utils.CameraWorker")
    def test_handle_vision_stuff_logs_when_not_local_vision(
        self, mock_camera_worker_cls: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handle_vision_stuff logs info when not using local vision."""
        from reachy_mini_conversation_app.utils import handle_vision_stuff

        args = argparse.Namespace(
            no_camera=False,
            head_tracker=None,
            local_vision=False,
        )
        mock_robot = MagicMock()
        mock_camera_worker = MagicMock()
        mock_camera_worker_cls.return_value = mock_camera_worker

        with caplog.at_level(logging.INFO, logger="reachy_mini_conversation_app.utils"):
            handle_vision_stuff(args, mock_robot)

        assert "gpt-realtime for vision" in caplog.text


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_debug_mode(self) -> None:
        """Test setup_logger in debug mode."""
        from reachy_mini_conversation_app.utils import setup_logger

        # Clear any existing handlers to ensure clean state
        root_logger = logging.getLogger()
        original_level = root_logger.level

        try:
            logger = setup_logger(debug=True)

            assert logger is not None
            assert logger.name == "reachy_mini_conversation_app.utils"

            # Check third-party loggers are set to INFO in debug mode
            assert logging.getLogger("aiortc").level == logging.INFO
            assert logging.getLogger("fastrtc").level == logging.INFO
            assert logging.getLogger("aioice").level == logging.INFO
            assert logging.getLogger("openai").level == logging.INFO
            assert logging.getLogger("websockets").level == logging.INFO
        finally:
            # Restore original state
            root_logger.setLevel(original_level)

    def test_setup_logger_normal_mode(self) -> None:
        """Test setup_logger in normal (non-debug) mode."""
        from reachy_mini_conversation_app.utils import setup_logger

        root_logger = logging.getLogger()
        original_level = root_logger.level

        try:
            logger = setup_logger(debug=False)

            assert logger is not None

            # Check third-party loggers are silenced in normal mode
            assert logging.getLogger("aiortc").level == logging.ERROR
            assert logging.getLogger("fastrtc").level == logging.ERROR
            assert logging.getLogger("aioice").level == logging.WARNING
        finally:
            root_logger.setLevel(original_level)

    def test_setup_logger_returns_module_logger(self) -> None:
        """Test setup_logger returns the correct module logger."""
        from reachy_mini_conversation_app.utils import setup_logger

        logger = setup_logger(debug=False)

        assert logger.name == "reachy_mini_conversation_app.utils"

    def test_setup_logger_sets_format(self) -> None:
        """Test setup_logger configures the correct format."""
        from reachy_mini_conversation_app.utils import setup_logger

        # This test verifies logging.basicConfig is called with format
        with patch("logging.basicConfig") as mock_basic_config:
            setup_logger(debug=False)

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            assert "format" in call_kwargs
            assert "%(asctime)s" in call_kwargs["format"]
            assert "%(levelname)s" in call_kwargs["format"]

    def test_setup_logger_sets_correct_level_for_debug(self) -> None:
        """Test setup_logger sets DEBUG level when debug=True."""
        from reachy_mini_conversation_app.utils import setup_logger

        with patch("logging.basicConfig") as mock_basic_config:
            setup_logger(debug=True)

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            assert call_kwargs["level"] == logging.DEBUG

    def test_setup_logger_sets_correct_level_for_info(self) -> None:
        """Test setup_logger sets INFO level when debug=False."""
        from reachy_mini_conversation_app.utils import setup_logger

        with patch("logging.basicConfig") as mock_basic_config:
            setup_logger(debug=False)

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            assert call_kwargs["level"] == logging.INFO

    def test_setup_logger_filters_warnings(self) -> None:
        """Test setup_logger sets up warning filters."""
        from reachy_mini_conversation_app.utils import setup_logger

        with patch("warnings.filterwarnings") as mock_filter:
            setup_logger(debug=False)

            # Should be called at least twice for the two warning filters
            assert mock_filter.call_count >= 2

            # Check one of the specific filters
            calls = mock_filter.call_args_list
            call_messages = [str(c) for c in calls]
            assert any("AVCaptureDeviceTypeExternal" in msg for msg in call_messages)
            assert any("aiortc" in msg for msg in call_messages)


class TestHandleVisionStuffEdgeCases:
    """Tests for edge cases in handle_vision_stuff function."""

    @patch("reachy_mini_conversation_app.utils.CameraWorker")
    def test_handle_vision_stuff_unknown_head_tracker(self, mock_camera_worker_cls: MagicMock) -> None:
        """Test handle_vision_stuff with unknown head tracker value (branch 60->66)."""
        from reachy_mini_conversation_app.utils import handle_vision_stuff

        args = argparse.Namespace(
            no_camera=False,
            head_tracker="unknown_tracker",  # Not yolo, not mediapipe
            local_vision=False,
        )
        mock_robot = MagicMock()
        mock_camera_worker = MagicMock()
        mock_camera_worker_cls.return_value = mock_camera_worker

        camera_worker, head_tracker, vision_manager = handle_vision_stuff(args, mock_robot)

        # Should have camera but no head tracker since unknown tracker is ignored
        assert camera_worker is mock_camera_worker
        assert head_tracker is None
        assert vision_manager is None
        mock_camera_worker_cls.assert_called_once_with(mock_robot, None)


class TestLogConnectionTroubleshooting:
    """Tests for log_connection_troubleshooting function."""

    def test_log_connection_troubleshooting_with_robot_name(self) -> None:
        """Test log_connection_troubleshooting logs correctly with robot_name."""
        from reachy_mini_conversation_app.utils import log_connection_troubleshooting

        mock_logger = MagicMock()
        log_connection_troubleshooting(mock_logger, robot_name="my_robot")

        # Check that error was called multiple times (6 calls total)
        assert mock_logger.error.call_count == 6

        # Check specific messages
        calls = [str(c) for c in mock_logger.error.call_args_list]
        assert any("Troubleshooting steps" in c for c in calls)
        assert any("Verify reachy-mini-daemon is running" in c for c in calls)
        assert any("--robot-name 'my_robot'" in c for c in calls)
        assert any("check network connectivity" in c for c in calls)
        assert any("Review daemon logs" in c for c in calls)
        assert any("Restart the daemon" in c for c in calls)

    def test_log_connection_troubleshooting_without_robot_name(self) -> None:
        """Test log_connection_troubleshooting logs correctly without robot_name."""
        from reachy_mini_conversation_app.utils import log_connection_troubleshooting

        mock_logger = MagicMock()
        log_connection_troubleshooting(mock_logger, robot_name=None)

        # Check that error was called multiple times (6 calls total)
        assert mock_logger.error.call_count == 6

        # Check specific messages - should suggest adding --robot-name flag
        calls = [str(c) for c in mock_logger.error.call_args_list]
        assert any("Troubleshooting steps" in c for c in calls)
        assert any("Verify reachy-mini-daemon is running" in c for c in calls)
        assert any("If daemon uses --robot-name" in c for c in calls)
        assert any("--robot-name <name>" in c for c in calls)
        assert any("check network connectivity" in c for c in calls)
        assert any("Review daemon logs" in c for c in calls)
        assert any("Restart the daemon" in c for c in calls)
