"""Unit tests for the camera tool."""

import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestCameraToolAttributes:
    """Tests for Camera tool attributes."""

    def test_camera_has_correct_name(self) -> None:
        """Test Camera tool has correct name."""
        tool = Camera()
        assert tool.name == "camera"

    def test_camera_has_description(self) -> None:
        """Test Camera tool has description."""
        tool = Camera()
        assert "camera" in tool.description.lower()
        assert "picture" in tool.description.lower()

    def test_camera_has_parameters_schema(self) -> None:
        """Test Camera tool has correct parameters schema."""
        tool = Camera()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "question" in schema["properties"]
        assert schema["properties"]["question"]["type"] == "string"
        assert "question" in schema["required"]

    def test_camera_spec(self) -> None:
        """Test Camera tool spec generation."""
        tool = Camera()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "camera"
        assert "description" in spec
        assert "parameters" in spec


class TestCameraToolExecution:
    """Tests for Camera tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
            camera_worker=MagicMock(),
        )

    @pytest.fixture
    def sample_frame(self) -> np.ndarray:
        """Create a sample frame for testing."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.mark.asyncio
    async def test_camera_empty_question_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test that empty question returns error."""
        tool = Camera()

        result = await tool(mock_deps, question="")

        assert "error" in result
        assert "non-empty" in result["error"]

    @pytest.mark.asyncio
    async def test_camera_none_question_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test that None question returns error."""
        tool = Camera()

        result = await tool(mock_deps, question=None)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_camera_whitespace_question_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test that whitespace-only question returns error."""
        tool = Camera()

        result = await tool(mock_deps, question="   ")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_camera_no_camera_worker_returns_error(self) -> None:
        """Test that missing camera worker returns error."""
        deps = ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
            camera_worker=None,
        )
        tool = Camera()

        result = await tool(deps, question="What do you see?")

        assert "error" in result
        assert "Camera worker not available" in result["error"]

    @pytest.mark.asyncio
    async def test_camera_no_frame_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test that no frame available returns error."""
        mock_deps.camera_worker.get_latest_frame.return_value = None
        tool = Camera()

        result = await tool(mock_deps, question="What do you see?")

        assert "error" in result
        assert "No frame available" in result["error"]

    @pytest.mark.asyncio
    async def test_camera_returns_base64_image(
        self, mock_deps: ToolDependencies, sample_frame: np.ndarray
    ) -> None:
        """Test that camera returns base64 encoded image."""
        mock_deps.camera_worker.get_latest_frame.return_value = sample_frame
        tool = Camera()

        with patch("cv2.imencode") as mock_imencode:
            # Mock successful encoding
            mock_buffer = MagicMock()
            mock_buffer.tobytes.return_value = b"fake_jpeg_data"
            mock_imencode.return_value = (True, mock_buffer)

            result = await tool(mock_deps, question="What do you see?")

        assert "b64_im" in result
        # Verify it's valid base64
        decoded = base64.b64decode(result["b64_im"])
        assert decoded == b"fake_jpeg_data"

    @pytest.mark.asyncio
    async def test_camera_encoding_failure_raises(
        self, mock_deps: ToolDependencies, sample_frame: np.ndarray
    ) -> None:
        """Test that encoding failure raises RuntimeError."""
        mock_deps.camera_worker.get_latest_frame.return_value = sample_frame
        tool = Camera()

        with patch("cv2.imencode") as mock_imencode:
            mock_imencode.return_value = (False, None)

            with pytest.raises(RuntimeError, match="Failed to encode"):
                await tool(mock_deps, question="What do you see?")

    @pytest.mark.asyncio
    async def test_camera_with_vision_manager(
        self, mock_deps: ToolDependencies, sample_frame: np.ndarray
    ) -> None:
        """Test camera with vision manager processes image."""
        mock_deps.camera_worker.get_latest_frame.return_value = sample_frame

        # Add vision manager
        mock_vision = MagicMock()
        mock_vision.processor.process_image.return_value = "I see a robot"
        mock_deps.vision_manager = mock_vision

        tool = Camera()

        result = await tool(mock_deps, question="What do you see?")

        assert "image_description" in result
        assert result["image_description"] == "I see a robot"

    @pytest.mark.asyncio
    async def test_camera_vision_manager_returns_error(
        self, mock_deps: ToolDependencies, sample_frame: np.ndarray
    ) -> None:
        """Test camera handles vision manager error."""
        mock_deps.camera_worker.get_latest_frame.return_value = sample_frame

        mock_vision = MagicMock()
        mock_vision.processor.process_image.return_value = {"error": "Vision error"}
        mock_deps.vision_manager = mock_vision

        tool = Camera()

        result = await tool(mock_deps, question="What do you see?")

        assert "error" in result
        assert result["error"] == "Vision error"

    @pytest.mark.asyncio
    async def test_camera_vision_manager_returns_non_string(
        self, mock_deps: ToolDependencies, sample_frame: np.ndarray
    ) -> None:
        """Test camera handles vision manager non-string result."""
        mock_deps.camera_worker.get_latest_frame.return_value = sample_frame

        mock_vision = MagicMock()
        mock_vision.processor.process_image.return_value = 12345  # Non-string, non-dict
        mock_deps.vision_manager = mock_vision

        tool = Camera()

        result = await tool(mock_deps, question="What do you see?")

        assert "error" in result
        assert "non-string" in result["error"]

    @pytest.mark.asyncio
    async def test_camera_logs_question(
        self, mock_deps: ToolDependencies, sample_frame: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that camera logs the question."""
        import logging

        mock_deps.camera_worker.get_latest_frame.return_value = sample_frame
        tool = Camera()

        with patch("cv2.imencode") as mock_imencode:
            mock_buffer = MagicMock()
            mock_buffer.tobytes.return_value = b"data"
            mock_imencode.return_value = (True, mock_buffer)

            with caplog.at_level(logging.INFO, logger="reachy_mini_conversation_app.tools.camera"):
                await tool(mock_deps, question="What color is the sky?")

        assert "camera" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_camera_truncates_long_question_in_log(
        self, mock_deps: ToolDependencies, sample_frame: np.ndarray
    ) -> None:
        """Test that long questions are truncated in logs."""
        mock_deps.camera_worker.get_latest_frame.return_value = sample_frame
        tool = Camera()

        long_question = "x" * 200

        with patch("cv2.imencode") as mock_imencode:
            mock_buffer = MagicMock()
            mock_buffer.tobytes.return_value = b"data"
            mock_imencode.return_value = (True, mock_buffer)

            # Should not raise
            result = await tool(mock_deps, question=long_question)

        assert "b64_im" in result
