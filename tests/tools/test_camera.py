"""Tests for the camera tool."""

import base64
from io import BytesIO
from unittest.mock import MagicMock

import av
import numpy as np
import pytest

from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_camera_tool_preserves_frame_color_for_uploaded_jpeg() -> None:
    """The JPEG uploaded to the model should preserve the intended frame color."""
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame.return_value = np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8)

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "b64_im" in result

    jpeg_bytes = base64.b64decode(result["b64_im"])
    with av.open(BytesIO(jpeg_bytes)) as container:
        decoded = next(container.decode(video=0)).to_ndarray(format="rgb24")
    red, green, blue = decoded[0, 0]

    assert red > 200
    assert green < 40
    assert blue < 40


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_camera_disabled() -> None:
    """With the camera disabled the tool returns an error and never reads a frame."""
    reachy_mini = MagicMock()
    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=False,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result
    reachy_mini.media.get_frame.assert_not_called()
