"""Tests for the camera tool."""

import base64
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_camera_tool_preserves_frame_color_for_uploaded_jpeg() -> None:
    """The JPEG uploaded to the model should preserve the intended frame color."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8)

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "b64_im" in result

    jpeg_bytes = base64.b64decode(result["b64_im"])
    decoded = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    pixel = decoded.getpixel((0, 0))
    assert isinstance(pixel, tuple)
    red, green, blue = pixel

    assert red > 200
    assert green < 40
    assert blue < 40


@pytest.mark.asyncio
async def test_camera_tool_uses_local_vision_processor_when_available() -> None:
    """The camera tool should use on-demand local vision when configured."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((32, 32, 3), dtype=np.uint8)

    vision_processor = MagicMock()
    vision_processor.process_image.return_value = "A red cup on a table."

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        vision_processor=vision_processor,
    )

    result = await Camera()(deps, question="What do you see?")

    assert result == {"image_description": "A red cup on a table."}
    vision_processor.process_image.assert_called_once_with(
        camera_worker.get_latest_frame.return_value,
        "What do you see?",
    )
