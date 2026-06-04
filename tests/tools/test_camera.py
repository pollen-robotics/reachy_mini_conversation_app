"""Tests for the camera tool."""

import wave
import base64
from io import BytesIO
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps_with_camera(
    *,
    reachy_mini: object | None = None,
    vision_processor: object | None = None,
) -> tuple[ToolDependencies, MagicMock]:
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
    deps = ToolDependencies(
        reachy_mini=reachy_mini if reachy_mini is not None else MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        vision_processor=vision_processor,
    )
    return deps, camera_worker


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
    vision_processor = MagicMock()
    vision_processor.process_image.return_value = "A red cup on a table."
    deps, camera_worker = _deps_with_camera(vision_processor=vision_processor)

    result = await Camera()(deps, question="What do you see?")

    assert result == {"image_description": "A red cup on a table."}
    vision_processor.process_image.assert_called_once_with(
        camera_worker.get_latest_frame.return_value,
        "What do you see?",
    )


@pytest.mark.asyncio
async def test_camera_tool_plays_snapshot_sound_after_frame_capture() -> None:
    """Capturing a frame should play the packaged camera snapshot sound."""
    play_sound = MagicMock()
    reachy_mini = SimpleNamespace(media=SimpleNamespace(play_sound=play_sound))
    deps, _camera_worker = _deps_with_camera(reachy_mini=reachy_mini)

    await Camera()(deps, question="What is in front of you?")

    play_sound.assert_called_once()
    sound_path = Path(play_sound.call_args.args[0])
    assert sound_path.name == "camera_snapshot.wav"
    assert sound_path.exists()
    with wave.open(str(sound_path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 44100
        assert wav.getnframes() / wav.getframerate() < 1.2


@pytest.mark.asyncio
async def test_camera_tool_continues_when_snapshot_sound_fails() -> None:
    """Audio playback is best-effort and must not break camera responses."""
    play_sound = MagicMock(side_effect=RuntimeError("speaker unavailable"))
    reachy_mini = SimpleNamespace(media=SimpleNamespace(play_sound=play_sound))
    deps, _camera_worker = _deps_with_camera(reachy_mini=reachy_mini)

    result = await Camera()(deps, question="What color is this?")

    assert "b64_im" in result
    play_sound.assert_called_once()
