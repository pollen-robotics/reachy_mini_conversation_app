"""Tests for utility helpers."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.utils import initialize_camera_and_vision


def test_initialize_camera_and_vision_raises_when_local_vision_init_fails() -> None:
    """Explicit local vision requests should fail fast when setup fails."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker=None,
        local_vision=True,
    )

    with patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker, \
         patch("reachy_mini_conversation_app.vision.processors.initialize_vision_processor", return_value=None):
        with pytest.raises(RuntimeError, match="Failed to initialize local vision processor"):
            initialize_camera_and_vision(args, MagicMock())

    mock_camera_worker.assert_called_once()
