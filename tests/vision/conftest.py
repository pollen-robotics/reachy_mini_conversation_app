"""Shared fixtures for vision module tests."""

from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def mock_torch() -> Generator[MagicMock, None, None]:
    """Mock torch module."""
    with patch("reachy_mini_conversation_app.vision.processors.torch") as mock:
        mock.cuda.is_available.return_value = False
        mock.backends.mps.is_available.return_value = False
        mock.float32 = "float32"
        mock.bfloat16 = "bfloat16"
        mock.no_grad.return_value.__enter__ = MagicMock()
        mock.no_grad.return_value.__exit__ = MagicMock()
        yield mock


@pytest.fixture
def mock_cv2() -> Generator[MagicMock, None, None]:
    """Mock cv2 module."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock:
        mock.IMWRITE_JPEG_QUALITY = 1
        mock.imencode.return_value = (True, np.array([255, 216, 255], dtype=np.uint8))
        yield mock


@pytest.fixture
def sample_image() -> NDArray[np.uint8]:
    """Create a sample test image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_processor() -> MagicMock:
    """Mock AutoProcessor."""
    mock = MagicMock()
    mock.apply_chat_template.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    mock.batch_decode.return_value = ["assistant\nThis is a test description."]
    mock.tokenizer.eos_token_id = 0
    return mock


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock AutoModelForImageTextToText."""
    mock = MagicMock()
    mock.generate.return_value = MagicMock()
    mock.eval.return_value = None
    mock.to.return_value = mock
    return mock


@pytest.fixture
def mock_camera() -> MagicMock:
    """Mock camera for VisionManager."""
    mock = MagicMock()
    mock.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return mock
