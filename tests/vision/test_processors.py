"""Unit tests for the vision processors module."""

import sys
import time
import threading
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from numpy.typing import NDArray


# Create a real exception class for torch.cuda.OutOfMemoryError
class MockOutOfMemoryError(Exception):
    """Mock CUDA OOM error."""

    pass


# Mock heavy dependencies before importing the module
@pytest.fixture(autouse=True)
def mock_heavy_imports():
    """Mock torch, cv2, and transformers before importing processors."""
    # Create mock modules
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.float32 = "float32"
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    mock_torch.cuda.OutOfMemoryError = MockOutOfMemoryError

    mock_cv2 = MagicMock()
    mock_cv2.IMWRITE_JPEG_QUALITY = 1
    mock_cv2.imencode.return_value = (True, np.array([255, 216, 255], dtype=np.uint8))

    mock_transformers = MagicMock()
    mock_huggingface = MagicMock()

    # Inject mocks into sys.modules
    with patch.dict(
        sys.modules,
        {
            "torch": mock_torch,
            "cv2": mock_cv2,
            "transformers": mock_transformers,
            "huggingface_hub": mock_huggingface,
        },
    ):
        yield {
            "torch": mock_torch,
            "cv2": mock_cv2,
            "transformers": mock_transformers,
            "huggingface_hub": mock_huggingface,
        }


class TestVisionConfig:
    """Tests for VisionConfig dataclass."""

    def test_default_values(self, mock_heavy_imports: dict) -> None:
        """Test VisionConfig has sensible defaults."""
        from reachy_mini_conversation_app.vision.processors import VisionConfig

        config = VisionConfig()

        assert config.vision_interval == 5.0
        assert config.max_new_tokens == 64
        assert config.jpeg_quality == 85
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.device_preference == "auto"

    def test_custom_values(self, mock_heavy_imports: dict) -> None:
        """Test VisionConfig with custom values."""
        from reachy_mini_conversation_app.vision.processors import VisionConfig

        config = VisionConfig(
            model_path="custom/model",
            vision_interval=10.0,
            max_new_tokens=128,
            jpeg_quality=90,
            max_retries=5,
            retry_delay=2.0,
            device_preference="cuda",
        )

        assert config.model_path == "custom/model"
        assert config.vision_interval == 10.0
        assert config.max_new_tokens == 128
        assert config.jpeg_quality == 90
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.device_preference == "cuda"


class TestVisionProcessorInit:
    """Tests for VisionProcessor initialization."""

    def test_init_with_default_config(self, mock_heavy_imports: dict) -> None:
        """Test VisionProcessor initializes with default config."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        processor = VisionProcessor()

        assert processor.vision_config is not None
        assert processor.processor is None
        assert processor.model is None
        assert processor._initialized is False

    def test_init_with_custom_config(self, mock_heavy_imports: dict) -> None:
        """Test VisionProcessor initializes with custom config."""
        from reachy_mini_conversation_app.vision.processors import (
            VisionConfig,
            VisionProcessor,
        )

        config = VisionConfig(vision_interval=10.0)
        processor = VisionProcessor(vision_config=config)

        assert processor.vision_config == config
        assert processor.vision_config.vision_interval == 10.0


class TestVisionProcessorDetermineDevice:
    """Tests for VisionProcessor._determine_device method."""

    def test_determine_device_cpu_preference(self, mock_heavy_imports: dict) -> None:
        """Test device determination with cpu preference."""
        from reachy_mini_conversation_app.vision.processors import (
            VisionConfig,
            VisionProcessor,
        )

        config = VisionConfig(device_preference="cpu")
        processor = VisionProcessor(vision_config=config)
        assert processor.device == "cpu"

    def test_determine_device_cuda_available(self, mock_heavy_imports: dict) -> None:
        """Test device determination with cuda available."""
        mock_heavy_imports["torch"].cuda.is_available.return_value = True
        mock_heavy_imports["torch"].backends.mps.is_available.return_value = False

        from reachy_mini_conversation_app.vision.processors import (
            VisionConfig,
            VisionProcessor,
        )

        # Need to reimport to get fresh instance with updated mock
        import importlib
        import reachy_mini_conversation_app.vision.processors as proc_module

        importlib.reload(proc_module)

        config = proc_module.VisionConfig(device_preference="cuda")
        processor = proc_module.VisionProcessor(vision_config=config)
        assert processor.device == "cuda"

    def test_determine_device_cuda_not_available(self, mock_heavy_imports: dict) -> None:
        """Test device determination when cuda requested but not available."""
        mock_heavy_imports["torch"].cuda.is_available.return_value = False
        mock_heavy_imports["torch"].backends.mps.is_available.return_value = False

        from reachy_mini_conversation_app.vision.processors import (
            VisionConfig,
            VisionProcessor,
        )

        config = VisionConfig(device_preference="cuda")
        processor = VisionProcessor(vision_config=config)
        assert processor.device == "cpu"

    def test_determine_device_auto_fallback_cpu(self, mock_heavy_imports: dict) -> None:
        """Test auto device selection falls back to cpu."""
        mock_heavy_imports["torch"].cuda.is_available.return_value = False
        mock_heavy_imports["torch"].backends.mps.is_available.return_value = False

        from reachy_mini_conversation_app.vision.processors import (
            VisionConfig,
            VisionProcessor,
        )

        config = VisionConfig(device_preference="auto")
        processor = VisionProcessor(vision_config=config)
        assert processor.device == "cpu"


class TestVisionProcessorInitialize:
    """Tests for VisionProcessor.initialize method."""

    def test_initialize_success(self, mock_heavy_imports: dict) -> None:
        """Test successful model initialization."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            with patch(
                "reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText.from_pretrained",
                return_value=mock_model,
            ):
                processor = VisionProcessor()
                result = processor.initialize()

                assert result is True
                assert processor._initialized is True
                assert processor.processor is not None
                assert processor.model is not None

    def test_initialize_failure(self, mock_heavy_imports: dict) -> None:
        """Test model initialization failure."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            side_effect=Exception("Model not found"),
        ):
            processor = VisionProcessor()
            result = processor.initialize()

            assert result is False
            assert processor._initialized is False


class TestVisionProcessorProcessImage:
    """Tests for VisionProcessor.process_image method."""

    def test_process_image_not_initialized(self, mock_heavy_imports: dict) -> None:
        """Test process_image returns error when not initialized."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        processor = VisionProcessor()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = processor.process_image(image)

        assert result == "Vision model not initialized"

    def test_process_image_success(self, mock_heavy_imports: dict) -> None:
        """Test successful image processing."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_processor.batch_decode.return_value = ["assistant\nThis is a test description."]
        mock_processor.tokenizer.eos_token_id = 0

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = MagicMock()

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            with patch(
                "reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText.from_pretrained",
                return_value=mock_model,
            ):
                processor = VisionProcessor()
                processor.initialize()

                image = np.zeros((480, 640, 3), dtype=np.uint8)
                result = processor.process_image(image)

                assert "test description" in result.lower()

    def test_process_image_encode_failure(self, mock_heavy_imports: dict) -> None:
        """Test process_image handles encode failure."""
        mock_heavy_imports["cv2"].imencode.return_value = (False, None)

        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            with patch(
                "reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText.from_pretrained",
                return_value=mock_model,
            ):
                processor = VisionProcessor()
                processor.initialize()

                image = np.zeros((480, 640, 3), dtype=np.uint8)
                result = processor.process_image(image)

                assert result == "Failed to encode image"

    def test_process_image_max_retries_exceeded(self, mock_heavy_imports: dict) -> None:
        """Test process_image returns error after max retries."""
        mock_heavy_imports["cv2"].imencode.return_value = (
            True,
            np.array([255, 216, 255], dtype=np.uint8),
        )

        from reachy_mini_conversation_app.vision.processors import (
            VisionConfig,
            VisionProcessor,
        )

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.side_effect = Exception("Persistent error")

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            with patch(
                "reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText.from_pretrained",
                return_value=mock_model,
            ):
                config = VisionConfig(max_retries=2, retry_delay=0.01)
                processor = VisionProcessor(vision_config=config)
                processor.initialize()

                image = np.zeros((480, 640, 3), dtype=np.uint8)
                result = processor.process_image(image)

                assert "error after 2 attempts" in result.lower()


class TestVisionProcessorExtractResponse:
    """Tests for VisionProcessor._extract_response method."""

    def test_extract_response_assistant_marker(self, mock_heavy_imports: dict) -> None:
        """Test response extraction with assistant marker."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        processor = VisionProcessor()

        result = processor._extract_response("Some preamble\nassistant\nThis is the response.")

        assert result == "This is the response."

    def test_extract_response_assistant_colon_marker(self, mock_heavy_imports: dict) -> None:
        """Test response extraction with Assistant: marker."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        processor = VisionProcessor()

        result = processor._extract_response("Question here\nAssistant: The answer is 42.")

        assert result == "The answer is 42."

    def test_extract_response_response_marker(self, mock_heavy_imports: dict) -> None:
        """Test response extraction with Response: marker."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        processor = VisionProcessor()

        result = processor._extract_response("Prompt\nResponse: Here is my answer.")

        assert result == "Here is my answer."

    def test_extract_response_double_newline(self, mock_heavy_imports: dict) -> None:
        """Test response extraction with double newline."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        processor = VisionProcessor()

        result = processor._extract_response("First part\n\nSecond part here.")

        assert result == "Second part here."

    def test_extract_response_fallback(self, mock_heavy_imports: dict) -> None:
        """Test response extraction fallback to full text."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        processor = VisionProcessor()

        result = processor._extract_response("  Just plain text  ")

        assert result == "Just plain text"


class TestVisionProcessorGetModelInfo:
    """Tests for VisionProcessor.get_model_info method."""

    def test_get_model_info_not_initialized(self, mock_heavy_imports: dict) -> None:
        """Test get_model_info when not initialized."""
        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        processor = VisionProcessor()

        info = processor.get_model_info()

        assert info["initialized"] is False
        assert "device" in info
        assert "model_path" in info

    def test_get_model_info_cuda_available(self, mock_heavy_imports: dict) -> None:
        """Test get_model_info with CUDA available."""
        mock_heavy_imports["torch"].cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        mock_heavy_imports["torch"].cuda.get_device_properties.return_value = mock_props

        from reachy_mini_conversation_app.vision.processors import VisionProcessor

        processor = VisionProcessor()
        info = processor.get_model_info()

        assert info["cuda_available"] is True
        assert info["gpu_memory"] == 8


class TestVisionManagerInit:
    """Tests for VisionManager initialization."""

    def test_init_success(self, mock_heavy_imports: dict) -> None:
        """Test VisionManager initializes successfully."""
        from reachy_mini_conversation_app.vision.processors import VisionManager

        mock_camera = MagicMock()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            with patch(
                "reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText.from_pretrained",
                return_value=mock_model,
            ):
                manager = VisionManager(mock_camera)

                assert manager.camera == mock_camera
                assert manager.processor is not None
                assert manager._thread is None

    def test_init_failure(self, mock_heavy_imports: dict) -> None:
        """Test VisionManager raises on initialization failure."""
        from reachy_mini_conversation_app.vision.processors import VisionManager

        mock_camera = MagicMock()

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            side_effect=Exception("Model error"),
        ):
            with pytest.raises(RuntimeError, match="initialization failed"):
                VisionManager(mock_camera)


class TestVisionManagerStartStop:
    """Tests for VisionManager start/stop methods."""

    def test_start_creates_thread(self, mock_heavy_imports: dict) -> None:
        """Test start creates worker thread."""
        from reachy_mini_conversation_app.vision.processors import VisionManager

        mock_camera = MagicMock()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            with patch(
                "reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText.from_pretrained",
                return_value=mock_model,
            ):
                manager = VisionManager(mock_camera)
                manager.start()

                try:
                    assert manager._thread is not None
                    assert manager._thread.is_alive()
                    assert manager._thread.daemon is True
                finally:
                    manager.stop()

    def test_stop_joins_thread(self, mock_heavy_imports: dict) -> None:
        """Test stop joins worker thread."""
        from reachy_mini_conversation_app.vision.processors import VisionManager

        mock_camera = MagicMock()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            with patch(
                "reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText.from_pretrained",
                return_value=mock_model,
            ):
                manager = VisionManager(mock_camera)
                manager.start()
                thread = manager._thread
                manager.stop()

                assert manager._stop_event.is_set()
                assert thread is not None
                assert not thread.is_alive()


class TestVisionManagerGetStatus:
    """Tests for VisionManager.get_status method."""

    def test_get_status(self, mock_heavy_imports: dict) -> None:
        """Test get_status returns expected structure."""
        from reachy_mini_conversation_app.vision.processors import VisionManager

        mock_camera = MagicMock()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch(
            "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            with patch(
                "reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText.from_pretrained",
                return_value=mock_model,
            ):
                manager = VisionManager(mock_camera)
                status = manager.get_status()

                assert "last_processed" in status
                assert "processor_info" in status
                assert "config" in status
                assert "interval" in status["config"]


class TestInitializeVisionManager:
    """Tests for initialize_vision_manager function."""

    def test_initialize_success(self, mock_heavy_imports: dict) -> None:
        """Test successful vision manager initialization."""
        from reachy_mini_conversation_app.vision.processors import (
            VisionManager,
            initialize_vision_manager,
        )

        mock_camera = MagicMock()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch(
            "reachy_mini_conversation_app.vision.processors.snapshot_download"
        ) as mock_download:
            with patch(
                "reachy_mini_conversation_app.vision.processors.AutoProcessor.from_pretrained",
                return_value=mock_processor,
            ):
                with patch(
                    "reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText.from_pretrained",
                    return_value=mock_model,
                ):
                    with patch("os.makedirs"):
                        result = initialize_vision_manager(mock_camera)

                        assert result is not None
                        assert isinstance(result, VisionManager)
                        mock_download.assert_called_once()

    def test_initialize_failure(self, mock_heavy_imports: dict) -> None:
        """Test vision manager initialization failure."""
        from reachy_mini_conversation_app.vision.processors import initialize_vision_manager

        mock_camera = MagicMock()

        with patch(
            "reachy_mini_conversation_app.vision.processors.snapshot_download",
            side_effect=Exception("Download failed"),
        ):
            with patch("os.makedirs"):
                result = initialize_vision_manager(mock_camera)

                assert result is None
