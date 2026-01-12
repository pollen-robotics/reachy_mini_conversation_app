"""Tests for the vision processing module."""

import time
from typing import Any
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.vision.processors import (
    VisionConfig,
    VisionManager,
    VisionProcessor,
    initialize_vision_manager,
)


def test_vision_config_defaults() -> None:
    """Test VisionConfig has sensible defaults."""
    config = VisionConfig()
    assert config.vision_interval == 5.0
    assert config.max_new_tokens == 64
    assert config.jpeg_quality == 85
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert config.device_preference == "auto"


def test_vision_config_custom_values() -> None:
    """Test VisionConfig accepts custom values."""
    config = VisionConfig(
        model_path="/custom/path",
        vision_interval=10.0,
        max_new_tokens=128,
        jpeg_quality=95,
        max_retries=5,
        retry_delay=2.0,
        device_preference="cpu",
    )
    assert config.model_path == "/custom/path"
    assert config.vision_interval == 10.0
    assert config.max_new_tokens == 128
    assert config.jpeg_quality == 95
    assert config.max_retries == 5
    assert config.retry_delay == 2.0
    assert config.device_preference == "cpu"



@pytest.fixture
def mock_torch() -> Any:
    """Mock torch module to avoid loading actual models."""
    with patch("reachy_mini_conversation_app.vision.processors.torch") as mock:
        mock.cuda.is_available.return_value = False
        mock.backends.mps.is_available.return_value = False
        mock.float32 = "float32"
        mock.bfloat16 = "bfloat16"
        yield mock


@pytest.fixture
def mock_transformers() -> Any:
    """Mock transformers module."""
    with patch("reachy_mini_conversation_app.vision.processors.AutoProcessor") as proc, \
         patch("reachy_mini_conversation_app.vision.processors.AutoModelForImageTextToText") as model:

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = {
            "input_ids": MagicMock(to=lambda x: MagicMock()),
            "attention_mask": MagicMock(to=lambda x: MagicMock()),
            "pixel_values": MagicMock(to=lambda x: MagicMock()),
        }
        mock_processor.batch_decode.return_value = ["assistant\nThis is a test description."]
        mock_processor.tokenizer.eos_token_id = 2
        proc.from_pretrained.return_value = mock_processor

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = None
        mock_model_instance.generate.return_value = [[1, 2, 3]]
        mock_model_instance.to.return_value = mock_model_instance
        model.from_pretrained.return_value = mock_model_instance

        yield {"processor": proc, "model": model}


def test_vision_processor_device_selection_cpu(mock_torch: Any) -> None:
    """Test VisionProcessor selects CPU when specified."""
    config = VisionConfig(device_preference="cpu")
    processor = VisionProcessor(config)
    assert processor.device == "cpu"


def test_vision_processor_device_selection_cuda_unavailable(mock_torch: Any) -> None:
    """Test VisionProcessor falls back to CPU when CUDA unavailable."""
    mock_torch.cuda.is_available.return_value = False
    config = VisionConfig(device_preference="cuda")
    processor = VisionProcessor(config)
    assert processor.device == "cpu"


def test_vision_processor_device_selection_cuda_available(mock_torch: Any) -> None:
    """Test VisionProcessor selects CUDA when available."""
    mock_torch.cuda.is_available.return_value = True
    config = VisionConfig(device_preference="cuda")
    processor = VisionProcessor(config)
    assert processor.device == "cuda"


def test_vision_processor_device_selection_mps_available(mock_torch: Any) -> None:
    """Test VisionProcessor selects MPS when available on Apple Silicon."""
    mock_torch.backends.mps.is_available.return_value = True
    config = VisionConfig(device_preference="mps")
    processor = VisionProcessor(config)
    assert processor.device == "mps"


def test_vision_processor_device_selection_auto_prefers_mps(mock_torch: Any) -> None:
    """Test VisionProcessor auto mode prefers MPS on Apple Silicon."""
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.cuda.is_available.return_value = False
    config = VisionConfig(device_preference="auto")
    processor = VisionProcessor(config)
    assert processor.device == "mps"


def test_vision_processor_device_selection_auto_prefers_cuda_over_cpu(mock_torch: Any) -> None:
    """Test VisionProcessor auto mode prefers CUDA over CPU."""
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.cuda.is_available.return_value = True
    config = VisionConfig(device_preference="auto")
    processor = VisionProcessor(config)
    assert processor.device == "cuda"


def test_vision_processor_initialization(mock_torch: Any, mock_transformers: Any) -> None:
    """Test VisionProcessor initializes successfully."""
    config = VisionConfig(model_path="test/model")
    processor = VisionProcessor(config)

    assert not processor._initialized
    result = processor.initialize()

    assert result is True
    assert processor._initialized
    mock_transformers["processor"].from_pretrained.assert_called_once_with("test/model")
    mock_transformers["model"].from_pretrained.assert_called_once()


def test_vision_processor_initialization_failure(mock_torch: Any) -> None:
    """Test VisionProcessor handles initialization failure gracefully."""
    with patch("reachy_mini_conversation_app.vision.processors.AutoProcessor") as mock_proc:
        mock_proc.from_pretrained.side_effect = Exception("Model not found")

        config = VisionConfig(model_path="invalid/model")
        processor = VisionProcessor(config)
        result = processor.initialize()

        assert result is False
        assert not processor._initialized


def test_vision_processor_process_image_not_initialized(mock_torch: Any) -> None:
    """Test process_image returns error when model not initialized."""
    processor = VisionProcessor()
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = processor.process_image(test_image)
    assert result == "Vision model not initialized"


def test_vision_processor_process_image_success(mock_torch: Any, mock_transformers: Any) -> None:
    """Test process_image processes an image successfully."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        # Mock cv2.imencode to return success
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        processor = VisionProcessor()
        processor.initialize()

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_image(test_image, "Describe this image.")

        assert isinstance(result, str)
        assert result == "This is a test description."


def test_vision_processor_process_image_encode_failure(mock_torch: Any, mock_transformers: Any) -> None:
    """Test process_image handles image encoding failure."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (False, None)
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        processor = VisionProcessor()
        processor.initialize()

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_image(test_image)

        assert result == "Failed to encode image"


def test_vision_processor_process_image_with_retry(mock_torch: Any, mock_transformers: Any) -> None:
    """Test process_image retries on failure."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        # Set up the OutOfMemoryError to be a proper exception
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})

        processor = VisionProcessor(VisionConfig(max_retries=3, retry_delay=0.01))
        processor.initialize()

        # Make the model generate fail twice, then succeed
        call_count = [0]
        assert processor.model is not None
        original_generate = processor.model.generate

        def failing_generate(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return original_generate(*args, **kwargs)

        processor.model.generate = failing_generate

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_image(test_image)

        assert isinstance(result, str)
        assert call_count[0] == 3


def test_vision_processor_extract_response_variants() -> None:
    """Test _extract_response handles different response formats."""
    processor = VisionProcessor()

    # Test with "assistant\n" marker
    result = processor._extract_response("user prompt\nassistant\nThe response text")
    assert result == "The response text"

    # Test with "Assistant:" marker
    result = processor._extract_response("User: prompt\nAssistant: Another response")
    assert result == "Another response"

    # Test fallback to full text
    result = processor._extract_response("Just some text without markers")
    assert result == "Just some text without markers"


def test_vision_processor_get_model_info(mock_torch: Any, mock_transformers: Any) -> None:
    """Test get_model_info returns correct information."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3

    processor = VisionProcessor(VisionConfig(model_path="test/model", device_preference="cpu"))
    processor.initialize()

    info = processor.get_model_info()

    assert info["initialized"] is True
    assert info["device"] == "cpu"
    assert info["model_path"] == "test/model"
    assert "cuda_available" in info


@pytest.fixture
def mock_camera() -> Mock:
    """Create a mock camera object."""
    camera = Mock()
    camera.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return camera


def test_vision_manager_initialization(mock_torch: Any, mock_transformers: Any, mock_camera: Mock) -> None:
    """Test VisionManager initializes successfully."""
    config = VisionConfig(vision_interval=2.0)
    manager = VisionManager(mock_camera, config)

    assert manager.vision_interval == 2.0
    assert manager.processor._initialized


def test_vision_manager_initialization_failure(mock_torch: Any, mock_camera: Mock) -> None:
    """Test VisionManager raises error when processor initialization fails."""
    with patch("reachy_mini_conversation_app.vision.processors.AutoProcessor") as mock_proc:
        mock_proc.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(RuntimeError, match="Vision processor initialization failed"):
            VisionManager(mock_camera, VisionConfig())


def test_vision_manager_start_stop(mock_torch: Any, mock_transformers: Any, mock_camera: Mock) -> None:
    """Test VisionManager can start and stop."""
    manager = VisionManager(mock_camera, VisionConfig())

    manager.start()
    assert manager._thread is not None
    assert manager._thread.is_alive()
    assert not manager._stop_event.is_set()

    time.sleep(0.1)  # Let thread run briefly

    manager.stop()
    assert manager._stop_event.is_set()
    assert not manager._thread.is_alive()


def test_vision_manager_processes_frames(mock_torch: Any, mock_transformers: Any, mock_camera: Mock) -> None:
    """Test VisionManager processes frames at intervals."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        config = VisionConfig(vision_interval=0.1)  # Fast interval for testing
        manager = VisionManager(mock_camera, config)

        manager.start()
        time.sleep(0.3)  # Wait for at least 2 processing cycles
        manager.stop()

        # Camera should have been called at least once
        assert mock_camera.get_latest_frame.call_count >= 1


def test_vision_manager_handles_none_frame(mock_torch: Any, mock_transformers: Any, mock_camera: Mock) -> None:
    """Test VisionManager handles None frame gracefully."""
    mock_camera.get_latest_frame.return_value = None

    config = VisionConfig(vision_interval=0.1)
    manager = VisionManager(mock_camera, config)

    manager.start()
    time.sleep(0.2)
    manager.stop()

    # Verify camera was called but no crashes occurred
    assert mock_camera.get_latest_frame.called


def test_vision_manager_handles_processing_error(mock_torch: Any, mock_transformers: Any, mock_camera: Mock) -> None:
    """Test VisionManager handles processing errors gracefully."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.side_effect = Exception("Processing error")
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        config = VisionConfig(vision_interval=0.1)
        manager = VisionManager(mock_camera, config)

        manager.start()
        time.sleep(0.2)
        manager.stop()

        # Verify thread stopped gracefully despite errors
        assert manager._stop_event.is_set()


def test_vision_manager_get_status(mock_torch: Any, mock_transformers: Any, mock_camera: Mock) -> None:
    """Test VisionManager get_status returns correct information."""
    manager = VisionManager(mock_camera, VisionConfig(vision_interval=5.0))

    status = manager.get_status()

    assert "last_processed" in status
    assert "processor_info" in status
    assert "config" in status
    assert status["config"]["interval"] == 5.0


def test_vision_manager_skips_invalid_responses(mock_torch: Any, mock_transformers: Any, mock_camera: Mock) -> None:
    """Test VisionManager doesn't update timestamp for invalid responses."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        # Make processor return invalid response
        config = VisionConfig(vision_interval=0.1)
        manager = VisionManager(mock_camera, config)

        # Mock the processor's process_image method to return invalid response
        with patch.object(manager.processor, 'process_image', return_value="Vision model not initialized"):
            initial_time = manager._last_processed_time

            manager.start()
            time.sleep(0.2)
            manager.stop()

            # Last processed time should not have been updated
            assert manager._last_processed_time == initial_time


def test_initialize_vision_manager_success(mock_torch: Any, mock_transformers: Any, mock_camera: Mock) -> None:
    """Test initialize_vision_manager creates VisionManager successfully."""
    with patch("reachy_mini_conversation_app.vision.processors.snapshot_download") as mock_download, \
         patch("reachy_mini_conversation_app.vision.processors.os.makedirs"), \
         patch("reachy_mini_conversation_app.vision.processors.config") as mock_config:

        mock_config.LOCAL_VISION_MODEL = "test/model"
        mock_config.HF_HOME = "/tmp/hf_cache"

        result = initialize_vision_manager(mock_camera)

        assert result is not None
        assert isinstance(result, VisionManager)
        mock_download.assert_called_once()


def test_initialize_vision_manager_download_failure(mock_torch: Any, mock_camera: Mock) -> None:
    """Test initialize_vision_manager handles download failure."""
    with patch("reachy_mini_conversation_app.vision.processors.snapshot_download") as mock_download, \
         patch("reachy_mini_conversation_app.vision.processors.os.makedirs"), \
         patch("reachy_mini_conversation_app.vision.processors.config") as mock_config:

        mock_config.LOCAL_VISION_MODEL = "test/model"
        mock_config.HF_HOME = "/tmp/hf_cache"
        mock_download.side_effect = Exception("Network error")

        result = initialize_vision_manager(mock_camera)

        assert result is None


def test_initialize_vision_manager_processor_failure(mock_torch: Any, mock_camera: Mock) -> None:
    """Test initialize_vision_manager handles processor initialization failure."""
    with patch("reachy_mini_conversation_app.vision.processors.snapshot_download"), \
         patch("reachy_mini_conversation_app.vision.processors.os.makedirs"), \
         patch("reachy_mini_conversation_app.vision.processors.config") as mock_config, \
         patch("reachy_mini_conversation_app.vision.processors.AutoProcessor") as mock_proc:

        mock_config.LOCAL_VISION_MODEL = "test/model"
        mock_config.HF_HOME = "/tmp/hf_cache"
        mock_proc.from_pretrained.side_effect = Exception("Model load error")

        result = initialize_vision_manager(mock_camera)

        assert result is None


def test_vision_processor_cuda_oom_recovery(mock_torch: Any, mock_transformers: Any) -> None:
    """Test VisionProcessor recovers from CUDA OOM errors."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        processor = VisionProcessor(VisionConfig(max_retries=2, retry_delay=0.01))
        processor.initialize()
        processor.device = "cuda"  # Force CUDA for this test

        # Make generate raise OOM error
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
        assert processor.model is not None
        processor.model.generate.side_effect = mock_torch.cuda.OutOfMemoryError("OOM")

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_image(test_image)

        assert "GPU out of memory" in result
        mock_torch.cuda.empty_cache.assert_called()


def test_vision_processor_cache_cleanup_mps(mock_torch: Any, mock_transformers: Any) -> None:
    """Test VisionProcessor cleans up MPS cache after processing."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        processor = VisionProcessor()
        processor.initialize()
        processor.device = "mps"  # Force MPS for this test

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        processor.process_image(test_image)

        # Should call mps empty_cache
        mock_torch.mps.empty_cache.assert_called()


def test_vision_manager_thread_safety(mock_torch: Any, mock_transformers: Any, mock_camera: Mock) -> None:
    """Test VisionManager thread safety with multiple start/stop cycles."""
    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        config = VisionConfig(vision_interval=0.05)
        manager = VisionManager(mock_camera, config)

        # Multiple start/stop cycles
        for _ in range(3):
            manager.start()
            time.sleep(0.1)
            manager.stop()
            time.sleep(0.05)

        # Should not crash or leave dangling threads
        assert manager._stop_event.is_set()
