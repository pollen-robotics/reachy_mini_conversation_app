"""Tests for the vision processing module."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.vision.local_vision import (
    LOCAL_VISION_RESPONSE_INSTRUCTIONS,
    VisionConfig,
    VisionProcessor,
    initialize_vision_processor,
)


def test_vision_config_defaults() -> None:
    """Test VisionConfig has sensible defaults."""
    config = VisionConfig()
    assert config.max_new_tokens == 64
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert config.device_preference == "auto"


def test_vision_config_custom_values() -> None:
    """Test VisionConfig accepts custom values."""
    config = VisionConfig(
        model_path="/custom/path",
        max_new_tokens=128,
        max_retries=5,
        retry_delay=2.0,
        device_preference="cpu",
    )
    assert config.model_path == "/custom/path"
    assert config.max_new_tokens == 128
    assert config.max_retries == 5
    assert config.retry_delay == 2.0
    assert config.device_preference == "cpu"



@pytest.fixture
def mock_torch() -> Any:
    """Mock torch module to avoid loading actual models."""
    with patch("reachy_mini_conversation_app.vision.local_vision.torch") as mock:
        mock.cuda.is_available.return_value = False
        mock.backends.mps.is_available.return_value = False
        mock.float32 = "float32"
        mock.bfloat16 = "bfloat16"
        yield mock


@pytest.fixture
def mock_transformers() -> Any:
    """Mock transformers module."""
    with patch("reachy_mini_conversation_app.vision.local_vision.AutoProcessor") as proc, \
         patch("reachy_mini_conversation_app.vision.local_vision.AutoModelForImageTextToText") as model:

        # Mock processor — apply_chat_template returns a BatchFeature-like object with .to()
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_batch
        mock_input_ids = MagicMock()
        mock_input_ids.shape = (1, 3)
        mock_batch.get.side_effect = lambda key, default=None: mock_input_ids if key == "input_ids" else default

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = mock_batch
        mock_processor.batch_decode.return_value = ["This is a test description."]
        mock_processor.tokenizer.eos_token_id = 2
        proc.from_pretrained.return_value = mock_processor

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None
        mock_model_instance.generate.return_value = [[1, 2, 3]]
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
    processor.initialize()

    assert processor._initialized
    mock_transformers["processor"].from_pretrained.assert_called_once_with("test/model")
    mock_transformers["model"].from_pretrained.assert_called_once_with(
        "test/model",
        dtype="float32",
    )


def test_vision_processor_initialization_cuda(mock_torch: Any, mock_transformers: Any) -> None:
    """Test CUDA initialization uses bfloat16 without extra attention wiring."""
    mock_torch.cuda.is_available.return_value = True

    processor = VisionProcessor(VisionConfig(model_path="test/model", device_preference="cuda"))
    processor.initialize()

    mock_transformers["model"].from_pretrained.assert_called_once_with(
        "test/model",
        dtype="bfloat16",
    )


def test_vision_processor_initialization_failure(mock_torch: Any) -> None:
    """Test VisionProcessor surfaces initialization failures."""
    with patch("reachy_mini_conversation_app.vision.local_vision.AutoProcessor") as mock_proc:
        mock_proc.from_pretrained.side_effect = Exception("Model not found")

        config = VisionConfig(model_path="invalid/model")
        processor = VisionProcessor(config)
        with pytest.raises(Exception, match="Model not found"):
            processor.initialize()
        assert not processor._initialized


def test_vision_processor_process_image_not_initialized(mock_torch: Any) -> None:
    """Test process_image returns error when model not initialized."""
    processor = VisionProcessor()
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = processor.process_image(test_image, "Describe this image.")
    assert result == "Vision model not initialized"


def test_vision_processor_process_image_rejects_blank_prompt(
    mock_torch: Any,
    mock_transformers: Any,
) -> None:
    """Test process_image requires a non-empty prompt."""
    processor = VisionProcessor()
    processor.initialize()

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        processor.process_image(test_image, "   ")


def test_vision_processor_process_image_success(mock_torch: Any, mock_transformers: Any) -> None:
    """Test process_image processes an image successfully."""
    processor = VisionProcessor()
    processor.initialize()

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.process_image(test_image, "Describe this image.")

    assert isinstance(result, str)
    assert result == "This is a test description."


def test_vision_processor_process_image_appends_local_vision_instructions(
    mock_torch: Any,
    mock_transformers: Any,
) -> None:
    """Test process_image preserves the question and appends response instructions."""
    processor = VisionProcessor()
    processor.initialize()

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    processor.process_image(test_image, "What is on the table?")

    processor_mock = mock_transformers["processor"].from_pretrained.return_value
    messages = processor_mock.apply_chat_template.call_args.args[0]
    assert messages[0]["content"][1]["text"] == (
        "What is on the table?\n\n"
        f"{LOCAL_VISION_RESPONSE_INSTRUCTIONS}"
    )


def test_vision_processor_process_image_with_retry(mock_torch: Any, mock_transformers: Any) -> None:
    """Test process_image retries on failure."""
    # Set up the OutOfMemoryError to be a proper exception
    mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})

    processor = VisionProcessor(VisionConfig(max_retries=3, retry_delay=0.01))
    processor.initialize()

    # Make the model generate fail twice, then succeed
    call_count = [0]
    model_mock = mock_transformers["model"].from_pretrained.return_value
    original_generate = model_mock.generate

    def failing_generate(*args: Any, **kwargs: Any) -> Any:
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("Temporary failure")
        return original_generate(*args, **kwargs)

    model_mock.generate = failing_generate

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.process_image(test_image, "Describe this image.")

    assert isinstance(result, str)
    assert call_count[0] == 3


def test_vision_processor_process_image_retries_input_transfer_failure(
    mock_torch: Any,
    mock_transformers: Any,
) -> None:
    """Test process_image retries when input transfer fails before generation."""
    processor = VisionProcessor(VisionConfig(max_retries=2, retry_delay=0.01))
    processor.initialize()

    batch_mock = mock_transformers["processor"].from_pretrained.return_value.apply_chat_template.return_value
    batch_mock.to.side_effect = [Exception("Temporary transfer failure"), batch_mock]

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.process_image(test_image, "Describe this image.")

    assert result == "This is a test description."
    assert batch_mock.to.call_count == 2


def test_initialize_vision_processor_success(mock_torch: Any, mock_transformers: Any) -> None:
    """Test initialize_vision_processor creates VisionProcessor successfully."""
    with patch("reachy_mini_conversation_app.vision.local_vision.snapshot_download") as mock_download, \
         patch("reachy_mini_conversation_app.vision.local_vision.os.makedirs"), \
         patch("reachy_mini_conversation_app.vision.local_vision.config") as mock_config:

        mock_config.LOCAL_VISION_MODEL = "test/model"
        mock_config.HF_HOME = "/tmp/hf_cache"

        result = initialize_vision_processor()

        assert isinstance(result, VisionProcessor)
        assert result._initialized
        mock_download.assert_called_once()

def test_initialize_vision_processor_download_failure(mock_torch: Any) -> None:
    """Test initialize_vision_processor surfaces download failures."""
    with patch("reachy_mini_conversation_app.vision.local_vision.snapshot_download") as mock_download, \
         patch("reachy_mini_conversation_app.vision.local_vision.os.makedirs"), \
         patch("reachy_mini_conversation_app.vision.local_vision.config") as mock_config:

        mock_config.LOCAL_VISION_MODEL = "test/model"
        mock_config.HF_HOME = "/tmp/hf_cache"
        mock_download.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            initialize_vision_processor()

def test_initialize_vision_processor_processor_failure(mock_torch: Any) -> None:
    """Test initialize_vision_processor surfaces processor initialization failures."""
    with patch("reachy_mini_conversation_app.vision.local_vision.snapshot_download"), \
         patch("reachy_mini_conversation_app.vision.local_vision.os.makedirs"), \
         patch("reachy_mini_conversation_app.vision.local_vision.config") as mock_config, \
         patch("reachy_mini_conversation_app.vision.local_vision.AutoProcessor") as mock_proc:

        mock_config.LOCAL_VISION_MODEL = "test/model"
        mock_config.HF_HOME = "/tmp/hf_cache"
        mock_proc.from_pretrained.side_effect = Exception("Model load error")

        with pytest.raises(Exception, match="Model load error"):
            initialize_vision_processor()


def test_vision_processor_cuda_oom_recovery(mock_torch: Any, mock_transformers: Any) -> None:
    """Test VisionProcessor recovers from CUDA OOM errors."""
    processor = VisionProcessor(VisionConfig(max_retries=2, retry_delay=0.01))
    processor.initialize()
    processor.device = "cuda"  # Force CUDA for this test

    # Make generate raise OOM error
    mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
    mock_transformers["model"].from_pretrained.return_value.generate.side_effect = mock_torch.cuda.OutOfMemoryError("OOM")

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.process_image(test_image, "Describe this image.")

    assert "GPU out of memory" in result
    mock_torch.cuda.empty_cache.assert_called()


def test_vision_processor_cache_cleanup_mps(mock_torch: Any, mock_transformers: Any) -> None:
    """Test VisionProcessor does not call empty_cache on the happy path."""
    processor = VisionProcessor()
    processor.initialize()
    processor.device = "mps"  # Force MPS for this test

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    processor.process_image(test_image, "Describe this image.")

    # empty_cache should NOT be called on the happy path
    mock_torch.mps.empty_cache.assert_not_called()
