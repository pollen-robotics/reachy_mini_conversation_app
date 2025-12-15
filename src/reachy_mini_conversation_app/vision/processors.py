import os
import time
import base64
import logging
import threading
from typing import Any, Dict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import snapshot_download

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    """Configuration for vision processing."""

    model_path: str = config.LOCAL_VISION_MODEL
    vision_interval: float = 5.0
    max_new_tokens: int = 64
    jpeg_quality: int = 85
    max_retries: int = 3
    retry_delay: float = 1.0
    device_preference: str = "auto"  # "auto", "cuda", "cpu"


class VisionProcessor:
    """Handles SmolVLM2 model loading and inference."""

    def __init__(self, vision_config: VisionConfig | None = None):
        """Initialize the vision processor."""
        self.vision_config = vision_config or VisionConfig()
        self.model_path = self.vision_config.model_path
        self.device = self._determine_device()
        self.processor = None
        self.model = None
        self._initialized = False

    def _determine_device(self) -> str:
        pref = self.vision_config.device_preference
        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if pref == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        # auto: prefer mps on Apple, then cuda, else cpu
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def initialize(self) -> bool:
        """Load model and processor onto the selected device."""
        try:
            logger.info(f"Loading SmolVLM2 model on {self.device} (HF_HOME={config.HF_HOME})")
            self.processor = AutoProcessor.from_pretrained(self.model_path)  # type: ignore

            # Select dtype depending on device
            if self.device == "cuda":
                dtype = torch.bfloat16
            elif self.device == "mps":
                dtype = torch.float32  # best for MPS
            else:
                dtype = torch.float32

            model_kwargs: Dict[str, Any] = {"dtype": dtype}

            # flash_attention_2 is CUDA-only; skip on MPS/CPU
            if self.device == "cuda":
                model_kwargs["_attn_implementation"] = "flash_attention_2"

            # Load model weights
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs).to(self.device)  # type: ignore

            if self.model is not None:
                self.model.eval()
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            return False

    def process_image(
        self,
        cv2_image: NDArray[np.uint8],
        prompt: str = "Briefly describe what you see in one sentence.",
    ) -> str:
        """Process CV2 image and return description with retry logic."""
        if not self._initialized or self.processor is None or self.model is None:
            return "Vision model not initialized"

        for attempt in range(self.vision_config.max_retries):
            try:
                # Convert to JPEG bytes
                success, jpeg_buffer = cv2.imencode(
                    ".jpg",
                    cv2_image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.vision_config.jpeg_quality],
                )
                if not success:
                    return "Failed to encode image"

                # Convert to base64
                image_base64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]

                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                # Move tensors to device WITHOUT forcing dtype (keeps input_ids as torch.long)
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self.vision_config.max_new_tokens,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )

                generated_texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                # Extract just the response part
                full_text = generated_texts[0]
                response = self._extract_response(full_text)

                # Clean up GPU memory if using CUDA
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()

                return response.replace(chr(10), " ").strip()

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM on attempt {attempt + 1}: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if attempt < self.vision_config.max_retries - 1:
                    time.sleep(self.vision_config.retry_delay * (attempt + 1))
                else:
                    return "GPU out of memory - vision processing failed"

            except Exception as e:
                logger.error(f"Vision processing failed (attempt {attempt + 1}): {e}")
                if attempt < self.vision_config.max_retries - 1:
                    time.sleep(self.vision_config.retry_delay)
                else:
                    return f"Vision processing error after {self.vision_config.max_retries} attempts"

    def _extract_response(self, full_text: str) -> str:
        """Extract the assistant's response from the full generated text."""
        # Handle different response formats
        markers = ["assistant\n", "Assistant:", "Response:", "\n\n"]

        for marker in markers:
            if marker in full_text:
                response = full_text.split(marker)[-1].strip()
                if response:  # Ensure we got a meaningful response
                    return response

        # Fallback: return the full text cleaned up
        return full_text.strip()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "initialized": self._initialized,
            "device": self.device,
            "model_path": self.model_path,
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory // (1024**3)
            if torch.cuda.is_available()
            else "N/A",
        }


class VisionManager:
    """Manages periodic vision processing and scene understanding."""

    def __init__(self, camera: Any, vision_config: VisionConfig | None = None):
        """Initialize vision manager with camera and configuration."""
        self.camera = camera
        self.vision_config = vision_config or VisionConfig()
        self.vision_interval = self.vision_config.vision_interval
        self.processor = VisionProcessor(self.vision_config)

        self._last_processed_time = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Initialize processor
        if not self.processor.initialize():
            logger.error("Failed to initialize vision processor")
            raise RuntimeError("Vision processor initialization failed")

    def start(self) -> None:
        """Start the vision processing loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._working_loop, daemon=True)
        self._thread.start()
        logger.info("Local vision processing started")

    def stop(self) -> None:
        """Stop the vision processing loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.info("Local vision processing stopped")

    def _working_loop(self) -> None:
        """Vision processing loop (runs in separate thread)."""
        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                if current_time - self._last_processed_time >= self.vision_interval:
                    frame = self.camera.get_latest_frame()
                    if frame is not None:
                        description = self.processor.process_image(
                            frame,
                            "Briefly describe what you see in one sentence.",
                        )

                        # Only update if we got a valid response
                        if description and not description.startswith(("Vision", "Failed", "Error")):
                            self._last_processed_time = current_time
                            logger.debug(f"Vision update: {description}")
                        else:
                            logger.warning(f"Invalid vision response: {description}")

                time.sleep(1.0)  # Check every second

            except Exception:
                logger.exception("Vision processing loop error")
                time.sleep(5.0)  # Longer sleep on error

        logger.info("Vision loop finished")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        return {
            "last_processed": self._last_processed_time,
            "processor_info": self.processor.get_model_info(),
            "config": {
                "interval": self.vision_interval,
            },
        }


def initialize_vision_manager(camera_worker: Any) -> VisionManager | None:
    """Initialize vision manager with model download and configuration.

    Args:
        camera_worker: CameraWorker instance for frame capture
    Returns:
        VisionManager instance or None if initialization fails

    """
    try:
        model_id = config.LOCAL_VISION_MODEL
        cache_dir = os.path.expanduser(config.HF_HOME)

        # Prepare cache directory
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        logger.info("HF_HOME set to %s", cache_dir)

        # Download model to cache
        logger.info(f"Downloading vision model {model_id} to cache...")
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            cache_dir=cache_dir,
        )
        logger.info(f"Model {model_id} downloaded to {cache_dir}")

        # Configure vision processing
        vision_config = VisionConfig(
            model_path=model_id,
            vision_interval=5.0,
            max_new_tokens=64,
            jpeg_quality=85,
            max_retries=3,
            retry_delay=1.0,
            device_preference="auto",
        )

        # Initialize vision manager
        vision_manager = VisionManager(camera_worker, vision_config)

        # Log device info
        device_info = vision_manager.processor.get_model_info()
        logger.info(
            f"Vision processing enabled: {device_info.get('model_path')} on {device_info.get('device')}",
        )

        return vision_manager

    except Exception as e:
        logger.error(f"Failed to initialize vision manager: {e}")
        return None
