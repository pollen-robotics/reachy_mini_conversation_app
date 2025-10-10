import os
import sys
import time
import base64
import asyncio
import logging
import threading
from typing import Any, Dict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import snapshot_download


logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    """Configuration for vision processing."""

    processor_type: str = "local"
    model_path: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    vision_interval: float = 5.0
    max_new_tokens: int = 64
    temperature: float = 0.7
    jpeg_quality: int = 85
    max_retries: int = 3
    retry_delay: float = 1.0
    device_preference: str = "auto"  # "auto", "cuda", "cpu"


class VisionProcessor:
    """Handles SmolVLM2 model loading and inference."""

    def __init__(self, config: VisionConfig = None):
        """Initialize the vision processor."""
        self.config = config or VisionConfig()
        self.model_path = self.config.model_path
        self.device = self._determine_device()
        self.processor = None
        self.model = None
        self._initialized = False

    def _determine_device(self) -> str:
        pref = self.config.device_preference
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
            logger.info(f"Loading SmolVLM2 model on {self.device} (HF_HOME={os.getenv('HF_HOME')})")
            self.processor = AutoProcessor.from_pretrained(self.model_path)

            # Select dtype depending on device
            if self.device == "cuda":
                dtype = torch.bfloat16
            elif self.device == "mps":
                dtype = torch.float32  # best for MPS
            else:
                dtype = torch.float32

            model_kwargs = {"dtype": dtype}

            # flash_attention_2 is CUDA-only; skip on MPS/CPU
            if self.device == "cuda":
                model_kwargs["_attn_implementation"] = "flash_attention_2"

            # Load model weights
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs).to(self.device)

            self.model.eval()
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            return False

    def process_image(
        self,
        cv2_image: np.ndarray,
        prompt: str = "Briefly describe what you see in one sentence.",
    ) -> str:
        """Process CV2 image and return description with retry logic."""
        if not self._initialized:
            return "Vision model not initialized"

        for attempt in range(self.config.max_retries):
            try:
                # Convert to JPEG bytes
                success, jpeg_buffer = cv2.imencode(
                    ".jpg",
                    cv2_image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
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
                        max_new_tokens=self.config.max_new_tokens,
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
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return "GPU out of memory - vision processing failed"

            except Exception as e:
                logger.error(f"Vision processing failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return f"Vision processing error after {self.config.max_retries} attempts"

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
            "processor_type": "local",
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

    def __init__(self, camera, config: VisionConfig = None):
        """Initialize vision manager with camera and configuration."""
        self.camera = camera
        self.config = config or VisionConfig()
        self.vision_interval = self.config.vision_interval
        self.processor = create_vision_processor(self.config)  # Use factory function

        self._current_description = ""
        self._last_processed_time = 0

        # Initialize processor
        if not self.processor.initialize():
            logger.error("Failed to initialize vision processor")
            raise RuntimeError("Vision processor initialization failed")

    async def enable(self, stop_event: threading.Event):
        """Vision processing loop (runs in separate thread)."""
        while not stop_event.is_set():
            try:
                current_time = time.time()

                if current_time - self._last_processed_time >= self.vision_interval:
                    success, frame = await asyncio.to_thread(self.camera.read)
                    if success and frame is not None:
                        description = await asyncio.to_thread(
                            lambda: self.processor.process_image(
                                frame, "Briefly describe what you see in one sentence."
                            )
                        )

                        # Only update if we got a valid response
                        if description and not description.startswith(("Vision", "Failed", "Error")):
                            self._current_description = description
                            self._last_processed_time = current_time

                            logger.info(f"Vision update: {description}")
                        else:
                            logger.warning(f"Invalid vision response: {description}")

                await asyncio.sleep(1.0)  # Check every second

            except Exception:
                logger.exception("Vision processing loop error")
                await asyncio.sleep(5.0)  # Longer sleep on error

        logger.info("Vision loop finished")

    async def get_current_description(self) -> str:
        """Get the most recent scene description (thread-safe)."""
        return self._current_description

    async def process_current_frame(self, prompt: str = "Describe what you see in detail.") -> Dict[str, Any]:
        """Process current camera frame with custom prompt."""
        try:
            success, frame = self.camera.read()
            if not success or frame is None:
                return {"error": "Failed to capture image from camera"}

            description = await asyncio.to_thread(lambda: self.processor.process_image(frame, prompt))

            return {
                "description": description,
                "timestamp": time.time(),
                "prompt": prompt,
            }

        except Exception as e:
            logger.exception("Failed to process current frame")
            return {"error": f"Frame processing failed: {str(e)}"}

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        return {
            "last_processed": self._last_processed_time,
            "processor_info": self.processor.get_model_info(),
            "config": {
                "interval": self.vision_interval,
                "processor_type": self.config.processor_type,
            },
        }


def init_camera(camera_index=0, simulation=True):
    """Initialize camera (real or simulated)."""
    api_preference = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else 0

    if simulation:
        # Default build-in camera in SIM
        # TODO: please, test on Linux and Windows
        camera = cv2.VideoCapture(0, api_preference)
    else:
        # TODO handle macos properly
        if sys.platform == "darwin":
            camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        else:
            camera = cv2.VideoCapture(camera_index)

    return camera


def create_vision_processor(config: VisionConfig):
    """Create the appropriate vision processor (factory)."""
    if config.processor_type == "openai":
        try:
            from .openai_vision import OpenAIVisionProcessor

            return OpenAIVisionProcessor(config)
        except ImportError:
            logger.error("OpenAI vision processor not available, falling back to local")
            return VisionProcessor(config)
    else:
        return VisionProcessor(config)


def init_vision(camera: cv2.VideoCapture, processor_type: str = "local") -> VisionManager:
    """Initialize vision manager with the specified processor type."""
    model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    cache_dir = os.path.expandvars(os.getenv("HF_HOME", "$HOME/.cache/huggingface"))

    # Only download model if using local processor
    if processor_type == "local":
        try:
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = cache_dir
            logger.info("HF_HOME set to %s", cache_dir)
        except Exception as e:
            logger.warning("Failed to prepare HF cache dir %s: %s", cache_dir, e)
            return None

        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            cache_dir=cache_dir,
        )
        logger.info(f"Prefetched model_id={model_id} into cache_dir={cache_dir}")

    # Configure vision processing
    vision_config = VisionConfig(
        processor_type=processor_type,
        model_path=model_id,
        vision_interval=5.0,
        max_new_tokens=64,
        temperature=0.7,
        jpeg_quality=85,
        max_retries=3,
        retry_delay=1.0,
        device_preference="auto",
    )

    vision_manager = VisionManager(camera, vision_config)

    device_info = vision_manager.processor.get_model_info()
    logger.info(
        f"Vision processing enabled: {device_info.get('model_path', device_info.get('processor_type'))} on {device_info.get('device', 'API')}",
    )

    return vision_manager
