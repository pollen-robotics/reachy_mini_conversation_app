from __future__ import annotations
import os
import time
import logging
from typing import Any, Protocol, cast
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from numpy.typing import NDArray
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import snapshot_download

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

LOCAL_VISION_RESPONSE_INSTRUCTIONS = (
    "Respond to the request using only details that are clearly visible in the image. "
    "Do not guess, infer hidden details, or invent missing information. "
    "If the answer is not clearly visible, say exactly: I can't tell from this image. "
    "Keep the answer short and factual."
)


class _VisionInputs(Mapping[str, object]):
    """Tokenized processor inputs that can be moved to the inference device."""

    def to(self, device: str) -> _VisionInputs:
        """Move inputs to the selected inference device."""
        raise NotImplementedError


class _VisionTokenizer(Protocol):
    """Tokenizer attributes used by generation."""

    eos_token_id: int | None


class _VisionProcessor(Protocol):
    """Small interface required from the Hugging Face image-text processor."""

    tokenizer: _VisionTokenizer

    def apply_chat_template(
        self,
        conversation: object,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool,
        return_tensors: str,
    ) -> _VisionInputs: ...

    def batch_decode(self, sequences: object, *, skip_special_tokens: bool) -> list[str]: ...


class _VisionModel(Protocol):
    """Small interface required from the Hugging Face image-text model."""

    def to(self, device: str) -> _VisionModel: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> Any: ...


@dataclass
class VisionConfig:
    """Configuration for vision processing."""

    model_path: str = config.LOCAL_VISION_MODEL
    max_new_tokens: int = 64
    max_retries: int = 3
    retry_delay: float = 1.0
    device_preference: str = "auto"  # "auto", "cuda", "mps", "cpu"


class VisionProcessor:
    """Handles SmolVLM2 model loading and inference."""

    def __init__(self, vision_config: VisionConfig | None = None):
        """Initialize the vision processor."""
        self.vision_config = vision_config or VisionConfig()
        self.device = self._determine_device()
        self.processor: _VisionProcessor | None = None
        self.model: _VisionModel | None = None
        self._initialized = False

    def _determine_device(self) -> str:
        """Choose the execution device from the configured preference."""
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

    def initialize(self) -> None:
        """Load model and processor onto the selected device."""
        logger.info("Loading SmolVLM2 model on %s (HF_HOME=%s)", self.device, config.HF_HOME)
        processor = cast(
            _VisionProcessor,
            AutoProcessor.from_pretrained(self.vision_config.model_path),  # type: ignore[no-untyped-call]
        )

        model_kwargs: dict[str, object] = {
            "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
        }

        model = cast(
            _VisionModel,
            AutoModelForImageTextToText.from_pretrained(
                self.vision_config.model_path,
                **model_kwargs,
            ),
        )
        model = model.to(self.device)

        model.eval()
        self.processor = processor
        self.model = model
        self._initialized = True

    def process_image(
        self,
        frame: NDArray[np.uint8],
        prompt: str,
    ) -> str:
        """Process a BGR camera frame and return a text description."""
        prompt_text = prompt.strip()
        if not prompt_text:
            raise ValueError("prompt must be a non-empty string")

        if not self._initialized or self.processor is None or self.model is None:
            return "Vision model not initialized"

        processor = self.processor
        model = self.model
        rgb_image = Image.fromarray(np.ascontiguousarray(frame[..., ::-1]))
        request_parts = [LOCAL_VISION_RESPONSE_INSTRUCTIONS]
        request_parts.insert(0, prompt_text)
        request = "\n\n".join(request_parts)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": request},
                ],
            },
        ]

        for attempt in range(self.vision_config.max_retries):
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)
                prompt_len = _last_shape_dim(inputs.get("input_ids"))

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self.vision_config.max_new_tokens,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )

                # Decode only the newly generated tokens, skipping the prompt
                if prompt_len is None:
                    new_token_ids = generated_ids
                elif getattr(generated_ids, "shape", None) is not None:
                    new_token_ids = generated_ids[:, prompt_len:]
                else:
                    new_token_ids = [token_ids[prompt_len:] for token_ids in generated_ids]
                response = processor.batch_decode(
                    new_token_ids,
                    skip_special_tokens=True,
                )[0]

                return str(response).replace("\n", " ").strip()

            except Exception as e:
                oom_error = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
                if isinstance(oom_error, type) and issubclass(oom_error, BaseException) and isinstance(e, oom_error):
                    logger.error(f"CUDA OOM on attempt {attempt + 1}: {e}")
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    if attempt < self.vision_config.max_retries - 1:
                        time.sleep(self.vision_config.retry_delay * (attempt + 1))
                        continue
                    return "GPU out of memory - vision processing failed"

                logger.error("Vision processing failed (attempt %s): %s", attempt + 1, e)
                if attempt < self.vision_config.max_retries - 1:
                    time.sleep(self.vision_config.retry_delay)
                else:
                    return f"Vision processing error after {self.vision_config.max_retries} attempts"

        return f"Vision processing error after {self.vision_config.max_retries} attempts"


def _last_shape_dim(value: object) -> int | None:
    """Return the last dimension from tensor-like objects that expose a shape."""
    shape = getattr(value, "shape", None)
    if not isinstance(shape, Sequence) or not shape:
        return None
    return int(shape[-1])


def initialize_vision_processor() -> VisionProcessor:
    """Download the vision model and return an initialized VisionProcessor."""
    try:
        model_id = config.LOCAL_VISION_MODEL
        cache_dir = os.path.expanduser(config.HF_HOME)

        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        logger.info("HF_HOME set to %s", cache_dir)

        logger.info("Downloading vision model %s to cache...", model_id)
        snapshot_download(repo_id=model_id, repo_type="model", cache_dir=cache_dir)

        vision_processor = VisionProcessor()
        vision_processor.initialize()

        logger.info(
            "Vision processing enabled: %s on %s",
            vision_processor.vision_config.model_path,
            vision_processor.device,
        )

        return vision_processor
    except Exception:
        logger.exception("Failed to initialize vision processor")
        raise
