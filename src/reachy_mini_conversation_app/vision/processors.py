import os
import time
import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from numpy.typing import NDArray
from transformers import AutoProcessor, ProcessorMixin, AutoModelForImageTextToText
from huggingface_hub import snapshot_download

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

LOCAL_VISION_RESPONSE_INSTRUCTIONS = (
    "Respond to the request using only details that are clearly visible in the image. "
    "Do not guess, infer hidden details, or invent missing information. "
    "If the answer is not clearly visible, say exactly: I can't tell from this image. "
    "Keep the answer short and factual."
)


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
        self.processor: ProcessorMixin | None = None
        self.model: torch.nn.Module | None = None
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
        processor: ProcessorMixin = AutoProcessor.from_pretrained(self.vision_config.model_path)  # type: ignore[no-untyped-call]

        model_kwargs: dict[str, object] = {
            "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
        }

        model: torch.nn.Module = AutoModelForImageTextToText.from_pretrained(
            self.vision_config.model_path,
            **model_kwargs,
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
                    messages,  # type: ignore[arg-type]
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)  # type: ignore[attr-defined]
                prompt_len = None
                input_ids = inputs.get("input_ids")
                input_shape = getattr(input_ids, "shape", None)
                if input_shape:
                    prompt_len = int(input_shape[-1])

                with torch.inference_mode():
                    generated_ids = model.generate(  # type: ignore[operator]
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self.vision_config.max_new_tokens,
                        pad_token_id=processor.tokenizer.eos_token_id,  # type: ignore[attr-defined]
                    )

                # Decode only the newly generated tokens, skipping the prompt
                if prompt_len is None:
                    new_token_ids = generated_ids
                elif getattr(generated_ids, "shape", None) is not None:
                    new_token_ids = generated_ids[:, prompt_len:]
                else:
                    new_token_ids = [token_ids[prompt_len:] for token_ids in generated_ids]
                response = processor.batch_decode(  # type: ignore[no-untyped-call]
                    new_token_ids,
                    skip_special_tokens=True,
                )[0]

                return str(response).replace("\n", " ").strip()

            except Exception as e:
                oom_error = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
                if (
                    isinstance(oom_error, type)
                    and issubclass(oom_error, BaseException)
                    and isinstance(e, oom_error)
                ):
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
