"""Local wake-word detection for standby mode.

While the app is in standby the realtime backend connection is closed, so no
audio leaves the robot. Mic frames are routed here instead: they are resampled
to 16 kHz, chunked, and scored by an openWakeWord model entirely on-device.
Frames are never buffered beyond the ~80 ms analysis window and never stored.

openwakeword is an optional dependency (``pip install
reachy-mini-conversation-app[wakeword]``); standby mode refuses to engage
without it rather than leaving the robot unwakeable by voice.
"""

from __future__ import annotations
import logging
from math import gcd
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.streaming import audio_to_float32


logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
# openWakeWord scores fixed 80 ms windows (1280 samples @ 16 kHz).
CHUNK_SAMPLES = 1280


class WakeWordDetector:
    """Streaming wrapper around an openWakeWord model.

    Feed arbitrary-size mic frames via :meth:`process`; it returns ``True``
    once the wake phrase is detected, after which internal state is reset so
    the detector can be reused for the next standby period.
    """

    def __init__(self, model_name: str, threshold: float) -> None:
        """Load ``model_name`` (a pretrained name or a path to an .onnx file)."""
        # Deferred import: openwakeword is an optional extra.
        from openwakeword.model import Model as OwwModel

        self._model = OwwModel(wakeword_models=[model_name], inference_framework="onnx")
        self._threshold = threshold
        self._buffer: NDArray[np.int16] = np.empty(0, dtype=np.int16)
        logger.info(
            "Wake word detector ready (model=%s, threshold=%.2f)", model_name, threshold
        )

    def reset(self) -> None:
        """Drop buffered audio and model state (call when leaving standby)."""
        self._buffer = np.empty(0, dtype=np.int16)
        try:
            self._model.reset()
        except Exception:
            logger.debug("openwakeword model reset failed (ignored)", exc_info=True)

    def _to_mono_16k_int16(self, sample_rate: int, frame: Any) -> NDArray[np.int16]:
        """Convert one mic frame to mono 16 kHz int16 samples."""
        samples = audio_to_float32(frame)
        if samples.ndim == 2:
            # channels-last convention, mirror LocalStream.play_loop
            if samples.shape[1] > samples.shape[0]:
                samples = samples.T
            samples = samples.mean(axis=1)
        if sample_rate != TARGET_SAMPLE_RATE:
            from scipy.signal import resample_poly

            divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)
            samples = resample_poly(samples, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor)
        return np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)

    def process(self, sample_rate: int, frame: Any) -> bool:
        """Score one mic frame; return True when the wake phrase is detected."""
        self._buffer = np.concatenate([self._buffer, self._to_mono_16k_int16(sample_rate, frame)])

        detected = False
        while self._buffer.size >= CHUNK_SAMPLES and not detected:
            chunk, self._buffer = self._buffer[:CHUNK_SAMPLES], self._buffer[CHUNK_SAMPLES:]
            scores = self._model.predict(chunk)
            best = max(scores.values()) if scores else 0.0
            if best >= self._threshold:
                logger.info("Wake word detected (score=%.2f)", best)
                detected = True

        if detected:
            self.reset()
        return detected


def build_wake_word_detector(model_name: str, threshold: float) -> WakeWordDetector | None:
    """Build the detector, returning None (with a log) when unavailable."""
    try:
        return WakeWordDetector(model_name, threshold)
    except ImportError:
        logger.error(
            "openwakeword is not installed; install the [wakeword] extra to use standby mode."
        )
    except Exception:
        logger.exception("Failed to initialize wake word model %r", model_name)
    return None
