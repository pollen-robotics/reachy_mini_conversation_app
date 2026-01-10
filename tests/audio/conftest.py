"""Shared fixtures for audio module tests."""

import math
import base64
from typing import Callable, Generator, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler


@pytest.fixture
def sample_rate() -> int:
    """Return standard sample rate for tests."""
    return 24000


@pytest.fixture
def sine_wave_generator(sample_rate: int) -> Callable[..., NDArray[np.int16]]:
    """Create a factory fixture to generate sine wave PCM data."""

    def _generate(
        duration_s: float = 0.3,
        frequency_hz: float = 220.0,
        amplitude: float = 0.6,
    ) -> NDArray[np.int16]:
        sample_count = int(sample_rate * duration_s)
        t = np.linspace(0, duration_s, sample_count, endpoint=False)
        wave = amplitude * np.sin(2 * math.pi * frequency_hz * t)
        pcm = np.clip(wave * np.iinfo(np.int16).max, -32768, 32767).astype(np.int16)
        return pcm

    return _generate


@pytest.fixture
def base64_audio_generator(
    sine_wave_generator: Callable[..., NDArray[np.int16]],
) -> Callable[..., str]:
    """Create a factory fixture to generate base64-encoded audio chunks."""

    def _generate(
        duration_s: float = 0.3,
        frequency_hz: float = 220.0,
        amplitude: float = 0.6,
    ) -> str:
        pcm = sine_wave_generator(duration_s, frequency_hz, amplitude)
        return base64.b64encode(pcm.tobytes()).decode("ascii")

    return _generate


@pytest.fixture
def silence_audio(sample_rate: int) -> NDArray[np.int16]:
    """Generate silent audio (zeros)."""
    return np.zeros(sample_rate, dtype=np.int16)


@pytest.fixture
def loud_audio(sample_rate: int) -> NDArray[np.int16]:
    """Generate loud audio for VAD testing."""
    duration_s = 0.5
    sample_count = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, sample_count, endpoint=False)
    # High amplitude sine wave
    wave = 0.9 * np.sin(2 * math.pi * 440.0 * t)
    return np.clip(wave * np.iinfo(np.int16).max, -32768, 32767).astype(np.int16)


@pytest.fixture
def quiet_audio(sample_rate: int) -> NDArray[np.int16]:
    """Generate quiet audio (just above noise floor)."""
    duration_s = 0.5
    sample_count = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, sample_count, endpoint=False)
    # Very low amplitude
    wave = 0.001 * np.sin(2 * math.pi * 440.0 * t)
    return np.clip(wave * np.iinfo(np.int16).max, -32768, 32767).astype(np.int16)


@pytest.fixture
def captured_offsets() -> List[Tuple[float, Tuple[float, float, float, float, float, float]]]:
    """List to capture wobbler offsets."""
    return []


@pytest.fixture
def head_wobbler(
    captured_offsets: List[Tuple[float, Tuple[float, float, float, float, float, float]]],
) -> Generator[HeadWobbler, None, None]:
    """Create a HeadWobbler instance with offset capture."""
    import time

    def capture(offsets: Tuple[float, float, float, float, float, float]) -> None:
        captured_offsets.append((time.time(), offsets))

    wobbler = HeadWobbler(set_speech_offsets=capture)
    yield wobbler
    # Ensure cleanup
    wobbler.stop()
