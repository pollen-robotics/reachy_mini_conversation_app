"""Tests for standby wake-word detection plumbing."""

import sys
import types

import numpy as np
import pytest

from reachy_mini_conversation_app.standby import (
    CHUNK_SAMPLES,
    TARGET_SAMPLE_RATE,
    WakeWordDetector,
    build_wake_word_detector,
)


class _FakeOwwModel:
    """Stands in for openwakeword.model.Model; scores are driven by the test."""

    def __init__(self, wakeword_models, inference_framework="onnx"):
        self.wakeword_models = wakeword_models
        self.inference_framework = inference_framework
        self.chunks: list[np.ndarray] = []
        self.scores: list[float] = []
        self.reset_calls = 0

    def predict(self, chunk):
        self.chunks.append(chunk)
        score = self.scores.pop(0) if self.scores else 0.0
        return {"fake_model": score}

    def reset(self):
        self.reset_calls += 1


@pytest.fixture
def fake_openwakeword(monkeypatch):
    """Install a fake openwakeword package and return the model instances it builds."""
    instances: list[_FakeOwwModel] = []

    def factory(*args, **kwargs):
        model = _FakeOwwModel(*args, **kwargs)
        instances.append(model)
        return model

    module = types.ModuleType("openwakeword")
    model_module = types.ModuleType("openwakeword.model")
    model_module.Model = factory
    module.model = model_module
    monkeypatch.setitem(sys.modules, "openwakeword", module)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)
    return instances


def _int16_frame(num_samples: int) -> np.ndarray:
    return np.zeros(num_samples, dtype=np.int16)


def test_detector_buffers_until_full_chunk(fake_openwakeword):
    """Frames smaller than one chunk buffer without scoring."""
    detector = WakeWordDetector("fake_model", threshold=0.5)
    model = fake_openwakeword[0]

    assert detector.process(TARGET_SAMPLE_RATE, _int16_frame(CHUNK_SAMPLES // 2)) is False
    assert model.chunks == []

    assert detector.process(TARGET_SAMPLE_RATE, _int16_frame(CHUNK_SAMPLES // 2)) is False
    assert len(model.chunks) == 1
    assert model.chunks[0].size == CHUNK_SAMPLES
    assert model.chunks[0].dtype == np.int16


def test_detector_fires_at_threshold_and_resets(fake_openwakeword):
    """Detection fires at threshold and resets detector state."""
    detector = WakeWordDetector("fake_model", threshold=0.5)
    model = fake_openwakeword[0]
    model.scores = [0.2, 0.9]

    assert detector.process(TARGET_SAMPLE_RATE, _int16_frame(CHUNK_SAMPLES)) is False
    assert detector.process(TARGET_SAMPLE_RATE, _int16_frame(CHUNK_SAMPLES)) is True
    # Detection resets buffered audio and model state for the next standby.
    assert model.reset_calls == 1
    assert detector._buffer.size == 0


def test_detector_resamples_other_rates(fake_openwakeword):
    """Non-16 kHz stereo input is resampled and mono-ized to chunks."""
    detector = WakeWordDetector("fake_model", threshold=0.5)
    model = fake_openwakeword[0]

    # 48 kHz stereo frame, channels-last: enough for exactly one 16 kHz chunk.
    frame = np.zeros((CHUNK_SAMPLES * 3, 2), dtype=np.float32)
    detector.process(48000, frame)
    assert len(model.chunks) == 1
    assert model.chunks[0].size == CHUNK_SAMPLES


def test_build_detector_returns_none_without_openwakeword(monkeypatch):
    """Missing openwakeword dependency yields None, not a crash."""
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    monkeypatch.setitem(sys.modules, "openwakeword.model", None)
    assert build_wake_word_detector("hey_jarvis_v0.1", 0.5) is None
