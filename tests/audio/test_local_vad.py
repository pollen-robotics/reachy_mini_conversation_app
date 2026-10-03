from unittest.mock import MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.audio.local_vad as vad_mod
from reachy_mini_conversation_app.audio.local_vad import SileroVAD


def test_streaming_speech_holds_through_pauses_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Partial frames retain model context, and brief pauses do not end listening."""
    session = MagicMock()
    probabilities = iter([0.8, *([0.1] * 12), 0.8, *([0.1] * 13), 0.8, 0.1])

    def infer(_outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.array([[next(probabilities)]], dtype=np.float32), np.ones_like(inputs["state"])]

    session.run.side_effect = infer
    monkeypatch.setattr(vad_mod.ort, "InferenceSession", lambda *_args, **_kwargs: session)
    detector = SileroVAD()
    half_chunk = np.full(256, 16384, dtype=np.int16)
    silence = np.zeros(512, dtype=np.int16)

    assert not detector.process(half_chunk, 16000)
    session.run.assert_not_called()
    assert detector.process(half_chunk, 16000)
    first_inputs = session.run.call_args.args[1]
    np.testing.assert_array_equal(first_inputs["input"][:, :64], 0.0)
    np.testing.assert_array_equal(first_inputs["input"][:, 64:], 0.5)

    for _ in range(12):
        assert detector.process(silence, 16000)
    assert detector.process(silence, 16000)
    for _ in range(12):
        assert detector.process(silence, 16000)
    assert not detector.process(silence, 16000)

    assert detector.process(half_chunk, 16000) is False
    assert detector.process(half_chunk, 16000)
    detector.process(half_chunk, 16000)
    detector.reset()
    assert not detector.process(half_chunk, 16000)
    assert not detector.process(half_chunk, 16000)
    reset_inputs = session.run.call_args.args[1]
    np.testing.assert_array_equal(reset_inputs["state"], 0.0)
    np.testing.assert_array_equal(reset_inputs["input"][:, :64], 0.0)


def test_bundled_model_accepts_streaming_silence() -> None:
    """The shipped model loads on CPU and accepts its streaming input shapes."""
    detector = SileroVAD()
    silence = np.zeros(16000, dtype=np.int16)
    assert not detector.process(silence, 16000)
    detector.reset()
    assert not detector.process(silence[:512], 16000)
    with pytest.raises(ValueError, match="16 kHz"):
        detector.process(silence, 48000)
