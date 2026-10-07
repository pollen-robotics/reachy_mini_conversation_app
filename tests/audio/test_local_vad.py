from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.audio import local_vad
from reachy_mini_conversation_app.audio.local_vad import WebRTCVAD


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide a pipeline whose speech events follow queued test decisions."""
    local_vad.Gst.init(None)
    pipeline = MagicMock()
    pipeline.decisions = []
    pipeline.messages = []
    source = MagicMock()
    sink = MagicMock()
    detector = MagicMock()
    detector.get_property.return_value = True
    pipeline.get_by_name.side_effect = {"source": source, "sink": sink, "detector": detector}.__getitem__

    def push_buffer(_signal: str, _buffer: object) -> object:
        structure = local_vad.Gst.Structure.new_empty("voice-activity")
        structure.set_value("stream-has-voice", pipeline.decisions.pop(0))
        pipeline.messages.append(local_vad.Gst.Message.new_element(None, structure))
        return local_vad.Gst.FlowReturn.OK

    source.emit.side_effect = push_buffer
    pipeline.get_bus.return_value.timed_pop_filtered.side_effect = lambda *_args: (
        pipeline.messages.pop(0) if pipeline.messages else None
    )
    monkeypatch.setattr(local_vad.Gst, "parse_launch", lambda _description: pipeline)
    return pipeline


def test_speech_holds_through_short_pauses_and_resets(pipeline: MagicMock) -> None:
    """Detected speech persists through short pauses and clears on reset."""
    detector = WebRTCVAD()
    pipeline.decisions = [True] + [False] * 40
    half_frame = np.zeros(80, dtype=np.int16)
    assert detector.process(half_frame, 16000) is False
    assert detector.process(half_frame, 16000) is True
    assert detector.process(np.zeros(39 * 160, dtype=np.int16), 16000) is True
    assert detector.process(np.zeros(160, dtype=np.int16), 16000) is False
    pipeline.decisions = [True, False]
    assert detector.process(np.zeros(160, dtype=np.int16), 16000) is True
    assert detector.process(np.zeros(160, dtype=np.int16), 16000, reset=True) is False
    detector.close()
    pipeline.set_state.assert_called_with(local_vad.Gst.State.NULL)


def test_unsupported_plugin_releases_pipeline(pipeline: MagicMock) -> None:
    """An installed plugin without VAD support is rejected without starting it."""
    pipeline.get_by_name("detector").get_property.return_value = False
    with pytest.raises(RuntimeError, match="does not support voice detection"):
        WebRTCVAD()
    pipeline.set_state.assert_called_once_with(local_vad.Gst.State.NULL)


def test_processing_failure_is_reported(pipeline: MagicMock) -> None:
    """A stalled pipeline fails instead of blocking microphone processing indefinitely."""
    detector = WebRTCVAD()
    pipeline.decisions = [True]
    pipeline.get_by_name("sink").emit.return_value = None
    with pytest.raises(TimeoutError, match="did not process"):
        detector.process(np.zeros(160, dtype=np.int16), 16000)
    detector.close()
