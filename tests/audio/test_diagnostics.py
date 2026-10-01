import wave
import logging
from pathlib import Path

import pytest

from reachy_mini_conversation_app.audio import diagnostics as diagnostics_mod
from reachy_mini_conversation_app.audio.diagnostics import AudioDiagnostics


def test_capture_stops_at_time_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The time limit finalizes WAVs and prevents further audio or event writes."""
    capture = AudioDiagnostics(tmp_path)
    capture.record_audio("sent", 16000, b"\x01\x00\x02\x00")
    monkeypatch.setattr(diagnostics_mod, "CAPTURE_SECONDS", 0)

    capture.record_audio("sent", 16000, b"\x03\x00")
    capture.record_event("input_audio_buffer.speech_started")
    capture.close()

    with wave.open(str(capture.directory / "sent.wav"), "rb") as recording:
        assert recording.getnframes() == 2
        assert recording.readframes(2) == b"\x01\x00\x02\x00"
    assert capture.closed
    assert len((capture.directory / "events.jsonl").read_text().splitlines()) == 1


def test_capture_logs_disk_failure_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A disk write failure disables diagnostics without raising into the audio loop."""
    capture = AudioDiagnostics(tmp_path)

    def fail_write(self: wave.Wave_write, pcm: bytes) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(wave.Wave_write, "writeframes", fail_write)
    with caplog.at_level(logging.WARNING):
        capture.record_audio("sent", 16000, b"\x01\x00")

    assert capture.closed
    assert "Audio diagnostic WAV write failed" in caplog.text
