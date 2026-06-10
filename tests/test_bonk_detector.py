"""Tests for the bonk detector (#522): audio impact signatures → caution list.

A bonk — the head striking the plastic cowling — is a short *broadband*
transient: a sudden onset well above the rolling background with substantial
high-frequency energy (unlike speech onsets, whose energy sits mostly below
2 kHz). The detector is pure (fed blocks, no audio deps) so the signature
logic is unit-testable; capture and caution-list persistence are seams.
"""

from __future__ import annotations
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from robot_comic.observer.bonk_detector import BonkDetector, watch, finalize_partial_wav, append_caution_entries


SR = 16_000
BLOCK = 800  # 50 ms


def _silence() -> np.ndarray:
    return np.zeros(BLOCK, dtype=np.float32)


def _click(amplitude: float = 0.6) -> np.ndarray:
    """A broadband impact: white-noise burst with a sharp decay envelope."""
    rng = np.random.default_rng(42)
    burst = rng.uniform(-1.0, 1.0, BLOCK).astype(np.float32)
    envelope = np.exp(-np.arange(BLOCK) / (BLOCK / 8)).astype(np.float32)
    return amplitude * burst * envelope


def _speech_like(amplitude: float = 0.6) -> np.ndarray:
    """A voiced onset: low-frequency harmonics (150/300/450 Hz)."""
    t = np.arange(BLOCK) / SR
    wave = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((150, 300, 450)))
    return (amplitude * wave / np.max(np.abs(wave))).astype(np.float32)


def _feed(detector: BonkDetector, blocks: list[np.ndarray]) -> list[dict]:
    hits = []
    for i, block in enumerate(blocks):
        hit = detector.update(ts_ms=i * 50, block=block, sample_rate=SR)
        if hit is not None:
            hits.append(hit)
    return hits


def test_click_after_silence_is_detected() -> None:
    detector = BonkDetector()
    hits = _feed(detector, [_silence()] * 10 + [_click()] + [_silence()] * 5)
    assert len(hits) == 1
    assert hits[0]["ts_ms"] == 10 * 50
    assert hits[0]["peak_dbfs"] > -30


def test_speech_onset_is_not_detected() -> None:
    """Loud but low-frequency onsets (speech/TTS) must not count as bonks."""
    detector = BonkDetector()
    hits = _feed(detector, [_silence()] * 10 + [_speech_like()] * 6)
    assert hits == []


def test_click_during_speech_is_detected() -> None:
    """A bonk while the robot is talking still pops above the rolling floor."""
    detector = BonkDetector()
    speech = [_speech_like(0.2) for _ in range(12)]
    speech_with_bonk = speech + [_click(0.9)] + [_speech_like(0.2)] * 4
    hits = _feed(detector, speech_with_bonk)
    assert len(hits) == 1
    assert hits[0]["ts_ms"] == 12 * 50


def test_detections_are_debounced() -> None:
    """Two clicks 50 ms apart are one impact event, not two."""
    detector = BonkDetector(debounce_ms=500)
    hits = _feed(detector, [_silence()] * 10 + [_click(), _click()] + [_silence()] * 5)
    assert len(hits) == 1


def test_quiet_clicks_below_floor_are_ignored() -> None:
    detector = BonkDetector()
    hits = _feed(detector, [_silence()] * 10 + [_click(0.005)] + [_silence()] * 5)
    assert hits == []


# ---------------------------------------------------------------------------
# Caution list persistence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# watch(): device selection + raw-capture recording (#531)
# ---------------------------------------------------------------------------


class _FakeStream:
    """Stands in for sd.InputStream; records the kwargs it was opened with."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        _FakeStream.last_kwargs = kwargs

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, n: int) -> tuple[np.ndarray, bool]:
        return np.zeros((n, 1), dtype=np.float32), False


class _StubSounddevice:
    InputStream = _FakeStream

    def query_devices(self) -> list[dict]:
        return [
            {"name": "Microphone Array (Intel Smart Sound)", "max_input_channels": 6},
            {"name": "Speakerphone (Brio 500)", "max_input_channels": 2},
        ]


@pytest.fixture()
def stub_sd(monkeypatch: pytest.MonkeyPatch) -> _StubSounddevice:
    stub = _StubSounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", stub)
    _FakeStream.last_kwargs = None
    return stub


def test_watch_resolves_device_name_from_env(stub_sd, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOT_AUDIO_DEVICE", "brio")
    watch(0.1, caution_list=str(tmp_path / "caution.jsonl"))
    assert _FakeStream.last_kwargs is not None
    assert _FakeStream.last_kwargs["device"] == 1


def test_watch_explicit_device_overrides_env(stub_sd, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOT_AUDIO_DEVICE", "brio")
    watch(0.1, device=0, caution_list=str(tmp_path / "caution.jsonl"))
    assert _FakeStream.last_kwargs is not None
    assert _FakeStream.last_kwargs["device"] == 0


def test_watch_records_wav(stub_sd, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOT_AUDIO_DEVICE", raising=False)
    wav_path = tmp_path / "session.wav"
    report = watch(0.2, caution_list=str(tmp_path / "caution.jsonl"), record_wav=str(wav_path))
    assert report["record_wav"] == str(wav_path)
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getframerate() == 16_000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        # 0.2 s at 16 kHz in 50 ms blocks -> 4 blocks of 800 frames
        assert wf.getnframes() == 3_200
    assert not (tmp_path / "session.wav.partial").exists(), "clean finish should remove the sidecar"


class _DyingStream(_FakeStream):
    """Delivers two blocks, then dies — models a watch killed mid-window."""

    reads = 0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        _DyingStream.reads = 0

    def read(self, n: int) -> tuple[np.ndarray, bool]:
        if _DyingStream.reads >= 2:
            raise RuntimeError("stream died")
        _DyingStream.reads += 1
        return np.zeros((n, 1), dtype=np.float32), False


def test_killed_watch_leaves_recoverable_partial(stub_sd, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOT_AUDIO_DEVICE", raising=False)
    monkeypatch.setattr(stub_sd, "InputStream", _DyingStream, raising=False)
    wav_path = tmp_path / "session.wav"
    with pytest.raises(RuntimeError, match="stream died"):
        watch(1.0, caution_list=str(tmp_path / "caution.jsonl"), record_wav=str(wav_path))
    partial = tmp_path / "session.wav.partial"
    assert partial.exists(), "audio captured before the kill must survive on disk"
    # 2 delivered blocks of 800 frames at 2 bytes/frame
    assert partial.stat().st_size == 2 * 800 * 2

    finalize_partial_wav(str(wav_path))
    assert not partial.exists()
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnframes() == 1_600


def test_append_caution_entries_writes_attributed_jsonl(tmp_path) -> None:
    import json

    path = tmp_path / "bonk_caution_list.jsonl"
    detections = [{"ts_ms": 1000, "peak_dbfs": -12.5, "hf_ratio": 0.7}]
    attributions = [{"t": 1.0, "energy": None, "tool": "play_emotion", "offset_s": 0.4}]
    append_caution_entries(str(path), detections, attributions)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["tool"] == "play_emotion"
    assert entry["peak_dbfs"] == -12.5
    assert entry["offset_s"] == 0.4
