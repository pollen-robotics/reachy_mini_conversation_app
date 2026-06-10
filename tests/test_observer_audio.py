"""Tests for Tier-1 of the closing-the-loop observer (audio witness).

Covers the pure interval-detection state machine, the dBFS helper, and the
input-device resolver in ``audio_witness``, plus the dependency-free reader in
``audio_activity``. The ``sounddevice`` capture loop (``run_witness``) needs
real hardware, so it isn't exercised here.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

import pytest

from robot_comic.observer import audio_witness, audio_activity


# ---------------------------------------------------------------------------
# dbfs (needs numpy)
# ---------------------------------------------------------------------------
def test_dbfs_empty_is_floor() -> None:
    pytest.importorskip("numpy")
    assert audio_witness.dbfs([]) == audio_witness._SILENCE_DBFS


def test_dbfs_silence_is_floor() -> None:
    np = pytest.importorskip("numpy")
    assert audio_witness.dbfs(np.zeros(160, dtype="float32")) == audio_witness._SILENCE_DBFS


def test_dbfs_full_scale_is_about_zero() -> None:
    np = pytest.importorskip("numpy")
    # A full-scale ±1.0 square wave has RMS 1.0 -> 0 dBFS.
    frame = np.ones(160, dtype="float32")
    assert audio_witness.dbfs(frame) == pytest.approx(0.0, abs=1e-6)


def test_dbfs_half_scale_is_about_minus_six() -> None:
    np = pytest.importorskip("numpy")
    frame = np.full(160, 0.5, dtype="float32")
    assert audio_witness.dbfs(frame) == pytest.approx(-6.02, abs=0.05)


# ---------------------------------------------------------------------------
# IntervalDetector (pure, no audio deps)
# ---------------------------------------------------------------------------
def test_detector_opens_and_closes_one_interval() -> None:
    det = audio_witness.IntervalDetector(db_on=-35.0, db_off=-45.0)
    assert det.update(1000, -60.0) is None  # quiet
    assert det.update(1100, -30.0) is None  # crosses ON -> opens, not yet closed
    assert det.update(1200, -20.0) is None  # louder, peak tracked
    closed = det.update(1300, -50.0)  # crosses OFF -> closes
    assert closed == {"start_ms": 1100, "end_ms": 1300, "dur_ms": 200, "peak_dbfs": -20.0}


def test_detector_hysteresis_holds_through_small_dip() -> None:
    det = audio_witness.IntervalDetector(db_on=-35.0, db_off=-45.0)
    det.update(0, -30.0)  # open
    # -40 is below ON but above OFF -> stays open (no new interval)
    assert det.update(100, -40.0) is None
    closed = det.update(200, -50.0)
    assert closed is not None
    assert closed["start_ms"] == 0
    assert closed["end_ms"] == 200


def test_detector_flush_closes_open_interval() -> None:
    det = audio_witness.IntervalDetector(db_on=-35.0, db_off=-45.0)
    det.update(500, -10.0)  # open, peak -10
    flushed = det.flush()
    assert flushed == {"start_ms": 500, "end_ms": 500, "dur_ms": 0, "peak_dbfs": -10.0}
    # second flush is a no-op
    assert det.flush() is None


def test_detector_flush_noop_when_idle() -> None:
    det = audio_witness.IntervalDetector()
    det.update(0, -80.0)
    assert det.flush() is None


# ---------------------------------------------------------------------------
# resolve_input_device (#531: the witness must aim at the Brio, not the default)
# ---------------------------------------------------------------------------
class _StubSounddevice:
    """Minimal sounddevice stand-in for name-based device resolution."""

    def __init__(self, devices: list[dict]) -> None:
        self._devices = devices

    def query_devices(self) -> list[dict]:
        return self._devices


_DEVICES = [
    {"name": "Microphone Array (Intel Smart Sound)", "max_input_channels": 6},
    {"name": "Speakers (Brio 500)", "max_input_channels": 0},  # output-only
    {"name": "Speakerphone (Brio 500)", "max_input_channels": 2},
]


def test_resolve_device_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOT_AUDIO_DEVICE", raising=False)
    assert audio_witness.resolve_input_device() is None


def test_resolve_device_int_passthrough() -> None:
    assert audio_witness.resolve_input_device(3) == 3


def test_resolve_device_numeric_string_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOT_AUDIO_DEVICE", "2")
    assert audio_witness.resolve_input_device() == 2


def test_resolve_device_name_substring_skips_output_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOT_AUDIO_DEVICE", "brio")
    monkeypatch.setitem(sys.modules, "sounddevice", _StubSounddevice(_DEVICES))
    assert audio_witness.resolve_input_device() == 2


def test_resolve_device_explicit_value_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOT_AUDIO_DEVICE", "brio")
    monkeypatch.setitem(sys.modules, "sounddevice", _StubSounddevice(_DEVICES))
    assert audio_witness.resolve_input_device("intel") == 0


def test_resolve_device_no_match_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", _StubSounddevice(_DEVICES))
    with pytest.raises(ValueError, match="yeti"):
        audio_witness.resolve_input_device("yeti")


# ---------------------------------------------------------------------------
# _append_interval + rotation
# ---------------------------------------------------------------------------
def test_append_interval_writes_and_rotates(tmp_path: Path) -> None:
    log = tmp_path / "audio.jsonl"
    audio_witness._append_interval(str(log), {"start_ms": 1, "end_ms": 2, "dur_ms": 1, "peak_dbfs": -3.0})
    audio_witness._append_interval(
        str(log), {"start_ms": 3, "end_ms": 4, "dur_ms": 1, "peak_dbfs": -4.0}, max_bytes=10
    )
    backup = tmp_path / "audio.jsonl.1"
    assert backup.exists(), "oversized log should rotate to a .1 backup"
    assert json.loads(backup.read_text(encoding="utf-8").strip())["start_ms"] == 1
    assert json.loads(log.read_text(encoding="utf-8").strip())["start_ms"] == 3


# ---------------------------------------------------------------------------
# Reader: audio_activity
# ---------------------------------------------------------------------------
def test_read_intervals_missing_file() -> None:
    assert audio_activity.read_intervals("/no/such/audio.jsonl") == []


def test_read_intervals_skips_corrupt(tmp_path: Path) -> None:
    log = tmp_path / "audio.jsonl"
    log.write_text(
        '{"start_ms":1,"end_ms":2,"dur_ms":1,"peak_dbfs":-3}\n'
        "garbage{\n"
        "\n"
        '{"start_ms":5,"end_ms":7,"dur_ms":2,"peak_dbfs":-9}\n',
        encoding="utf-8",
    )
    starts = [iv["start_ms"] for iv in audio_activity.read_intervals(str(log))]
    assert starts == [1, 5]


def test_recent_audio_activity_filters_by_window(tmp_path: Path) -> None:
    log = tmp_path / "audio.jsonl"
    # now=110_000, window=10s -> cutoff=100_000 (inclusive on end_ms)
    log.write_text(
        '{"start_ms":90000,"end_ms":95000,"dur_ms":5000,"peak_dbfs":-10}\n'
        '{"start_ms":104000,"end_ms":106000,"dur_ms":2000,"peak_dbfs":-8}\n',
        encoding="utf-8",
    )
    got = audio_activity.recent_audio_activity(str(log), window_s=10.0, now_ms=110_000)
    assert [iv["start_ms"] for iv in got] == [104000]


def test_summarize_activity_rollup() -> None:
    intervals = [
        {"start_ms": 0, "end_ms": 1000, "dur_ms": 1000, "peak_dbfs": -20.0},
        {"start_ms": 2000, "end_ms": 2500, "dur_ms": 500, "peak_dbfs": -8.0},
    ]
    summary = audio_activity.summarize_activity(intervals)
    assert summary["sound_present"] is True
    assert summary["interval_count"] == 2
    assert summary["total_active_ms"] == 1500
    assert summary["peak_dbfs"] == -8.0
    assert summary["latest_end_ms"] == 2500


def test_summarize_activity_empty() -> None:
    assert audio_activity.summarize_activity([]) == {
        "sound_present": False,
        "interval_count": 0,
        "total_active_ms": 0,
        "peak_dbfs": None,
        "latest_end_ms": None,
    }
