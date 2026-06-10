"""Tier-1 closing-the-loop audio witness.

An always-on monitor that reads the *coding-laptop* microphone, measures the
loudness of each short block, and logs "sound present" intervals to a JSONL
file. It is the independent confirmation that the robot's speaker actually made
noise — the Tier-0 event log only reports what the app *thinks* it did.

Run it on the machine whose mic can hear the robot (native Windows recommended;
WSL2 mic capture is unreliable)::

    ROBOT_AUDIO_LOG=/path/to/audio.jsonl python -m robot_comic.observer.audio_witness

Requires ``sounddevice`` + ``numpy`` (``uv pip install sounddevice numpy``); both
are imported lazily so the reader (``audio_activity``) and the MCP server stay
usable without them. See ``docs/closing-the-loop.md``.
"""

from __future__ import annotations
import os
import json
import math
import time
import argparse
from typing import Any, Optional


# dBFS thresholds mirror audio/speech_tapper.py's VAD envelope. Hysteresis: a
# block at/above DB_ON opens an interval; it stays open until a block at/below
# DB_OFF closes it, so brief dips don't chop one utterance into many.
DB_ON = float(os.getenv("ROBOT_AUDIO_DB_ON", "-35"))
DB_OFF = float(os.getenv("ROBOT_AUDIO_DB_OFF", "-45"))
# Rotate the log to a single ``.1`` backup past this size (matches telemetry).
MAX_BYTES = int(os.getenv("ROBOT_AUDIO_LOG_MAX_BYTES", str(5_000_000)))

_SILENCE_DBFS = -120.0


def resolve_input_device(value: int | str | None = None) -> int | None:
    """Resolve the witness input device to a sounddevice index.

    ``value`` (or, when omitted, the ``ROBOT_AUDIO_DEVICE`` env var — read at
    call time, not import time) may be a device index or a case-insensitive
    name substring (e.g. ``"Brio"``); a substring resolves to the first input-
    capable device whose name contains it. Empty/unset means the system
    default (``None``). Raises ``ValueError`` when a substring matches no
    input device — a witness silently listening to the wrong mic looks
    exactly like a healthy witness hearing silence (#531).
    """
    if value is None:
        value = os.getenv("ROBOT_AUDIO_DEVICE", "")
    if isinstance(value, int):
        return value
    raw = value.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    import sounddevice as sd

    needle = raw.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) > 0 and needle in str(dev.get("name", "")).lower():
            return idx
    raise ValueError(f"no audio input device matching {raw!r}")


def dbfs(frame: Any) -> float:
    """Return the RMS level of a float32 mono frame (values in [-1, 1]) in dBFS.

    Empty or silent input returns a fixed floor rather than ``-inf`` so the
    value stays JSON-serializable and comparable.
    """
    import numpy as np

    arr = np.asarray(frame, dtype=np.float64).ravel()
    if arr.size == 0:
        return _SILENCE_DBFS
    rms = float(np.sqrt(np.mean(arr * arr)))
    if rms <= 1e-12:
        return _SILENCE_DBFS
    return 20.0 * math.log10(rms)


class IntervalDetector:
    """Turn a stream of ``(ts_ms, dbfs)`` samples into sound-present intervals.

    Pure (no audio deps) so the on/off hysteresis is fully unit-testable.
    """

    def __init__(self, db_on: float = DB_ON, db_off: float = DB_OFF) -> None:
        """Configure the open/close dBFS thresholds (``db_on`` >= ``db_off``)."""
        self.db_on = db_on
        self.db_off = db_off
        self._active = False
        self._start_ms: Optional[int] = None
        self._peak: float = _SILENCE_DBFS
        self._last_ts: Optional[int] = None

    def update(self, ts_ms: int, db: float) -> Optional[dict[str, Any]]:
        """Feed one sample; return a closed interval dict when one just ended."""
        self._last_ts = ts_ms
        if not self._active:
            if db >= self.db_on:
                self._active = True
                self._start_ms = ts_ms
                self._peak = db
            return None
        self._peak = max(self._peak, db)
        if db <= self.db_off:
            return self._close(ts_ms)
        return None

    def flush(self) -> Optional[dict[str, Any]]:
        """Close any still-open interval (e.g. on shutdown). Returns it or None."""
        if self._active and self._last_ts is not None:
            return self._close(self._last_ts)
        return None

    def _close(self, end_ms: int) -> dict[str, Any]:
        """Emit the current interval and reset to the idle state."""
        start = self._start_ms if self._start_ms is not None else end_ms
        interval = {
            "start_ms": start,
            "end_ms": end_ms,
            "dur_ms": end_ms - start,
            "peak_dbfs": round(self._peak, 1),
        }
        self._active = False
        self._start_ms = None
        self._peak = _SILENCE_DBFS
        return interval


def _append_interval(path: str, interval: dict[str, Any], *, max_bytes: int = MAX_BYTES) -> None:
    """Append one interval as a JSON line, rotating to ``<path>.1`` past size."""
    try:
        if os.path.getsize(path) >= max_bytes:
            os.replace(path, path + ".1")
    except OSError:
        pass  # file missing or unstattable — nothing to rotate
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(interval, separators=(",", ":")) + "\n")


def run_witness(
    path: str,
    *,
    device: Optional[int | str] = None,
    samplerate: int = 16_000,
    block_ms: int = 100,
    db_on: float = DB_ON,
    db_off: float = DB_OFF,
) -> None:
    """Capture the mic and append sound-present intervals to ``path`` until Ctrl-C.

    ``device`` accepts an index or name substring; ``None`` falls back to
    ``ROBOT_AUDIO_DEVICE``, then the system default (see
    ``resolve_input_device``). Imports ``sounddevice`` lazily so this module
    stays importable (and its pure helpers testable) on hosts without an audio
    stack.
    """
    import sounddevice as sd

    device_index = resolve_input_device(device)
    detector = IntervalDetector(db_on, db_off)
    block = max(1, int(samplerate * block_ms / 1000))
    with sd.InputStream(
        channels=1, samplerate=samplerate, blocksize=block, dtype="float32", device=device_index
    ) as stream:
        try:
            while True:
                frames, _ = stream.read(block)
                ts_ms = int(time.time() * 1000)
                interval = detector.update(ts_ms, dbfs(frames))
                if interval is not None:
                    _append_interval(path, interval)
        except KeyboardInterrupt:
            tail = detector.flush()
            if tail is not None:
                _append_interval(path, tail)


def main() -> None:
    """Console entry point: read ``ROBOT_AUDIO_LOG`` + flags, run the witness."""
    parser = argparse.ArgumentParser(description="Tier-1 robot audio witness")
    parser.add_argument(
        "--device",
        default=None,
        help="input device index or name substring, e.g. 'Brio' (default: ROBOT_AUDIO_DEVICE, then system default)",
    )
    parser.add_argument("--samplerate", type=int, default=16_000)
    parser.add_argument("--block-ms", type=int, default=100)
    args = parser.parse_args()

    path = os.getenv("ROBOT_AUDIO_LOG", "").strip()
    if not path:
        raise SystemExit("ROBOT_AUDIO_LOG is not set — point it at the audio-witness JSONL path.")
    run_witness(path, device=args.device, samplerate=args.samplerate, block_ms=args.block_ms)


if __name__ == "__main__":
    main()
