"""Bonk detector (#522): audio impact signatures correlated into a caution list.

A bonk — the head striking the plastic cowling — reads as a short
**low-frequency thump** on the witness microphone: an extreme peak onset over
the rolling background that rings down within ~200 ms. Ground truth (10 real
cowling taps, 2026-06-10): peaks −1..−9 dBFS, onsets +44..+54 dB, HF ratio
0.00–0.11. The shell resonates low — impacts are *not* broadband here; the
broadband transients in a real room are keyboard/mouse clicks, so a high HF
ratio now *rejects* a candidate instead of confirming it.

An onset over the rolling background becomes a pending candidate and confirms
only if the level decays ``ROBOT_BONK_DECAY_DB`` within
``ROBOT_BONK_DECAY_BLOCKS`` blocks — speech sustains, impacts ring down.
(Validated offline: 10/10 ground-truth taps, 0 false positives on speech
onsets after silence, which an immediate-confirm-on-extreme-onset variant
let through.)

Pipeline: mic blocks → ``BonkDetector`` (pure, unit-tested) → correlate each
detection with the Tier-0 event log (same join as the motion witness) →
append ``(tool, ts, level)`` entries to a persistent caution list the
operator can review before re-limiting emotions (keep the emotion, clamp the
range — not a denylist).

CLI (workstation, mic able to hear the robot)::

    python -m robot_comic.observer.bonk_detector --seconds 300 \
        --events D:/logs/ricci_events.log
"""

from __future__ import annotations
import os
import json
import math
import time
import argparse
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robot_comic.observer.audio_witness import dbfs, resolve_input_device
from robot_comic.observer.motion_witness import load_events, correlate_spikes_with_events


# Onset: the block's PEAK level must clear the rolling RMS background by this
# much to become a candidate. Impacts have a high crest factor, so peak-vs-RMS
# keeps a bonk visible even while the robot is talking over it.
ONSET_DB = float(os.getenv("ROBOT_BONK_ONSET_DB", "25"))
# …and the block's absolute level must clear this floor (ground-truth taps
# peaked −1..−9 dBFS on the witness mic; quiet taps from the street don't
# qualify).
FLOOR_DBFS = float(os.getenv("ROBOT_BONK_FLOOR_DBFS", "-20"))
# Impacts are low-frequency thumps; broadband energy above HF_CUT_HZ marks a
# keyboard/mouse click instead, so high HF *rejects* the candidate.
HF_RATIO_MAX = float(os.getenv("ROBOT_BONK_HF_RATIO_MAX", "0.30"))
HF_CUT_HZ = float(os.getenv("ROBOT_BONK_HF_CUT_HZ", "2000"))
# Moderate-onset candidates confirm only if the level rings down this much…
DECAY_DB = float(os.getenv("ROBOT_BONK_DECAY_DB", "15"))
# …within this many blocks of the (re-anchored) peak. Speech sustains instead.
DECAY_BLOCKS = int(os.getenv("ROBOT_BONK_DECAY_BLOCKS", "4"))
DEBOUNCE_MS = int(os.getenv("ROBOT_BONK_DEBOUNCE_MS", "500"))

DEFAULT_CAUTION_LIST = os.path.join(os.path.expanduser("~"), ".robot_comic", "observer", "bonk_caution_list.jsonl")

_HISTORY_BLOCKS = 10


def _peak_dbfs(block: NDArray[np.float32]) -> float:
    """Peak (not RMS) level of the block in dBFS."""
    peak = float(np.max(np.abs(np.asarray(block, dtype=np.float64))))
    if peak <= 1e-12:
        return -120.0
    return 20.0 * math.log10(peak)


def _hf_ratio(block: NDArray[np.float32], sample_rate: int, cut_hz: float) -> float:
    """Fraction of the block's spectral energy at or above ``cut_hz``."""
    spectrum = np.abs(np.fft.rfft(block.astype(np.float64))) ** 2
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(len(block), d=1.0 / sample_rate)
    return float(np.sum(spectrum[freqs >= cut_hz]) / total)


class BonkDetector:
    """Pure block-fed impact detector: onset + floor + low-HF + ring-down."""

    def __init__(
        self,
        *,
        onset_db: float = ONSET_DB,
        floor_dbfs: float = FLOOR_DBFS,
        hf_ratio_max: float = HF_RATIO_MAX,
        hf_cut_hz: float = HF_CUT_HZ,
        decay_db: float = DECAY_DB,
        decay_blocks: int = DECAY_BLOCKS,
        debounce_ms: int = DEBOUNCE_MS,
    ) -> None:
        """Configure the impact signature thresholds."""
        self.onset_db = onset_db
        self.floor_dbfs = floor_dbfs
        self.hf_ratio_max = hf_ratio_max
        self.hf_cut_hz = hf_cut_hz
        self.decay_db = decay_db
        self.decay_blocks = decay_blocks
        self.debounce_ms = debounce_ms
        self._recent_db: list[float] = []
        self._last_hit_ms: int | None = None
        self._pending: dict[str, Any] | None = None
        self._pending_wait = 0

    def update(self, *, ts_ms: int, block: NDArray[np.float32], sample_rate: int) -> dict[str, Any] | None:
        """Feed one block; return a detection dict when an impact is confirmed."""
        level = _peak_dbfs(block)
        background = float(np.median(self._recent_db)) if self._recent_db else -120.0
        # Background history tracks RMS so sustained speech sets the floor the
        # peak must clear; the impact block itself still enters the history.
        self._recent_db.append(dbfs(block))
        if len(self._recent_db) > _HISTORY_BLOCKS:
            self._recent_db.pop(0)

        if self._pending is not None:
            return self._advance_pending(ts_ms, level)

        if level < self.floor_dbfs:
            return None
        if level - background < self.onset_db:
            return None
        if self._last_hit_ms is not None and (ts_ms - self._last_hit_ms) < self.debounce_ms:
            return None
        ratio = _hf_ratio(block, sample_rate, self.hf_cut_hz)
        if ratio > self.hf_ratio_max:
            return None  # broadband transient = keyboard/click, not a cowling thump

        self._pending = {
            "ts_ms": ts_ms,
            "peak_dbfs": round(level, 1),
            "onset_db": round(level - background, 1),
            "hf_ratio": round(ratio, 3),
        }
        self._pending_wait = 0
        return None

    def _advance_pending(self, ts_ms: int, level: float) -> dict[str, Any] | None:
        """Resolve a pending moderate candidate: re-anchor, confirm, or discard."""
        assert self._pending is not None
        anchor = float(self._pending["peak_dbfs"])
        if level > anchor:
            # Louder follow-up hit (double tap / clipped burst): re-anchor the
            # decay reference and restart the window; keep the original onset ts.
            self._pending["peak_dbfs"] = round(level, 1)
            self._pending_wait = 0
            return None
        if level <= anchor - self.decay_db:
            detection = self._pending
            detection["confirm"] = "decay"
            self._pending = None
            self._last_hit_ms = ts_ms
            return detection
        self._pending_wait += 1
        if self._pending_wait >= self.decay_blocks:
            self._pending = None  # held its level: sustained sound, not an impact
        return None


def append_caution_entries(
    path: str,
    detections: list[dict[str, Any]],
    attributions: list[dict[str, Any]],
) -> None:
    """Append one caution-list line per detection, attributed where possible."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for det, att in zip(detections, attributions):
            entry = {
                "ts_ms": det["ts_ms"],
                "peak_dbfs": det["peak_dbfs"],
                "hf_ratio": det.get("hf_ratio"),
                "tool": att.get("tool"),
                "offset_s": att.get("offset_s"),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")


def _block_to_pcm(block: NDArray[np.float32]) -> bytes:
    """One float32 block as raw 16-bit little-endian PCM bytes."""
    return (np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def finalize_partial_wav(wav_path: str, *, sample_rate: int = 16_000) -> str:
    """Convert ``<wav_path>.partial`` (raw 16-bit mono PCM) into ``wav_path``.

    The watch streams raw PCM to the sidecar as it captures, so a watch that
    is killed mid-window still leaves recoverable audio; this wraps it in a
    WAV header and removes the sidecar. Called automatically on a clean
    finish, and usable manually after a kill.
    """
    import wave

    partial = wav_path + ".partial"
    with open(partial, "rb") as fh:
        pcm = fh.read()
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    os.remove(partial)
    return wav_path


def watch(
    seconds: float,
    *,
    events_path: str | None = None,
    caution_list: str = DEFAULT_CAUTION_LIST,
    sample_rate: int = 16_000,
    block_ms: int = 50,
    device: int | str | None = None,
    record_wav: str | None = None,
) -> dict[str, Any]:
    """Listen on the witness mic for ``seconds``; detect, attribute, persist.

    ``device`` accepts an index or name substring; ``None`` falls back to
    ``ROBOT_AUDIO_DEVICE``, then the system default. ``record_wav`` saves the
    raw capture so thresholds can be tuned offline against a recorded tap
    session instead of re-tapping live. The capture streams unbuffered to
    ``<record_wav>.partial`` while the window is open, so a watch killed
    mid-window keeps its audio (recover with ``finalize_partial_wav``).
    """
    import sounddevice as sd

    device_index = resolve_input_device(device)
    detector = BonkDetector()
    block_size = int(sample_rate * block_ms / 1000)
    detections: list[dict[str, Any]] = []
    start_wall = time.time()

    raw_fh = None
    if record_wav:
        os.makedirs(os.path.dirname(record_wav) or ".", exist_ok=True)
        raw_fh = open(record_wav + ".partial", "wb", buffering=0)
    try:
        with sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_size, device=device_index
        ) as stream:
            n_blocks = int(math.ceil(seconds * 1000 / block_ms))
            for _ in range(n_blocks):
                block, _overflowed = stream.read(block_size)
                ts_ms = int((time.time() - start_wall) * 1000)
                mono = block[:, 0]
                if raw_fh is not None:
                    raw_fh.write(_block_to_pcm(mono))
                hit = detector.update(ts_ms=ts_ms, block=mono, sample_rate=sample_rate)
                if hit is not None:
                    hit["wall_ts"] = int((start_wall + hit["ts_ms"] / 1000.0) * 1000)
                    detections.append(hit)
                    print(
                        f"BONK? t={hit['ts_ms'] / 1000:.1f}s level={hit['peak_dbfs']} dBFS "
                        f"onset=+{hit['onset_db']} dB hf={hit['hf_ratio']}",
                        flush=True,
                    )
    finally:
        if raw_fh is not None:
            raw_fh.close()

    if record_wav:
        finalize_partial_wav(record_wav, sample_rate=sample_rate)

    attributions: list[dict[str, Any]] = []
    if detections and events_path:
        events = load_events(events_path)
        spikes = [{"t": d["ts_ms"] / 1000.0, "energy": d["peak_dbfs"], "wall_ts": d["wall_ts"]} for d in detections]
        attributions = correlate_spikes_with_events(spikes, events, window_s=3.0)
    else:
        attributions = [{"tool": None, "offset_s": None} for _ in detections]

    if detections:
        append_caution_entries(caution_list, detections, attributions)

    return {
        "seconds": seconds,
        "device": device_index,
        "detections": detections,
        "attributions": attributions,
        "caution_list": caution_list if detections else None,
        "record_wav": record_wav,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Listen for cowling-impact signatures and build the caution list")
    parser.add_argument("--seconds", type=float, default=120.0, help="listen window (default: %(default)s)")
    parser.add_argument("--events", default=None, help="Tier-0 events file for attribution")
    parser.add_argument("--caution-list", default=DEFAULT_CAUTION_LIST, help="caution list JSONL path")
    parser.add_argument(
        "--device",
        default=None,
        help="input device index or name substring, e.g. 'Brio' (default: ROBOT_AUDIO_DEVICE, then system default)",
    )
    parser.add_argument("--record-wav", default=None, help="also save the raw capture to this WAV for offline tuning")
    args = parser.parse_args()

    print(f"listening {args.seconds:g}s for impact signatures...", flush=True)
    report = watch(
        args.seconds,
        events_path=args.events,
        caution_list=args.caution_list,
        device=args.device,
        record_wav=args.record_wav,
    )
    print(f"{len(report['detections'])} detection(s)")
    for det, att in zip(report["detections"], report["attributions"]):
        tool = att.get("tool") or "UNATTRIBUTED"
        print(f"  t={det['ts_ms'] / 1000:.1f}s {det['peak_dbfs']} dBFS hf={det['hf_ratio']} -> {tool}")
    if report["caution_list"]:
        print(f"caution list updated: {report['caution_list']}")
    if report["record_wav"]:
        print(f"raw capture saved: {report['record_wav']}")


if __name__ == "__main__":
    main()
