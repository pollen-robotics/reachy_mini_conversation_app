"""Bonk detector (#522): audio impact signatures correlated into a caution list.

A bonk — the head striking the plastic cowling — reads as a short *broadband*
transient on the witness microphone: a sudden onset well above the rolling
background level with substantial high-frequency energy. Speech and TTS
onsets carry their energy mostly below ~2 kHz, so the high-frequency ratio
separates the two even when the robot is mid-riff.

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

from robot_comic.observer.audio_witness import dbfs
from robot_comic.observer.motion_witness import load_events, correlate_spikes_with_events


# Onset: the block's PEAK level must clear the rolling RMS background by this
# much. Impacts have a high crest factor, so peak-vs-RMS keeps a bonk visible
# even while the robot is talking over it.
ONSET_DB = float(os.getenv("ROBOT_BONK_ONSET_DB", "15"))
# …and the block's absolute level must clear this floor (quiet taps from the
# street don't qualify).
FLOOR_DBFS = float(os.getenv("ROBOT_BONK_FLOOR_DBFS", "-30"))
# Impacts are broadband: at least this fraction of block energy above HF_CUT_HZ.
HF_RATIO_MIN = float(os.getenv("ROBOT_BONK_HF_RATIO_MIN", "0.35"))
HF_CUT_HZ = float(os.getenv("ROBOT_BONK_HF_CUT_HZ", "2000"))
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
    """Pure block-fed impact detector: onset + floor + broadband signature."""

    def __init__(
        self,
        *,
        onset_db: float = ONSET_DB,
        floor_dbfs: float = FLOOR_DBFS,
        hf_ratio_min: float = HF_RATIO_MIN,
        hf_cut_hz: float = HF_CUT_HZ,
        debounce_ms: int = DEBOUNCE_MS,
    ) -> None:
        """Configure the impact signature thresholds."""
        self.onset_db = onset_db
        self.floor_dbfs = floor_dbfs
        self.hf_ratio_min = hf_ratio_min
        self.hf_cut_hz = hf_cut_hz
        self.debounce_ms = debounce_ms
        self._recent_db: list[float] = []
        self._last_hit_ms: int | None = None

    def update(self, *, ts_ms: int, block: NDArray[np.float32], sample_rate: int) -> dict[str, Any] | None:
        """Feed one block; return a detection dict when an impact just landed."""
        level = _peak_dbfs(block)
        background = float(np.median(self._recent_db)) if self._recent_db else -120.0
        # Background history tracks RMS so sustained speech sets the floor the
        # peak must clear; the impact block itself still enters the history.
        self._recent_db.append(dbfs(block))
        if len(self._recent_db) > _HISTORY_BLOCKS:
            self._recent_db.pop(0)

        if level < self.floor_dbfs:
            return None
        if level - background < self.onset_db:
            return None
        if self._last_hit_ms is not None and (ts_ms - self._last_hit_ms) < self.debounce_ms:
            return None
        ratio = _hf_ratio(block, sample_rate, self.hf_cut_hz)
        if ratio < self.hf_ratio_min:
            return None

        self._last_hit_ms = ts_ms
        return {
            "ts_ms": ts_ms,
            "peak_dbfs": round(level, 1),
            "onset_db": round(level - background, 1),
            "hf_ratio": round(ratio, 3),
        }


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


def watch(
    seconds: float,
    *,
    events_path: str | None = None,
    caution_list: str = DEFAULT_CAUTION_LIST,
    sample_rate: int = 16_000,
    block_ms: int = 50,
) -> dict[str, Any]:
    """Listen on the witness mic for ``seconds``; detect, attribute, persist."""
    import sounddevice as sd

    detector = BonkDetector()
    block_size = int(sample_rate * block_ms / 1000)
    detections: list[dict[str, Any]] = []
    start_wall = time.time()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_size) as stream:
        n_blocks = int(math.ceil(seconds * 1000 / block_ms))
        for _ in range(n_blocks):
            block, _overflowed = stream.read(block_size)
            ts_ms = int((time.time() - start_wall) * 1000)
            hit = detector.update(ts_ms=ts_ms, block=block[:, 0], sample_rate=sample_rate)
            if hit is not None:
                hit["wall_ts"] = int((start_wall + hit["ts_ms"] / 1000.0) * 1000)
                detections.append(hit)
                print(
                    f"BONK? t={hit['ts_ms'] / 1000:.1f}s level={hit['peak_dbfs']} dBFS "
                    f"onset=+{hit['onset_db']} dB hf={hit['hf_ratio']}",
                    flush=True,
                )

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
        "detections": detections,
        "attributions": attributions,
        "caution_list": caution_list if detections else None,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Listen for cowling-impact signatures and build the caution list")
    parser.add_argument("--seconds", type=float, default=120.0, help="listen window (default: %(default)s)")
    parser.add_argument("--events", default=None, help="Tier-0 events file for attribution")
    parser.add_argument("--caution-list", default=DEFAULT_CAUTION_LIST, help="caution list JSONL path")
    args = parser.parse_args()

    print(f"listening {args.seconds:g}s for impact signatures...", flush=True)
    report = watch(args.seconds, events_path=args.events, caution_list=args.caution_list)
    print(f"{len(report['detections'])} detection(s)")
    for det, att in zip(report["detections"], report["attributions"]):
        tool = att.get("tool") or "UNATTRIBUTED"
        print(f"  t={det['ts_ms'] / 1000:.1f}s {det['peak_dbfs']} dBFS hf={det['hf_ratio']} -> {tool}")
    if report["caution_list"]:
        print(f"caution list updated: {report['caution_list']}")


if __name__ == "__main__":
    main()
