"""Tests for the bonk detector (#522): audio impact signatures → caution list.

A bonk — the head striking the plastic cowling — is a short *broadband*
transient: a sudden onset well above the rolling background with substantial
high-frequency energy (unlike speech onsets, whose energy sits mostly below
2 kHz). The detector is pure (fed blocks, no audio deps) so the signature
logic is unit-testable; capture and caution-list persistence are seams.
"""

from __future__ import annotations

import numpy as np

from robot_comic.observer.bonk_detector import BonkDetector, append_caution_entries


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
