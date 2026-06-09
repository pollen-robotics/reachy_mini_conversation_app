"""Optional latency probes for the full conversation app."""

from __future__ import annotations
import os

import numpy as np
from numpy.typing import NDArray


POST_ASSISTANT_BEEP_ROLE = "latency_probe"
POST_ASSISTANT_BEEP_CONTENT = "post_assistant_beep"
POST_ASSISTANT_BEEP_ENV = "REACHY_MINI_LATENCY_PROBE_BEEP"
RECORDING_STATS_ENV = "REACHY_MINI_LATENCY_PROBE_RECORDING"
PROBE_BEEP_FREQUENCY_HZ = 1000.0
PROBE_BEEP_DETECTION_MIN_RMS = 0.01
PROBE_BEEP_DETECTION_MIN_SCORE = 0.65
POST_ASSISTANT_BEEP_GAP_S = 0.35
AUDIBLE_AUDIO_RMS_THRESHOLD = 0.006
SLOW_FIRST_AUDIBLE_AUDIO_MS = 2500.0


def _env_flag(name: str) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def post_assistant_beep_enabled() -> bool:
    """Return whether to play a diagnostic beep after assistant audio is queued."""
    return _env_flag(POST_ASSISTANT_BEEP_ENV)


def recording_stats_enabled() -> bool:
    """Return whether to log record-loop timing diagnostics."""
    return _env_flag(RECORDING_STATS_ENV)


def normalized_audio_rms(audio_frame: NDArray[np.generic]) -> float:
    """Return an approximate normalized RMS for int or float audio."""
    audio = np.asarray(audio_frame)
    if audio.size == 0:
        return 0.0
    if audio.ndim > 1:
        audio = audio.astype(np.float32).mean(axis=0 if audio.shape[0] < audio.shape[-1] else 1)
    else:
        audio = audio.astype(np.float32)
    if np.issubdtype(np.asarray(audio_frame).dtype, np.integer):
        audio = audio / float(np.iinfo(np.asarray(audio_frame).dtype).max)
    audio = audio.reshape(-1)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


def probe_beep_score(audio_frame: NDArray[np.float32], sample_rate: int) -> tuple[float, float]:
    """Return a simple 1 kHz tone score and RMS for a recorder frame."""
    audio = np.asarray(audio_frame, dtype=np.float32)
    if audio.size == 0:
        return 0.0, 0.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.reshape(-1)
    if audio.size < 32:
        return 0.0, 0.0

    audio = audio - float(np.mean(audio))
    rms = float(np.sqrt(np.mean(audio * audio)))
    if rms <= 0.0:
        return 0.0, 0.0

    t = np.arange(audio.size, dtype=np.float32) / sample_rate
    sine = np.sin(2 * np.pi * PROBE_BEEP_FREQUENCY_HZ * t)
    cosine = np.cos(2 * np.pi * PROBE_BEEP_FREQUENCY_HZ * t)
    tone_level = 2.0 * float(np.hypot(np.dot(audio, sine), np.dot(audio, cosine))) / audio.size
    return tone_level / rms, rms


def is_probe_beep_detected(score: float, rms: float) -> bool:
    """Return whether a recorder frame looks like the diagnostic beep."""
    return rms >= PROBE_BEEP_DETECTION_MIN_RMS and score >= PROBE_BEEP_DETECTION_MIN_SCORE


def make_probe_beep(sample_rate: int, *, channels: int = 1) -> NDArray[np.float32]:
    """Build a short two-beep pulse for audible latency checks."""
    beep_s = 0.08
    gap_s = 0.12
    amplitude = 0.35

    t = np.arange(int(sample_rate * beep_s), dtype=np.float32) / sample_rate
    beep = amplitude * np.sin(2 * np.pi * PROBE_BEEP_FREQUENCY_HZ * t)
    gap = np.zeros(int(sample_rate * gap_s), dtype=np.float32)
    mono = np.concatenate([beep, gap, beep]).astype(np.float32)
    if channels <= 1:
        return mono
    return np.repeat(mono[:, None], channels, axis=1)
