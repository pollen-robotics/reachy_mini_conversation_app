"""Regression tests for the audio-driven head wobble behaviour."""

import math
import time
import base64
import threading
from typing import Any, List, Tuple
from collections.abc import Callable

import numpy as np

from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler


def _make_audio_chunk(duration_s: float = 0.3, frequency_hz: float = 220.0) -> str:
    """Generate a base64-encoded mono PCM16 sine wave."""
    sample_rate = 24000
    sample_count = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, sample_count, endpoint=False)
    wave = 0.6 * np.sin(2 * math.pi * frequency_hz * t)
    pcm = np.clip(wave * np.iinfo(np.int16).max, -32768, 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def _wait_for(predicate: Callable[[], bool], timeout: float = 0.6) -> bool:
    """Poll `predicate` until true or timeout."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _start_wobbler() -> Tuple[HeadWobbler, List[Tuple[float, Tuple[float, float, float, float, float, float]]]]:
    captured: List[Tuple[float, Tuple[float, float, float, float, float, float]]] = []

    def capture(offsets: Tuple[float, float, float, float, float, float]) -> None:
        captured.append((time.time(), offsets))

    wobbler = HeadWobbler(set_speech_offsets=capture)
    wobbler.start()
    return wobbler, captured


def test_reset_drops_pending_offsets() -> None:
    """Reset should stop wobble output derived from pre-reset audio."""
    wobbler, captured = _start_wobbler()
    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert _wait_for(lambda: len(captured) > 0), "wobbler did not emit initial offsets"

        pre_reset_count = len(captured)
        wobbler.reset()
        time.sleep(0.3)
        assert len(captured) == pre_reset_count, "offsets continued after reset without new audio"
    finally:
        wobbler.stop()


def test_reset_allows_future_offsets() -> None:
    """After reset, fresh audio must still produce wobble offsets."""
    wobbler, captured = _start_wobbler()
    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert _wait_for(lambda: len(captured) > 0), "wobbler did not emit initial offsets"

        wobbler.reset()
        pre_second_count = len(captured)

        wobbler.feed(_make_audio_chunk(duration_s=0.35, frequency_hz=440.0))
        assert _wait_for(lambda: len(captured) > pre_second_count), "no offsets after reset"
        assert wobbler._thread is not None and wobbler._thread.is_alive()
    finally:
        wobbler.stop()


def test_reset_during_inflight_chunk_keeps_worker(monkeypatch: Any) -> None:
    """Simulate reset during chunk processing to ensure the worker survives."""
    wobbler, captured = _start_wobbler()
    ready = threading.Event()
    release = threading.Event()

    original_feed = wobbler.sway.feed

    def blocking_feed(pcm, sr):  # type: ignore[no-untyped-def]
        ready.set()
        release.wait(timeout=2.0)
        return original_feed(pcm, sr)

    monkeypatch.setattr(wobbler.sway, "feed", blocking_feed)

    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert ready.wait(timeout=1.0), "worker thread did not dequeue audio"

        wobbler.reset()
        release.set()

        # Allow the worker to finish processing the first chunk (which should be discarded)
        time.sleep(0.1)

        assert wobbler._thread is not None and wobbler._thread.is_alive(), "worker thread died after reset"

        pre_second = len(captured)
        wobbler.feed(_make_audio_chunk(duration_s=0.35, frequency_hz=440.0))
        assert _wait_for(lambda: len(captured) > pre_second), "no offsets emitted after in-flight reset"
        assert wobbler._thread.is_alive()
    finally:
        wobbler.stop()
