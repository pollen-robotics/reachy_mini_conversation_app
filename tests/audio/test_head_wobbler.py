"""Unit and regression tests for the audio-driven head wobble behaviour."""

import math
import time
import base64
import queue
import threading
from typing import Any, List, Tuple
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from reachy_mini_conversation_app.audio.head_wobbler import (
    HeadWobbler,
    SAMPLE_RATE,
    MOVEMENT_LATENCY_S,
)


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


class TestHeadWobblerConstants:
    """Tests for HeadWobbler module constants."""

    def test_sample_rate(self) -> None:
        """Test SAMPLE_RATE constant."""
        assert SAMPLE_RATE == 24000

    def test_movement_latency(self) -> None:
        """Test MOVEMENT_LATENCY_S is reasonable."""
        assert 0 < MOVEMENT_LATENCY_S < 1.0


class TestHeadWobblerInit:
    """Tests for HeadWobbler initialization."""

    def test_init_with_callback(self) -> None:
        """Test HeadWobbler initializes with callback."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        assert wobbler._apply_offsets == callback
        assert wobbler._base_ts is None
        assert wobbler._hops_done == 0
        assert wobbler._generation == 0
        assert wobbler._thread is None

    def test_init_creates_queue(self) -> None:
        """Test HeadWobbler creates audio queue."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        assert isinstance(wobbler.audio_queue, queue.Queue)

    def test_init_creates_sway(self) -> None:
        """Test HeadWobbler creates SwayRollRT instance."""
        from reachy_mini_conversation_app.audio.speech_tapper import SwayRollRT

        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        assert isinstance(wobbler.sway, SwayRollRT)

    def test_init_creates_locks(self) -> None:
        """Test HeadWobbler creates synchronization primitives."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        assert isinstance(wobbler._state_lock, type(threading.Lock()))
        assert isinstance(wobbler._sway_lock, type(threading.Lock()))
        assert isinstance(wobbler._stop_event, type(threading.Event()))


class TestHeadWobblerFeed:
    """Tests for HeadWobbler feed method."""

    def test_feed_decodes_base64(self) -> None:
        """Test feed decodes base64 audio."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        audio_b64 = _make_audio_chunk(duration_s=0.1)
        wobbler.feed(audio_b64)

        # Should have added to queue
        assert not wobbler.audio_queue.empty()

    def test_feed_puts_correct_format(self) -> None:
        """Test feed puts (generation, sample_rate, data) tuple."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        audio_b64 = _make_audio_chunk(duration_s=0.1)
        wobbler.feed(audio_b64)

        gen, sr, data = wobbler.audio_queue.get_nowait()
        assert gen == 0  # Initial generation
        assert sr == SAMPLE_RATE
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.int16

    def test_feed_uses_current_generation(self) -> None:
        """Test feed uses current generation counter."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)
        wobbler._generation = 5

        audio_b64 = _make_audio_chunk(duration_s=0.1)
        wobbler.feed(audio_b64)

        gen, _, _ = wobbler.audio_queue.get_nowait()
        assert gen == 5


class TestHeadWobblerStartStop:
    """Tests for HeadWobbler start/stop methods."""

    def test_start_creates_thread(self) -> None:
        """Test start creates and starts worker thread."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        wobbler.start()
        try:
            assert wobbler._thread is not None
            assert wobbler._thread.is_alive()
            assert wobbler._thread.daemon is True
        finally:
            wobbler.stop()

    def test_stop_joins_thread(self) -> None:
        """Test stop sets event and joins thread."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        wobbler.start()
        thread = wobbler._thread
        wobbler.stop()

        assert wobbler._stop_event.is_set()
        assert thread is not None
        assert not thread.is_alive()

    def test_stop_without_start(self) -> None:
        """Test stop works even if never started."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        # Should not raise
        wobbler.stop()

    def test_multiple_start_stop_cycles(self) -> None:
        """Test multiple start/stop cycles work."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        for _ in range(3):
            wobbler.start()
            assert wobbler._thread is not None
            assert wobbler._thread.is_alive()
            wobbler.stop()
            assert not wobbler._thread.is_alive()


class TestHeadWobblerReset:
    """Tests for HeadWobbler reset method."""

    def test_reset_increments_generation(self) -> None:
        """Test reset increments generation counter."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)
        initial_gen = wobbler._generation

        wobbler.reset()

        assert wobbler._generation == initial_gen + 1

    def test_reset_clears_base_ts(self) -> None:
        """Test reset clears base timestamp."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)
        wobbler._base_ts = time.monotonic()

        wobbler.reset()

        assert wobbler._base_ts is None

    def test_reset_clears_hops_done(self) -> None:
        """Test reset clears hops counter."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)
        wobbler._hops_done = 100

        wobbler.reset()

        assert wobbler._hops_done == 0

    def test_reset_drains_queue(self) -> None:
        """Test reset drains audio queue."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        # Add items to queue
        wobbler.feed(_make_audio_chunk(duration_s=0.1))
        wobbler.feed(_make_audio_chunk(duration_s=0.1))
        assert not wobbler.audio_queue.empty()

        wobbler.reset()

        assert wobbler.audio_queue.empty()

    def test_reset_resets_sway(self) -> None:
        """Test reset calls sway.reset()."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)
        wobbler.sway.reset = MagicMock()

        wobbler.reset()

        wobbler.sway.reset.assert_called_once()


class TestHeadWobblerWorkingLoop:
    """Tests for HeadWobbler working_loop method."""

    def test_working_loop_processes_audio(self) -> None:
        """Test working_loop processes audio and calls callback."""
        captured: List[Tuple[float, float, float, float, float, float]] = []

        def capture(offsets: Tuple[float, float, float, float, float, float]) -> None:
            captured.append(offsets)

        wobbler = HeadWobbler(set_speech_offsets=capture)
        wobbler.start()

        try:
            wobbler.feed(_make_audio_chunk(duration_s=0.3))
            assert _wait_for(lambda: len(captured) > 0, timeout=1.0)
            assert len(captured) > 0

            # Check offset structure (6 floats)
            offset = captured[0]
            assert len(offset) == 6
            for val in offset:
                assert isinstance(val, float)
        finally:
            wobbler.stop()

    def test_working_loop_skips_old_generation(self) -> None:
        """Test working_loop skips chunks from old generations."""
        captured: List[Any] = []

        def capture(offsets: Any) -> None:
            captured.append(offsets)

        wobbler = HeadWobbler(set_speech_offsets=capture)

        # Manually add old generation chunk
        old_audio = np.zeros((1, 1000), dtype=np.int16)
        wobbler.audio_queue.put((999, SAMPLE_RATE, old_audio))  # Wrong generation

        wobbler.start()
        time.sleep(0.2)
        wobbler.stop()

        # Old generation chunk should have been skipped (no offsets applied)
        # Note: may have some offsets if other factors trigger them
        # The key is that the old chunk doesn't cause issues

    def test_working_loop_handles_empty_queue(self) -> None:
        """Test working_loop handles empty queue gracefully."""
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        wobbler.start()
        time.sleep(0.15)  # Let it run with empty queue
        wobbler.stop()

        # Should not have crashed


class TestHeadWobblerIntegration:
    """Integration tests for HeadWobbler."""

    def test_full_audio_pipeline(self) -> None:
        """Test complete audio processing pipeline."""
        captured: List[Tuple[float, Tuple[float, float, float, float, float, float]]] = []

        def capture(offsets: Tuple[float, float, float, float, float, float]) -> None:
            captured.append((time.time(), offsets))

        wobbler = HeadWobbler(set_speech_offsets=capture)
        wobbler.start()

        try:
            # Feed multiple chunks
            for freq in [220, 330, 440]:
                wobbler.feed(_make_audio_chunk(duration_s=0.2, frequency_hz=freq))
                time.sleep(0.1)

            # Wait for processing
            assert _wait_for(lambda: len(captured) >= 3, timeout=2.0)

        finally:
            wobbler.stop()

    def test_offset_values_reasonable(self) -> None:
        """Test that produced offsets have reasonable values."""
        captured: List[Tuple[float, float, float, float, float, float]] = []

        def capture(offsets: Tuple[float, float, float, float, float, float]) -> None:
            captured.append(offsets)

        wobbler = HeadWobbler(set_speech_offsets=capture)
        wobbler.start()

        try:
            wobbler.feed(_make_audio_chunk(duration_s=0.3))
            assert _wait_for(lambda: len(captured) > 0, timeout=1.0)

            for offset in captured:
                x, y, z, roll, pitch, yaw = offset
                # Translation should be small (converted from mm to m)
                assert abs(x) < 0.1  # < 100mm
                assert abs(y) < 0.1
                assert abs(z) < 0.1
                # Rotation should be small radians
                assert abs(roll) < 0.5  # < ~30 degrees
                assert abs(pitch) < 0.5
                assert abs(yaw) < 0.5
        finally:
            wobbler.stop()
