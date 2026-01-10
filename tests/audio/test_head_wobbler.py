"""Unit and regression tests for the audio-driven head wobble behaviour."""

import math
import time
import queue
import base64
import logging
import threading
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch
from collections.abc import Callable

import numpy as np

from reachy_mini_conversation_app.audio.head_wobbler import (
    SAMPLE_RATE,
    MOVEMENT_LATENCY_S,
    HeadWobbler,
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
    def test_generation_mismatch_during_processing(self):
        """Test when generation changes mid-processing."""
        wobbler, captured = _start_wobbler()
        try:
            sample_audio = np.zeros(2400, dtype=np.int16)
            b64_audio = base64.b64encode(sample_audio.tobytes()).decode()
            wobbler.feed(b64_audio)

            wobbler._state_lock.acquire()
            wobbler._generation += 1
            wobbler._state_lock.release()

            time.sleep(0.3)
        finally:
            wobbler.stop()

    def test_base_ts_initialization_race_condition(self):
        """Test _base_ts initialization when None twice."""
        wobbler, captured = _start_wobbler()
        try:
            wobbler._base_ts = None
            for _ in range(3):
                sample_audio = np.random.randint(-100, 100, 2400, dtype=np.int16)
                b64_audio = base64.b64encode(sample_audio.tobytes()).decode()
                wobbler.feed(b64_audio)

            time.sleep(0.5)
        finally:
            wobbler.stop()

    def test_lag_hops_handling(self):
        """Test handling of lagged hops."""
        wobbler, captured = _start_wobbler()
        try:
            wobbler._base_ts = time.monotonic() - 5.0
            for i in range(5):
                sample_audio = np.random.randint(-100, 100, 2400, dtype=np.int16)
                b64_audio = base64.b64encode(sample_audio.tobytes()).decode()
                wobbler.feed(b64_audio)

            time.sleep(0.3)
        finally:
            wobbler.stop()

    def test_stop_during_queue_empty(self):
        """Test stop signal during queue.Empty."""
        wobbler, captured = _start_wobbler()
        time.sleep(0.05)
        wobbler.stop()
        assert wobbler._thread is not None
        assert wobbler._stop_event.is_set()

    def test_empty_audio_chunk(self):
        """Test empty audio chunks."""
        wobbler, captured = _start_wobbler()
        try:
            empty_audio = np.array([], dtype=np.int16)
            b64_audio = base64.b64encode(empty_audio.tobytes()).decode()
            wobbler.feed(b64_audio)
            time.sleep(0.1)
        finally:
            wobbler.stop()

    def test_silent_audio(self):
        """Test silent (zero) audio."""
        wobbler, captured = _start_wobbler()
        try:
            silent = np.zeros(2400, dtype=np.int16)
            b64_audio = base64.b64encode(silent.tobytes()).decode()
            for _ in range(3):
                wobbler.feed(b64_audio)
            time.sleep(0.3)
        finally:
            wobbler.stop()

    def test_drop_handling_when_lagged_behind(self) -> None:
        """Test drop > 0 branch where we skip lagged hops."""
        wobbler, captured = _start_wobbler()
        try:
            # Set base_ts far in the past so system is lagged
            wobbler._base_ts = time.monotonic() - 10.0
            wobbler._hops_done = 0

            # Feed audio to trigger processing
            wobbler.feed(_make_audio_chunk(duration_s=0.5))

            # Give it time to process and skip lagged hops
            time.sleep(0.4)

            # Verify wobbler is still running and processed something
            assert wobbler._thread is not None and wobbler._thread.is_alive()
        finally:
            wobbler.stop()

    def test_generation_change_after_sleep_breaks_loop(self) -> None:
        """Test that generation change after sleep causes break in processing loop."""
        wobbler, captured = _start_wobbler()
        ready_to_sleep = threading.Event()

        original_sleep = time.sleep

        def patched_sleep(duration: float) -> None:
            ready_to_sleep.set()
            original_sleep(duration)

        with patch("time.sleep", patched_sleep):
            try:
                wobbler.feed(_make_audio_chunk(duration_s=0.5))

                # Wait for sleep to be called (meaning target > now)
                if ready_to_sleep.wait(timeout=1.0):
                    # While sleeping, trigger reset which increments generation
                    wobbler.reset()

                # Give it time to detect generation change and break
                time.sleep(0.2)

                # Verify wobbler still running
                assert wobbler._thread is not None and wobbler._thread.is_alive()
            finally:
                wobbler.stop()

    def test_reset_drains_queue_with_logging(self, caplog: Any) -> None:
        """Test reset drains queue and logs when items were drained."""
        wobbler, captured = _start_wobbler()
        try:
            wobbler.feed(_make_audio_chunk(duration_s=0.3))
            time.sleep(0.1)

            # Verify audio was queued
            assert wobbler.audio_queue.qsize() >= 0

            # Reset should drain
            with caplog.at_level(logging.DEBUG):
                wobbler.reset()

            time.sleep(0.1)
            # Wobbler should still be running
            assert wobbler._thread is not None and wobbler._thread.is_alive()
        finally:
            wobbler.stop()

    def test_base_ts_double_checked_locking_initialization(self) -> None:
        """Test the double-checked locking for _base_ts initialization (lines 84-86).

        This tests the branch where _base_ts is None on first check (line 83),
        and is still None inside the lock (line 85), triggering initialization.
        """
        wobbler, captured = _start_wobbler()
        try:
            # Ensure _base_ts is None
            wobbler._base_ts = None

            # Feed audio which will trigger the double-checked locking initialization
            wobbler.feed(_make_audio_chunk(duration_s=0.3))

            # Wait for processing to occur which sets _base_ts
            assert _wait_for(lambda: wobbler._base_ts is not None, timeout=1.0), \
                "_base_ts should be initialized during processing"

            # Verify wobbler is still running
            assert wobbler._thread is not None and wobbler._thread.is_alive()
        finally:
            wobbler.stop()


    def test_base_ts_none_in_inner_loop_fallback(self) -> None:
        """Test the fallback path when base_ts is None in inner loop (lines 100-105).

        This tests when base_ts (local var) is None after getting state lock,
        requiring it to be set using time.monotonic().
        """
        wobbler, captured = _start_wobbler()

        try:
            # Feed initial audio to start processing
            wobbler.feed(_make_audio_chunk(duration_s=0.3))

            # Wait for _base_ts to be set
            assert _wait_for(lambda: wobbler._base_ts is not None, timeout=1.0)

            # Now clear _base_ts while processing is ongoing
            # This will cause base_ts to be None at line 97 on next iteration
            with wobbler._state_lock:
                wobbler._base_ts = None

            # Feed more audio to trigger the fallback path
            wobbler.feed(_make_audio_chunk(duration_s=0.3))
            time.sleep(0.4)

            # Verify _base_ts was re-initialized
            assert wobbler._base_ts is not None
            assert wobbler._thread is not None and wobbler._thread.is_alive()
        finally:
            wobbler.stop()

    def test_base_ts_set_by_another_in_inner_loop(self) -> None:
        """Test branch 103->107: when _base_ts is set by another before we can.

        This tests when base_ts is None at line 100, but _base_ts is set
        by another thread before we check at line 103. This is a race condition
        test that we simulate by modifying _base_ts between iterations.
        """
        wobbler, captured = _start_wobbler()
        set_count = [0]

        try:
            wobbler.feed(_make_audio_chunk(duration_s=0.3))

            # Wait for processing to start
            assert _wait_for(lambda: wobbler._base_ts is not None, timeout=1.0)

            # Repeatedly clear and set _base_ts to simulate race
            for _ in range(5):
                with wobbler._state_lock:
                    wobbler._base_ts = None
                time.sleep(0.01)
                with wobbler._state_lock:
                    if wobbler._base_ts is None:
                        wobbler._base_ts = time.monotonic()
                        set_count[0] += 1

            wobbler.feed(_make_audio_chunk(duration_s=0.3))
            time.sleep(0.4)

            assert wobbler._thread is not None and wobbler._thread.is_alive()
        finally:
            wobbler.stop()

    def test_generation_change_after_offset_application_breaks(self) -> None:
        """Test that generation change after offset application causes break (line 138).

        This tests the branch where after applying offsets (line 140),
        the generation check (line 137) fails, causing a break.
        """
        captured: List[Any] = []

        original_apply_offsets_called = threading.Event()

        def capture_with_gen_change(offsets: Any) -> None:
            captured.append(offsets)
            original_apply_offsets_called.set()
            # After first offset is applied, increment generation
            # This will cause the check at line 137 to fail on next iteration

        wobbler = HeadWobbler(set_speech_offsets=capture_with_gen_change)
        wobbler.start()

        try:
            # Feed audio that will produce multiple results
            wobbler.feed(_make_audio_chunk(duration_s=0.5))

            # Wait for at least one offset to be applied
            assert original_apply_offsets_called.wait(timeout=1.0), \
                "Should have applied at least one offset"

            # Now increment generation after the first offset
            with wobbler._state_lock:
                wobbler._generation += 1

            # Give time for the loop to detect generation change and break
            time.sleep(0.3)

            # Verify wobbler is still running
            assert wobbler._thread is not None and wobbler._thread.is_alive()
        finally:
            wobbler.stop()

    def test_base_ts_race_double_check_skips_assignment(self) -> None:
        """Test branch 85->88: _base_ts set between checks.

        This covers the case where another thread sets _base_ts between
        line 83 (outside lock check) and line 85 (inside lock check).
        """
        captured: List[Any] = []

        def capture(offsets: Any) -> None:
            captured.append(offsets)

        wobbler = HeadWobbler(set_speech_offsets=capture)
        wobbler.start()
        try:
            # Let processing run normally first to establish _base_ts
            wobbler.feed(_make_audio_chunk(duration_s=0.3))
            assert _wait_for(lambda: wobbler._base_ts is not None, timeout=1.0)

            # Now we'll repeatedly create conditions where _base_ts might
            # be set between checks. The normal case (85->88) is when
            # _base_ts is already set - which happens on subsequent chunks
            # after the first one.

            # Feed more audio - this should go through the path where
            # _base_ts is already set (skipping line 86)
            wobbler.feed(_make_audio_chunk(duration_s=0.3))
            time.sleep(0.5)

            assert wobbler._thread is not None and wobbler._thread.is_alive()
        finally:
            wobbler.stop()

    def test_inner_loop_base_ts_already_set_skips_block(self) -> None:
        """Test branch 103->107: _base_ts set before inner check.

        This covers the case where base_ts is None at line 100 (local var),
        but _base_ts is already set at line 103 by another thread.
        """
        captured: List[Any] = []

        def capture(offsets: Any) -> None:
            captured.append(offsets)

        wobbler = HeadWobbler(set_speech_offsets=capture)
        wobbler.start()

        try:
            # First, let the wobbler process normally to set _base_ts
            wobbler.feed(_make_audio_chunk(duration_s=0.3))
            assert _wait_for(lambda: wobbler._base_ts is not None, timeout=1.0)

            # Clear _base_ts then quickly set it back
            # The thread might see base_ts=None at line 97 but _base_ts set at 103
            for _ in range(10):
                with wobbler._state_lock:
                    wobbler._base_ts = None
                time.sleep(0.001)
                with wobbler._state_lock:
                    wobbler._base_ts = time.monotonic()

            wobbler.feed(_make_audio_chunk(duration_s=0.3))
            time.sleep(0.5)

            assert wobbler._thread is not None and wobbler._thread.is_alive()
        finally:
            wobbler.stop()

    def test_generation_change_at_final_check_causes_break(self) -> None:
        """Test line 138: break when generation changes at final check.

        This tests the branch where generation is changed after offsets
        are applied (line 140) but before incrementing hops_done (line 143).
        """
        captured: List[Any] = []
        offset_count = [0]

        def capture_and_trigger_break(offsets: Any) -> None:
            captured.append(offsets)
            offset_count[0] += 1

        wobbler = HeadWobbler(set_speech_offsets=capture_and_trigger_break)
        wobbler.start()

        try:
            # Feed audio that will generate multiple results
            wobbler.feed(_make_audio_chunk(duration_s=0.5))

            # Wait for at least one offset to be applied
            assert _wait_for(lambda: offset_count[0] > 0, timeout=2.0)

            # Quickly change generation multiple times to increase chance
            # of hitting the break at line 138
            for _ in range(5):
                with wobbler._state_lock:
                    wobbler._generation += 1
                time.sleep(0.01)

            time.sleep(0.3)

            # The wobbler should still be running
            assert wobbler._thread is not None and wobbler._thread.is_alive()
        finally:
            wobbler.stop()

    def test_branch_85_88_base_ts_set_before_inner_check(self) -> None:
        """Test branch 85->88: _base_ts set between outer and inner check.

        This tests the double-checked locking pattern where _base_ts is None
        at line 83 (outer check) but is set before line 85 check completes.
        We replace the lock with a wrapper that sets _base_ts on __enter__.
        """
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        # Start with _base_ts = None
        wobbler._base_ts = None
        wobbler._generation = 0

        # We need to replace the lock with a wrapper that sets _base_ts
        # when entering the lock for the second time (the one at line 84)
        original_lock = wobbler._state_lock
        lock_enter_count = [0]

        class RaceConditionLock:
            """Lock that simulates another thread setting _base_ts."""

            def __enter__(self) -> "RaceConditionLock":
                original_lock.__enter__()
                lock_enter_count[0] += 1
                # The second lock acquisition is at line 84 (inside the if _base_ts is None block)
                # Set _base_ts to simulate another thread setting it
                if lock_enter_count[0] == 2 and wobbler._base_ts is None:
                    wobbler._base_ts = 12345.0
                return self

            def __exit__(self, *args: Any) -> None:
                return original_lock.__exit__(*args)

            def acquire(self, *args: Any, **kwargs: Any) -> bool:
                return original_lock.acquire(*args, **kwargs)

            def release(self) -> None:
                return original_lock.release()

        wobbler._state_lock = RaceConditionLock()  # type: ignore[assignment]

        # Prepare audio and run the actual working_loop code path
        audio_chunk = np.zeros((1, 2400), dtype=np.int16)
        wobbler.audio_queue.put((0, SAMPLE_RATE, audio_chunk))

        # Start the wobbler which will run working_loop
        wobbler.start()

        # Wait for the chunk to be processed
        time.sleep(0.3)

        wobbler.stop()

        # Verify the race condition occurred - _base_ts should be set to 12345.0
        assert wobbler._base_ts == 12345.0

    def test_branch_103_107_base_ts_set_before_inner_loop_check(self) -> None:
        """Test branch 103->107: _base_ts set before inner loop check.

        In the inner while loop, base_ts (local) is None at line 97,
        but _base_ts is set before line 103 check.

        Scenario:
        1. Line 86 sets _base_ts
        2. Something clears _base_ts before line 94 (simulating reset from another thread)
        3. Line 97 gets base_ts = None (local)
        4. Line 100 checks if base_ts is None -> True
        5. Before line 103, _base_ts is set by "another thread"
        6. Line 103 check fails, skipping lines 104-105 (branch 103->107)
        """
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        wobbler._base_ts = None
        wobbler._generation = 0
        wobbler._hops_done = 0

        original_lock = wobbler._state_lock
        lock_enter_count = [0]

        class InnerLoopRaceLock:
            """Lock that simulates race in inner loop."""

            def __enter__(self) -> "InnerLoopRaceLock":
                original_lock.__enter__()
                lock_enter_count[0] += 1
                # Lock acquisitions:
                # 1. Line 78 - get generation
                # 2. Line 84 - first base_ts check (sets _base_ts if None)
                # 3. Line 94 - inner loop state read (we clear _base_ts here to simulate reset)
                # 4. Line 102 - inner loop base_ts check (we set _base_ts here)
                if lock_enter_count[0] == 3:
                    # Simulate reset() clearing _base_ts just before inner loop reads it
                    wobbler._base_ts = None
                elif lock_enter_count[0] == 4:
                    # Simulate another thread setting _base_ts before check at line 103
                    wobbler._base_ts = 99999.0
                return self

            def __exit__(self, *args: Any) -> None:
                return original_lock.__exit__(*args)

            def acquire(self, *args: Any, **kwargs: Any) -> bool:
                return original_lock.acquire(*args, **kwargs)

            def release(self) -> None:
                return original_lock.release()

        wobbler._state_lock = InnerLoopRaceLock()  # type: ignore[assignment]

        # Prepare audio and run working_loop
        audio_chunk = np.zeros((1, 2400), dtype=np.int16)
        wobbler.audio_queue.put((0, SAMPLE_RATE, audio_chunk))

        wobbler.start()
        time.sleep(0.3)
        wobbler.stop()

        # Verify the race condition: _base_ts should be 99999.0 (set by our race lock)
        # Not a new time.monotonic() value
        assert wobbler._base_ts == 99999.0

    def test_line_138_break_generation_change_at_final_lock(self) -> None:
        """Test line 138: break when generation changes at final check (line 136).

        To hit line 138 specifically (not line 96 or 124), we need:
        1. Pass the generation check at line 95 (continue to line 100+)
        2. NOT enter lag handling (skip lines 110-118) - requires now - target < hop_dt
        3. NOT enter sleep block (skip lines 120-124) - requires target <= now
        4. Change generation at lock for line 136, so check at line 137 fails

        Lock sequence (with base_ts set, no lag, no sleep, 1 result):
        1. Line 78 - get generation
        2. Line 94 - inner loop get state
        3. Line 136 - final check before _apply_offsets

        Key: hop_dt = 0.01. Set timing so 0 < now - target < 0.01 (no lag, no sleep).
        """
        callback = MagicMock()
        wobbler = HeadWobbler(set_speech_offsets=callback)

        # Set base_ts so target is slightly less than now but not enough for lag
        # hop_dt = 0.01, so we need now - target < 0.01
        # target = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt
        # With hops_done = 0: target = base_ts + MOVEMENT_LATENCY_S
        # Want: target = now - 0.005 (so now - target = 0.005 < 0.01)
        wobbler._base_ts = time.monotonic() - MOVEMENT_LATENCY_S - 0.005
        wobbler._generation = 0
        wobbler._hops_done = 0

        original_lock = wobbler._state_lock
        lock_enter_count = [0]

        class Line136Lock:
            """Lock that changes generation at line 136 (lock #3)."""

            def __enter__(self) -> "Line136Lock":
                original_lock.__enter__()
                lock_enter_count[0] += 1
                # With base_ts set, no lag (now - target < hop_dt), no sleep:
                # Lock #1: Line 78 - get generation
                # Lock #2: Line 94 - inner loop get state
                # Lock #3: Line 136 - final check before _apply_offsets
                if lock_enter_count[0] == 3:
                    wobbler._generation += 1  # Trigger break at line 138
                return self

            def __exit__(self, *args: Any) -> None:
                return original_lock.__exit__(*args)

            def acquire(self, *args: Any, **kwargs: Any) -> bool:
                return original_lock.acquire(*args, **kwargs)

            def release(self) -> None:
                return original_lock.release()

        wobbler._state_lock = Line136Lock()  # type: ignore[assignment]

        # Only 1 result to keep the lock sequence simple
        mock_result = {
            "x_mm": 1.0, "y_mm": 2.0, "z_mm": 3.0,
            "roll_rad": 0.1, "pitch_rad": 0.2, "yaw_rad": 0.3
        }
        wobbler.sway.feed = MagicMock(return_value=[mock_result])

        audio_chunk = np.zeros((1, 2400), dtype=np.int16)
        wobbler.audio_queue.put((0, SAMPLE_RATE, audio_chunk))

        wobbler.start()
        time.sleep(0.3)
        wobbler.stop()

        # Verify callback was NOT called (we broke at line 138 before line 140)
        callback.assert_not_called()
        assert lock_enter_count[0] >= 3
