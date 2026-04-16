"""Moves head given audio samples."""

import time
import queue
import base64
import logging
import threading
from typing import Tuple
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.audio.speech_tapper import HOP_MS, SwayRollRT


SAMPLE_RATE = 24000
MOVEMENT_LATENCY_S = 0.2  # seconds between audio and robot movement
logger = logging.getLogger(__name__)


class HeadWobbler:
    """Converts audio deltas (base64) into head movement offsets."""

    def __init__(self, set_speech_offsets: Callable[[Tuple[float, float, float, float, float, float]], None]) -> None:
        """Initialize the head wobbler."""
        self._apply_offsets = set_speech_offsets
        self._base_ts: float | None = None
        self._hops_done: int = 0

        self.audio_queue: "queue.Queue[Tuple[int, int, NDArray[np.int16], float]]" = queue.Queue()
        self.sway = SwayRollRT()

        # Synchronization primitives
        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._generation = 0
        self._reset_after_audio = False

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def feed(self, delta_b64: str, start_delay_s: float = 0.0) -> None:
        """Thread-safe: push base64 audio into the consumer queue."""
        buf = np.frombuffer(base64.b64decode(delta_b64), dtype=np.int16).reshape(1, -1)
        self.feed_pcm(buf, SAMPLE_RATE, start_delay_s=start_delay_s)

    def feed_pcm(self, pcm: NDArray[np.int16], sample_rate: int, start_delay_s: float = 0.0) -> None:
        """Thread-safe: push PCM audio into the consumer queue."""
        with self._state_lock:
            generation = self._generation
            self._reset_after_audio = False
        self.audio_queue.put((generation, sample_rate, pcm, max(0.0, start_delay_s)))

    def request_reset_after_current_audio(self) -> None:
        """Reset once the current generation finishes playing."""
        should_reset_now = False
        with self._state_lock:
            self._reset_after_audio = True
            should_reset_now = self._base_ts is None and self.audio_queue.empty()
        if should_reset_now:
            self.reset()

    def start(self) -> None:
        """Start the head wobbler loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Head wobbler started")

    def stop(self) -> None:
        """Stop the head wobbler loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.debug("Head wobbler stopped")

    def working_loop(self) -> None:
        """Convert audio deltas into head movement offsets."""
        hop_dt = HOP_MS / 1000.0

        logger.debug("Head wobbler thread started")
        while not self._stop_event.is_set():
            queue_ref = self.audio_queue
            try:
                chunk_generation, sr, chunk, start_delay_s = queue_ref.get(timeout=hop_dt)  # (gen, sr, data, start_delay)
            except queue.Empty:
                if self._should_reset_after_audio(hop_dt):
                    self.reset()
                continue

            try:
                with self._state_lock:
                    current_generation = self._generation
                if chunk_generation != current_generation:
                    continue

                if self._base_ts is None:
                    with self._state_lock:
                        if self._base_ts is None:
                            self._base_ts = time.monotonic() + start_delay_s

                pcm = np.asarray(chunk).squeeze(0)
                with self._sway_lock:
                    results = self.sway.feed(pcm, sr)

                i = 0
                while i < len(results):
                    with self._state_lock:
                        if self._generation != current_generation:
                            break
                        base_ts = self._base_ts
                        hops_done = self._hops_done

                    if base_ts is None:
                        base_ts = time.monotonic()
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = base_ts
                                hops_done = self._hops_done

                    target = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt
                    now = time.monotonic()

                    if now - target >= hop_dt:
                        lag_hops = int((now - target) / hop_dt)
                        drop = min(lag_hops, len(results) - i - 1)
                        if drop > 0:
                            with self._state_lock:
                                self._hops_done += drop
                                hops_done = self._hops_done
                            i += drop
                            continue

                    if target > now:
                        time.sleep(target - now)
                        with self._state_lock:
                            if self._generation != current_generation:
                                break

                    r = results[i]
                    offsets = (
                        r["x_mm"] / 1000.0,
                        r["y_mm"] / 1000.0,
                        r["z_mm"] / 1000.0,
                        r["roll_rad"],
                        r["pitch_rad"],
                        r["yaw_rad"],
                    )

                    with self._state_lock:
                        if self._generation != current_generation:
                            break

                    self._apply_offsets(offsets)

                    with self._state_lock:
                        self._hops_done += 1
                    i += 1
            finally:
                queue_ref.task_done()
        logger.debug("Head wobbler thread exited")

    def _should_reset_after_audio(self, hop_dt: float) -> bool:
        """Return True when a requested reset has reached the end of queued audio."""
        with self._state_lock:
            if not self._reset_after_audio or self._base_ts is None:
                return False
            if not self.audio_queue.empty():
                return False
            reset_at = self._base_ts + MOVEMENT_LATENCY_S + self._hops_done * hop_dt
        return time.monotonic() >= reset_at

    '''
    def drain_audio_queue(self) -> None:
        """Empty the audio queue."""
        try:
            while True:
                self.audio_queue.get_nowait()
        except QueueEmpty:
            pass
    '''

    def reset(self) -> None:
        """Reset the internal state."""
        with self._state_lock:
            self._generation += 1
            self._base_ts = None
            self._hops_done = 0
            self._reset_after_audio = False

        # Drain any queued audio chunks from previous generations
        drained_any = False
        while True:
            try:
                _, _, _, _ = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            else:
                drained_any = True
                self.audio_queue.task_done()

        with self._sway_lock:
            self.sway.reset()

        self._apply_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        if drained_any:
            logger.debug("Head wobbler queue drained during reset")
