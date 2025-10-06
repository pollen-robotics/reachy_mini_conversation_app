"""Moves head given audio samples."""

import time
import queue
import base64
import logging
import threading
from typing import Optional

import numpy as np

from reachy_mini_conversation_demo.audio.speech_tapper import HOP_MS, SwayRollRT


SAMPLE_RATE = 24000
MOVEMENT_LATENCY_S = 0.08  # seconds between audio and robot movement
logger = logging.getLogger(__name__)


class HeadWobbler:
    """Converts audio deltas (base64) into head movement offsets."""

    def __init__(self, set_offsets):
        """Initialize the head wobbler."""
        self._apply_offsets = set_offsets
        self._base_ts: Optional[float] = None
        self._hops_done: int = 0

        self.audio_queue: queue.Queue = queue.Queue()
        self.sway = SwayRollRT()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def feed(self, delta_b64: str) -> None:
        """Thread-safe: push audio into the consumer queue."""
        buf = np.frombuffer(base64.b64decode(delta_b64), dtype=np.int16).reshape(1, -1)
        self.audio_queue.put((SAMPLE_RATE, buf))

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
            try:
                sr, chunk = self.audio_queue.get_nowait()  # (1,N) int16
            except queue.Empty:
                # avoid while to never exit
                time.sleep(MOVEMENT_LATENCY_S)
                continue

            pcm = np.asarray(chunk).squeeze(0)
            results = self.sway.feed(pcm, sr)
            self.audio_queue.task_done()

            if self._base_ts is None:
                self._base_ts = time.time()

            i = 0
            while i < len(results):
                if self._base_ts is None:
                    self._base_ts = time.time()
                    continue

                target = self._base_ts + MOVEMENT_LATENCY_S + self._hops_done * hop_dt
                now = time.time()

                if now - target >= hop_dt:
                    lag_hops = int((now - target) / hop_dt)
                    drop = min(lag_hops, len(results) - i - 1)
                    if drop > 0:
                        self._hops_done += drop
                        i += drop
                        continue

                if target > now:
                    time.sleep(target - now)

                r = results[i]
                offsets = (
                    r["x_mm"] / 1000.0,
                    r["y_mm"] / 1000.0,
                    r["z_mm"] / 1000.0,
                    r["roll_rad"],
                    r["pitch_rad"],
                    r["yaw_rad"],
                )

                self._apply_offsets(offsets)

                self._hops_done += 1
                i += 1
        logger.debug("Head wobbler thread exited")

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
        # self.drain_audio_queue()
        self.audio_queue = queue.Queue()
        self._base_ts = None
        self._hops_done = 0
        self.sway.reset()
