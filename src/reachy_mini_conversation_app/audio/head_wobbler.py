"""Moves head given audio samples."""

import time
import queue
import base64
import threading
from typing import Tuple
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.audio.speech_tapper import HOP_MS, SwayRollRT
from reachy_mini_conversation_app.audio.head_wobbler_benchmark import HeadWobblerDiagnostics


SAMPLE_RATE = 24000
MOVEMENT_LATENCY_S = 0.08  # seconds between audio and robot movement


class HeadWobbler:
    """Converts audio deltas (base64) into head movement offsets."""

    def __init__(
        self,
        set_speech_offsets: Callable[[Tuple[float, float, float, float, float, float]], None],
        enable_benchmark: bool | None = None,
    ) -> None:
        """Initialize the head wobbler."""
        self._apply_offsets = set_speech_offsets
        self._base_ts: float | None = None
        self._hops_done: int = 0

        self.audio_queue: "queue.Queue[Tuple[int, int, NDArray[np.int16]]]" = queue.Queue()
        self.sway = SwayRollRT()

        # Synchronization primitives
        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._generation = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        enable_diag = enable_benchmark if enable_benchmark is not None else True
        self._benchmark = HeadWobblerDiagnostics(HOP_MS, enable_diag)
        self._benchmark_log_interval = 5.0
        self._next_benchmark_log = time.monotonic() + self._benchmark_log_interval

    def feed(self, delta_b64: str) -> None:
        """Thread-safe: push audio into the consumer queue."""
        with self._benchmark.section("receive.total"):
            with self._benchmark.section("receive.decode"):
                pcm_bytes = base64.b64decode(delta_b64)
            with self._benchmark.section("receive.numpy_view"):
                buf = np.frombuffer(pcm_bytes, dtype=np.int16).reshape(1, -1)
        with self._state_lock:
            generation = self._generation
        self.audio_queue.put((generation, SAMPLE_RATE, buf))

    def start(self) -> None:
        """Start the head wobbler loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        if self._benchmark.enabled:
            print("Head wobble benchmark enabled", flush=True)

    def stop(self) -> None:
        """Stop the head wobbler loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        print("Head wobbler stopped", flush=True)
        if self._benchmark.enabled:
            print("Head wobble benchmark results:\n%s" % self.benchmark_report(), flush=True)

    def benchmark_report(self) -> str:
        """Return a formatted benchmark report."""
        return self._benchmark.benchmark_report()

    def working_loop(self) -> None:
        """Convert audio deltas into head movement offsets."""
        hop_dt = HOP_MS / 1000.0

        print("Head wobbler thread started", flush=True)
        while not self._stop_event.is_set():
            if self._benchmark.enabled and time.monotonic() >= self._next_benchmark_log:
                self._next_benchmark_log = time.monotonic() + self._benchmark_log_interval
                print("Head wobble benchmark (live):\n%s" % self.benchmark_report(), flush=True)

            queue_ref = self.audio_queue
            try:
                with self._benchmark.section("queue.wait"):
                    chunk_generation, sr, chunk = queue_ref.get(timeout=hop_dt / 2.0)
            except queue.Empty:
                continue

            chunk_index = self._benchmark.next_chunk_index()
            chunk_wall_start = time.perf_counter()
            chunk_audio_span = 0.0
            processed_results = 0
            drop_count = 0
            len_results = 0
            chunk_processed = False
            chunk_measured = 0.0

            # Here we have a chunk to process
            try:
                with self._benchmark.section("chunk.total"):
                    with self._state_lock:
                        current_generation = self._generation

                    if chunk_generation != current_generation:
                        continue

                    if self._base_ts is None:
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = time.monotonic()

                    pcm = np.asarray(chunk).squeeze(0)
                    # Each chunk batches roughly 250 ms of audio (6000 samples @ 24 kHz), so a 10 ms hop yields ~25 offsets.
                    chunk_audio_span = float(pcm.shape[-1]) / float(sr) if sr > 0 else 0.0
                    sway_start = time.perf_counter()
                    with self._sway_lock:
                        results = self.sway.feed(pcm, sr)
                    sway_duration = time.perf_counter() - sway_start
                    chunk_measured += sway_duration
                    self._benchmark.add_duration("chunk.main_calc.sway_feed", sway_duration)
                    len_results = len(results)

                    i = 0
                    chunk_processed = True
                    while i < len_results:
                        with self._state_lock:
                            if self._generation != current_generation:
                                chunk_processed = False
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
                            # We're late for the scheduled hop, so drop older offsets to catch up.
                            lag_hops = int((now - target) / hop_dt)
                            drop = min(lag_hops, len_results - i - 1)
                            if drop > 0:
                                drop_count += drop
                                with self._state_lock:
                                    self._hops_done += drop
                                    hops_done = self._hops_done
                                i += drop
                                continue

                        if target > now:
                            # We're ahead of schedule (pure slack), so wait until the hop should be applied.
                            sleep_duration = target - now
                            sleep_start = time.perf_counter()
                            time.sleep(sleep_duration)
                            sleep_duration = time.perf_counter() - sleep_start
                            chunk_measured += sleep_duration
                            self._benchmark.add_duration("chunk.slack.sleep", sleep_duration)
                            with self._state_lock:
                                if self._generation != current_generation:
                                    chunk_processed = False
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
                                chunk_processed = False
                                break

                        apply_start = time.perf_counter()
                        self._apply_offsets(offsets)
                        apply_duration = time.perf_counter() - apply_start
                        chunk_measured += apply_duration
                        self._benchmark.add_duration("chunk.communicate.apply_offsets", apply_duration)

                        with self._state_lock:
                            self._hops_done += 1
                        processed_results += 1
                        i += 1
            finally:
                queue_ref.task_done()
                chunk_duration = time.perf_counter() - chunk_wall_start
                if chunk_processed:
                    logic_duration = chunk_duration - chunk_measured
                    if logic_duration < 0:
                        logic_duration = 0.0
                    self._benchmark.add_duration("chunk.logic.rest", logic_duration)
                    self._benchmark.record_chunk(processed_results, len_results, drop_count)
                print(
                    self._benchmark.chunk_summary(
                        chunk_index,
                        chunk_audio_span,
                        chunk_duration,
                        processed_results,
                        len_results,
                        drop_count,
                        chunk_processed,
                    ),
                    flush=True,
                )
            # Here we finished processing the chunk
        print("Head wobbler thread exited", flush=True)

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

        # Drain any queued audio chunks from previous generations
        drained_any = False
        while True:
            try:
                _, _, _ = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            else:
                drained_any = True
                self.audio_queue.task_done()

        with self._sway_lock:
            self.sway.reset()

        if drained_any:
            print("Head wobbler queue drained during reset", flush=True)
