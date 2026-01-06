"""Moves head given audio samples."""

import time
import queue
import base64
import threading
from dataclasses import dataclass
from typing import Iterator, Tuple
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.audio.speech_tapper import HOP_MS, SwayRollRT


SAMPLE_RATE = 24000
MOVEMENT_LATENCY_S = 0.08  # seconds between audio and robot movement

@dataclass
class _SectionStat:
    """Aggregated statistics for a benchmark section."""

    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences for variance (Welford)
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, duration: float) -> None:
        """Update the aggregates in-place."""
        self.count += 1
        delta = duration - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (duration - self.mean)
        self.total += duration
        self.minimum = min(self.minimum, duration)
        self.maximum = max(self.maximum, duration)

    def variance(self) -> float:
        """Return the population variance for the section."""
        if self.count == 0:
            return 0.0
        return self.m2 / self.count

    def snapshot(self) -> dict[str, float]:
        """Create a serializable snapshot of the metrics."""
        if self.count == 0:
            return {
                "count": 0,
                "avg_s": 0.0,
                "var_s2": 0.0,
                "total_s": 0.0,
                "min_s": 0.0,
                "max_s": 0.0,
            }

        return {
            "count": float(self.count),
            "avg_s": self.mean,
            "var_s2": self.variance(),
            "total_s": self.total,
            "min_s": self.minimum,
            "max_s": self.maximum,
        }


class HeadWobblerBenchmark:
    """Collect timing statistics for the wobble pipeline."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._stats: dict[str, _SectionStat] = {}
        self._stats_lock = threading.Lock()

    def _record(self, name: str, duration: float) -> None:
        with self._stats_lock:
            stat = self._stats.setdefault(name, _SectionStat())
            stat.update(duration)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Context manager to accumulate timing for a named section."""
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._record(name, end - start)

    def add_duration(self, name: str, duration: float) -> None:
        """Directly add a measured duration to the aggregates."""
        if not self.enabled:
            return
        if duration < 0:
            duration = 0.0
        self._record(name, duration)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Return copies of the collected metrics."""
        if not self.enabled:
            return {}
        with self._stats_lock:
            return {name: stat.snapshot() for name, stat in self._stats.items()}

    def format_report(self) -> str:
        """Format a human-readable summary of the metrics."""
        if not self.enabled:
            return "Head wobble benchmark disabled"
        snap = self.snapshot()
        if not snap:
            return "Head wobble benchmark enabled but no samples recorded"

        total_sum = sum(section["total_s"] for section in snap.values())
        lines = ["Section                     Count    Avg (ms)   Var (ms^2)  Total (ms)   %Total  Min/Max (ms)"]
        for name, stats in sorted(snap.items(), key=lambda item: item[1]["total_s"], reverse=True):
            avg_ms = stats["avg_s"] * 1000.0
            var_ms = stats["var_s2"] * 1_000_000.0
            total_ms = stats["total_s"] * 1000.0
            min_ms = stats["min_s"] * 1000.0
            max_ms = stats["max_s"] * 1000.0
            pct = (stats["total_s"] / total_sum * 100.0) if total_sum else 0.0
            indent = "  " * name.count(".")
            section_name = f"{indent}{name}"
            lines.append(
                f"{section_name:<27} {int(stats['count']):>6}  "
                f"{avg_ms:>10.3f}  {var_ms:>11.3f}  {total_ms:>10.3f}  "
                f"{pct:>6.2f}%  {min_ms:>5.2f}/{max_ms:>5.2f}",
            )
        return "\n".join(lines)


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
        self._benchmark = HeadWobblerBenchmark(enabled=enable_benchmark if enable_benchmark is not None else True)
        self._benchmark_log_interval = 5.0
        self._next_benchmark_log = time.monotonic() + self._benchmark_log_interval
        self._realtime_audio_total = 0.0
        self._realtime_compute_total = 0.0
        self._realtime_max_ratio = 0.0
        self._realtime_overruns = 0
        self._realtime_chunks = 0
        self._realtime_slack_total = 0.0
        self._realtime_deficit_total = 0.0
        self._realtime_worst_slack = 0.0
        self._realtime_worst_deficit = 0.0
        self._realtime_lock = threading.Lock()

    def feed(self, delta_b64: str) -> None:
        """Thread-safe: push audio into the consumer queue."""
        with self._benchmark.section("feed.decode"):
            pcm_bytes = base64.b64decode(delta_b64)
        with self._benchmark.section("feed.numpy_view"):
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

    def benchmark_snapshot(self) -> dict[str, dict[str, float]]:
        """Return raw benchmark data."""
        return self._benchmark.snapshot()

    def benchmark_report(self) -> str:
        """Return a formatted benchmark report."""
        base_report = self._benchmark.format_report()
        status_line = self._realtime_status_line()
        if status_line:
            return f"{base_report}\n{status_line}"
        return base_report

    def _realtime_snapshot(self) -> dict[str, float]:
        with self._realtime_lock:
            return {
                "audio_s": self._realtime_audio_total,
                "compute_s": self._realtime_compute_total,
                "max_ratio": self._realtime_max_ratio,
                "chunks": float(self._realtime_chunks),
                "overruns": float(self._realtime_overruns),
                "slack_s": self._realtime_slack_total,
                "deficit_s": self._realtime_deficit_total,
                "worst_slack_s": self._realtime_worst_slack,
                "worst_deficit_s": self._realtime_worst_deficit,
            }

    def _realtime_status_line(self) -> str:
        snap = self._realtime_snapshot()
        chunks = int(snap["chunks"])
        audio = snap["audio_s"]
        compute = snap["compute_s"]
        if chunks == 0 or audio <= 0.0:
            return ""
        avg_ratio = compute / audio
        max_ratio = snap["max_ratio"]
        overruns = int(snap["overruns"])
        status = "PASS" if avg_ratio <= 1.0 and max_ratio <= 1.1 else "WARN"
        detail = (
            f"avg utilization {avg_ratio * 100.0:.1f}%, "
            f"peak {max_ratio * 100.0:.1f}%, chunks={chunks}"
        )
        if overruns:
            detail += f", overruns={overruns}"
        if snap["slack_s"] > 0:
            avg_slack = (snap["slack_s"] / chunks) * 1000.0
            detail += f", avg slack {avg_slack:.2f}ms"
            detail += f", peak slack {snap['worst_slack_s'] * 1000.0:.2f}ms"
        if snap["deficit_s"] > 0:
            avg_deficit = (snap["deficit_s"] / overruns) * 1000.0 if overruns else snap["deficit_s"] * 1000.0
            detail += f", avg deficit {avg_deficit:.2f}ms"
            detail += f", worst deficit {snap['worst_deficit_s'] * 1000.0:.2f}ms"
        return f"Realtime status: {status} ({detail})"

    def working_loop(self) -> None:
        """Convert audio deltas into head movement offsets."""
        hop_dt = HOP_MS / 1000.0

        print("Head wobbler thread started", flush=True)
        while not self._stop_event.is_set():
            if self._benchmark.enabled and time.monotonic() >= self._next_benchmark_log:
                self._next_benchmark_log = time.monotonic() + self._benchmark_log_interval
                print("Head wobble benchmark (live):\n%s" % self.benchmark_report(), flush=True)
            chunk_start: float | None = None
            chunk_audio_span = 0.0
            queue_ref = self.audio_queue
            queue_poll_start = time.perf_counter()
            try:
                chunk_generation, sr, chunk = queue_ref.get(timeout=hop_dt / 2.0)  # the timeout throttles the loop
            except queue.Empty:
                poll_duration = time.perf_counter() - queue_poll_start
                self._benchmark.add_duration("queue.poll", poll_duration)
                continue
            else:
                poll_duration = time.perf_counter() - queue_poll_start
                self._benchmark.add_duration("queue.poll", poll_duration)

            try:
                with self._state_lock:
                    current_generation = self._generation
                if chunk_generation != current_generation:
                    continue

                chunk_start = time.perf_counter()
                if self._base_ts is None:
                    with self._state_lock:
                        if self._base_ts is None:
                            self._base_ts = time.monotonic()

                pcm = np.asarray(chunk).squeeze(0)
                chunk_audio_span = float(pcm.shape[-1]) / float(sr) if sr > 0 else 0.0
                with self._benchmark.section("sway.feed"):
                    with self._sway_lock:
                        results = self.sway.feed(pcm, sr)

                i = 0
                while i < len(results):
                    iteration_start = time.perf_counter()
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
                        sleep_duration = target - now
                        sleep_start = time.perf_counter()
                        time.sleep(sleep_duration)
                        self._benchmark.add_duration("schedule.sleep", time.perf_counter() - sleep_start)
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

                    with self._benchmark.section("apply.offsets"):
                        self._apply_offsets(offsets)

                    with self._state_lock:
                        self._hops_done += 1
                    i += 1
                    self._benchmark.add_duration("result.iteration", time.perf_counter() - iteration_start)
            finally:
                queue_ref.task_done()
                if chunk_start is not None:
                    chunk_duration = time.perf_counter() - chunk_start
                    self._benchmark.add_duration("chunk.total", chunk_duration)
                    if chunk_audio_span > 0:
                        ratio = chunk_duration / chunk_audio_span if chunk_audio_span > 0 else 0.0
                        with self._realtime_lock:
                            self._realtime_audio_total += chunk_audio_span
                            self._realtime_compute_total += chunk_duration
                            self._realtime_chunks += 1
                            self._realtime_max_ratio = max(self._realtime_max_ratio, ratio)
                            if ratio > 1.0:
                                self._realtime_overruns += 1
                            diff = chunk_audio_span - chunk_duration
                            if diff >= 0:
                                self._realtime_slack_total += diff
                                self._realtime_worst_slack = max(self._realtime_worst_slack, diff)
                            else:
                                deficit = -diff
                                self._realtime_deficit_total += deficit
                                self._realtime_worst_deficit = max(self._realtime_worst_deficit, deficit)
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
