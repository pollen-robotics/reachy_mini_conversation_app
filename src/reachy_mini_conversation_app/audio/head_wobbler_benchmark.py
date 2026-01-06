"""Benchmark helpers for the head wobbler."""

from __future__ import annotations
import time
import threading
from typing import Iterator, ContextManager
from contextlib import contextmanager
from dataclasses import dataclass


DISPLAY_NAMES: dict[str, str] = {
    "chunk.total": "chunk.total",
    "chunk.main_calc.sway_feed": "(main calc) sway.feed",
    "chunk.communicate.apply_offsets": "(communicate) apply.offsets",
    "chunk.slack.sleep": "(slack) sleep",
    "chunk.logic.rest": "(rest) logic",
    "queue.wait": "queue.wait",
    "receive.total": "receive.audio",
    "receive.decode": "decode",
    "receive.numpy_view": "numpy_view",
}

PARENT_SECTIONS: dict[str, str | None] = {
    "chunk.total": None,
    "chunk.main_calc.sway_feed": "chunk.total",
    "chunk.communicate.apply_offsets": "chunk.total",
    "chunk.slack.sleep": "chunk.total",
    "chunk.logic.rest": "chunk.total",
    "queue.wait": None,
    "receive.total": None,
    "receive.decode": "receive.total",
    "receive.numpy_view": "receive.total",
}


@dataclass
class _SectionStat:
    """Aggregated statistics for a benchmark section."""

    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, duration: float) -> None:
        self.count += 1
        delta = duration - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (duration - self.mean)
        self.total += duration
        self.minimum = min(self.minimum, duration)
        self.maximum = max(self.maximum, duration)

    def variance(self) -> float:
        if self.count == 0:
            return 0.0
        return self.m2 / self.count

    def snapshot(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "count": 0.0,
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


class _TimingCollector:
    """Context-manager driven timing collector."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._stats: dict[str, _SectionStat] = {}
        self._lock = threading.Lock()

    def _record(self, name: str, duration: float) -> None:
        with self._lock:
            stat = self._stats.setdefault(name, _SectionStat())
            stat.update(duration)

    def section(self, name: str) -> ContextManager[None]:
        """Return a context manager that records elapsed time for `name`."""

        @contextmanager
        def _section() -> Iterator[None]:
            if not self.enabled:
                yield
                return

            start = time.perf_counter()
            try:
                yield
            finally:
                self._record(name, time.perf_counter() - start)

        return _section()

    def add_duration(self, name: str, duration: float) -> None:
        if not self.enabled:
            return
        if duration < 0:
            duration = 0.0
        self._record(name, duration)

    def snapshot(self) -> dict[str, dict[str, float]]:
        if not self.enabled:
            return {}
        with self._lock:
            return {name: stat.snapshot() for name, stat in self._stats.items()}

    def format_report(self) -> str:
        snap = self.snapshot()
        if not snap:
            return "Head wobble benchmark enabled but no samples recorded"

        lines = [
            "Section                     Count    Avg (ms)   Var (ms^2)  Total (ms)  Parent%  Min/Max (ms)",
        ]
        for name, stats in sorted(snap.items(), key=lambda item: item[1]["total_s"], reverse=True):
            parent = PARENT_SECTIONS.get(name)
            parent_total = snap[parent]["total_s"] if parent and parent in snap else stats["total_s"]
            pct = (stats["total_s"] / parent_total * 100.0) if parent_total else 0.0
            indent = "  " if parent else ""
            display = DISPLAY_NAMES.get(name, name)
            avg_ms = stats["avg_s"] * 1000.0
            var_ms = stats["var_s2"] * 1_000_000.0
            total_ms = stats["total_s"] * 1000.0
            min_ms = stats["min_s"] * 1000.0
            max_ms = stats["max_s"] * 1000.0
            lines.append(
                f"{indent}{display:<25} {int(stats['count']):>6}  "
                f"{avg_ms:>10.3f}  {var_ms:>11.3f}  {total_ms:>10.3f}  "
                f"{pct:>6.2f}%  {min_ms:>5.2f}/{max_ms:>5.2f}",
            )
        return "\n".join(lines)


class _ChunkCounters:
    """Track chunk and offset counts for summary."""

    def __init__(self) -> None:
        self._chunks = 0
        self._offsets_sent = 0
        self._offsets_total = 0
        self._drops = 0
        self._lock = threading.Lock()

    def record(self, processed_offsets: int, total_offsets: int, drops: int) -> None:
        with self._lock:
            self._chunks += 1
            self._offsets_sent += processed_offsets
            self._offsets_total += total_offsets
            self._drops += drops

    def summary(self) -> tuple[int, int, int, int]:
        with self._lock:
            return self._chunks, self._offsets_sent, self._offsets_total, self._drops


class HeadWobblerDiagnostics:
    """Wrapper exposing benchmark utilities to the head wobbler."""

    def __init__(self, hop_ms: float, enabled: bool) -> None:
        """Store hop size and configure subordinate trackers."""
        self.enabled = enabled
        self._hop_ms = hop_ms
        self._timing = _TimingCollector(enabled)
        self._chunk_counter = 0
        self._chunk_lock = threading.Lock()
        self._chunk_counters = _ChunkCounters()

    def section(self, name: str) -> ContextManager[None]:
        """Get a context manager that times the named section."""
        return self._timing.section(name)

    def add_duration(self, name: str, duration: float) -> None:
        """Accumulate `duration` seconds under the named section."""
        self._timing.add_duration(name, duration)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Return a copy of the current section statistics."""
        return self._timing.snapshot()

    def benchmark_report(self) -> str:
        """Format the full benchmark report plus chunk counters."""
        hop_info = f"HOP_DT={self._hop_ms / 1000.0:.4f}s ({self._hop_ms:.1f} ms per offset)"
        timer_report = self._timing.format_report()
        chunks, offsets_sent, offsets_total, drops = self._chunk_counters.summary()
        footer = f"Chunks={chunks}, offsets_sent/total={offsets_sent}/{offsets_total}, drops={drops}"
        return f"{hop_info}\n{timer_report}\n{footer}"

    def next_chunk_index(self) -> int:
        """Return a monotonically increasing chunk identifier."""
        with self._chunk_lock:
            idx = self._chunk_counter
            self._chunk_counter += 1
            return idx

    def record_chunk(self, processed_offsets: int, total_offsets: int, drops: int) -> None:
        """Update aggregate counts for offsets sent/available/dropped."""
        self._chunk_counters.record(processed_offsets, total_offsets, drops)

    def chunk_summary(
        self,
        chunk_index: int,
        audio_span: float,
        duration: float,
        processed_offsets: int,
        total_offsets: int,
        drops: int,
        processed: bool,
    ) -> str:
        """Format a single chunk summary line."""
        audio_ms = audio_span * 1000.0
        duration_ms = duration * 1000.0
        ratio_pct = (duration / audio_span * 100.0) if audio_span > 0 else 0.0
        status = "OK" if processed and audio_span > 0 else "SKIPPED"
        return (
            f"Chunk#{chunk_index:05d} [{status}] "
            f"audio={audio_ms:7.2f}ms "
            f"wall={duration_ms:7.2f}ms "
            f"ratio={ratio_pct:6.2f}% "
            f"offsets={processed_offsets}/{total_offsets} "
            f"drops={drops}"
        )
