"""Benchmark helpers for the head wobbler."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


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

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self._record(name, time.perf_counter() - start)

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

        total_sum = sum(section["total_s"] for section in snap.values())
        lines = ["Section                     Count    Avg (ms)   Var (ms^2)  Total (ms)   %Total  Min/Max (ms)"]
        for name, stats in sorted(snap.items(), key=lambda item: item[1]["total_s"], reverse=True):
            parts = name.split(".")
            indent = "  " * (len(parts) - 1)
            display = f"{indent}{name}"
            avg_ms = stats["avg_s"] * 1000.0
            var_ms = stats["var_s2"] * 1_000_000.0
            total_ms = stats["total_s"] * 1000.0
            min_ms = stats["min_s"] * 1000.0
            max_ms = stats["max_s"] * 1000.0
            pct = (stats["total_s"] / total_sum * 100.0) if total_sum else 0.0
            lines.append(
                f"{display:<27} {int(stats['count']):>6}  "
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
        self._lock = threading.Lock()

    def record(self, processed_offsets: int, total_offsets: int) -> None:
        with self._lock:
            self._chunks += 1
            self._offsets_sent += processed_offsets
            self._offsets_total += total_offsets

    def summary(self) -> tuple[int, int, int]:
        with self._lock:
            return self._chunks, self._offsets_sent, self._offsets_total


class HeadWobblerDiagnostics:
    """Wrapper exposing benchmark utilities to the head wobbler."""

    def __init__(self, hop_ms: float, enabled: bool) -> None:
        self.enabled = enabled
        self._hop_ms = hop_ms
        self._timing = _TimingCollector(enabled)
        self._chunk_counter = 0
        self._chunk_lock = threading.Lock()
        self._chunk_counters = _ChunkCounters()

    def section(self, name: str) -> Iterator[None]:
        return self._timing.section(name)

    def add_duration(self, name: str, duration: float) -> None:
        self._timing.add_duration(name, duration)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return self._timing.snapshot()

    def benchmark_report(self) -> str:
        hop_info = f"HOP_DT={self._hop_ms / 1000.0:.4f}s ({self._hop_ms:.1f} ms per offset)"
        timer_report = self._timing.format_report()
        chunks, offsets_sent, offsets_total = self._chunk_counters.summary()
        footer = f"Chunks={chunks}, offsets_sent/total={offsets_sent}/{offsets_total}"
        return f"{hop_info}\n{timer_report}\n{footer}"

    def next_chunk_index(self) -> int:
        with self._chunk_lock:
            idx = self._chunk_counter
            self._chunk_counter += 1
            return idx

    def record_chunk(self, processed_offsets: int, total_offsets: int) -> None:
        self._chunk_counters.record(processed_offsets, total_offsets)

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
