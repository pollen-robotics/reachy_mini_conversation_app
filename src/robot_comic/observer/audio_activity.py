"""Reader for the Tier-1 audio-witness log.

Parses the newline-delimited "sound present" intervals written by
``robot_comic.observer.audio_witness`` and exposes the recent ones filtered by
a time window, plus a compact rollup.

Independently confirms the robot's speaker actually produced sound — as opposed
to the Tier-0 event log, which only reports what the *app* believes happened.

Stdlib only, so it is unit-testable and importable by the MCP server without the
optional ``sounddevice`` / ``numpy`` capture dependencies.
"""

from __future__ import annotations
import json
import time
from typing import Any, Optional


def _now_ms() -> int:
    """Return the current wall-clock time in epoch milliseconds."""
    return int(time.time() * 1000)


def read_intervals(path: str, *, max_lines: int = 5000) -> list[dict[str, Any]]:
    """Return parsed intervals (oldest first) from the witness JSONL.

    Tolerant of a missing file (returns ``[]``) and of torn/partial lines, which
    are skipped. Only the last ``max_lines`` lines are parsed so a long-running
    witness log stays cheap to read.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.readlines()
    except (FileNotFoundError, OSError):
        return []

    intervals: list[dict[str, Any]] = []
    for line in raw[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            intervals.append(obj)
    return intervals


def recent_audio_activity(
    path: str,
    window_s: float = 30.0,
    *,
    now_ms: Optional[int] = None,
    max_lines: int = 5000,
) -> list[dict[str, Any]]:
    """Intervals that ended within the last ``window_s`` seconds.

    ``now_ms`` is injectable for deterministic tests; it defaults to the current
    wall clock. An interval counts as recent if its ``end_ms`` is at or after the
    cutoff (so an interval still overlapping the window is included).
    """
    if now_ms is None:
        now_ms = _now_ms()
    cutoff = now_ms - int(window_s * 1000)
    out: list[dict[str, Any]] = []
    for iv in read_intervals(path, max_lines=max_lines):
        end = iv.get("end_ms")
        if isinstance(end, (int, float)) and not isinstance(end, bool) and end >= cutoff:
            out.append(iv)
    return out


def summarize_activity(intervals: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact rollup for at-a-glance agent assertions about heard sound."""
    durations = [
        iv["dur_ms"]
        for iv in intervals
        if isinstance(iv.get("dur_ms"), (int, float)) and not isinstance(iv.get("dur_ms"), bool)
    ]
    peaks = [
        iv["peak_dbfs"]
        for iv in intervals
        if isinstance(iv.get("peak_dbfs"), (int, float)) and not isinstance(iv.get("peak_dbfs"), bool)
    ]
    ends = [
        iv["end_ms"]
        for iv in intervals
        if isinstance(iv.get("end_ms"), (int, float)) and not isinstance(iv.get("end_ms"), bool)
    ]
    return {
        "sound_present": bool(intervals),
        "interval_count": len(intervals),
        "total_active_ms": sum(durations),
        "peak_dbfs": max(peaks) if peaks else None,
        "latest_end_ms": max(ends) if ends else None,
    }
