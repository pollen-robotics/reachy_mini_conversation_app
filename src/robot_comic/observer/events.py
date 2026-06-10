"""Reader for the Tier-0 robot event log.

Parses the newline-delimited JSON written by
``robot_comic.telemetry.JsonlEventExporter`` and exposes recent events filtered
by a time window, plus a compact rollup an agent can read at a glance.

Deliberately dependency-free (stdlib only) so it is unit-testable without the
optional ``mcp`` package the server side needs.
"""

from __future__ import annotations
import json
import time
from typing import Any, Optional


# Span-name prefixes that indicate the robot produced audible speech. Kept here
# (not in the server) so the heuristic is covered by the reader's unit tests.
_SPEECH_PREFIXES = ("tts.", "welcome.wav", "first_greeting.")


def _now_ms() -> int:
    """Return the current wall-clock time in epoch ms (matches span ``ts``)."""
    return int(time.time() * 1000)


def read_events(path: str, *, max_lines: int = 2000) -> list[dict[str, Any]]:
    """Return parsed events (oldest first) from the JSONL file.

    Tolerant by design: a missing file yields ``[]`` (the app may not have
    written anything yet), and partial/corrupt lines — e.g. a line half-written
    when we read mid-export — are skipped rather than raising. Only the last
    ``max_lines`` lines are parsed so a large rolling log stays cheap to read.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.readlines()
    except (FileNotFoundError, OSError):
        return []

    events: list[dict[str, Any]] = []
    for line in raw[-max_lines:]:
        line = line.strip()
        # The relay sink (robot-comic-logsink) prefixes lines with "RCSPAN "
        # for the monitor TUI; accept those files as event sources too.
        if line.startswith("RCSPAN "):
            line = line[len("RCSPAN ") :]
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue  # skip a torn/partial line
        if isinstance(obj, dict):
            events.append(obj)
    return events


def recent_events(
    path: str,
    window_s: float = 30.0,
    *,
    now_ms: Optional[int] = None,
    max_lines: int = 2000,
) -> list[dict[str, Any]]:
    """Events whose ``ts`` (epoch ms) falls within the last ``window_s`` seconds.

    ``now_ms`` is injectable for deterministic tests; it defaults to the current
    wall clock.
    """
    if now_ms is None:
        now_ms = _now_ms()
    cutoff = now_ms - int(window_s * 1000)
    out: list[dict[str, Any]] = []
    for event in read_events(path, max_lines=max_lines):
        ts = event.get("ts")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts >= cutoff:
            out.append(event)
    return out


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact rollup of a window of events for at-a-glance agent assertions."""
    spoke = any(str(e.get("name", "")).startswith(_SPEECH_PREFIXES) for e in events)
    tools: set[str] = set()
    for e in events:
        name = (e.get("attrs") or {}).get("tool.name")
        if isinstance(name, str) and name:
            tools.add(name)
    timestamps = [
        e["ts"] for e in events if isinstance(e.get("ts"), (int, float)) and not isinstance(e.get("ts"), bool)
    ]
    return {
        "count": len(events),
        "spoke": spoke,
        "tools_fired": sorted(tools),
        "latest_ts": max(timestamps, default=None),
    }
