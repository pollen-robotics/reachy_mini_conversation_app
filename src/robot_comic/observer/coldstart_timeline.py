"""Cold-start timeline assembly (#541).

Pure parsing/merging logic for the witnessed cold-start capture: journal
lines (robot clock), relayed RCSPAN telemetry (robot clock), and workstation
audio/video witness events are merged into a single t0-relative timeline,
with the inter-milestone gaps ranked so optimization effort goes at the
biggest gap rather than the most visible one.

The capture orchestration (ssh, sound device, camera) lives in
``coldstart_capture``; everything here is side-effect free and unit-tested.
"""

from __future__ import annotations
import re
import json
from typing import Any
from datetime import datetime


TimelineEvent = dict[str, Any]


_SINK_PREFIX = "RCSPAN "

# journald `-o short-iso`: "<iso-ts> <host> <unit>[pid]: <message>"
_JOURNAL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:?\d{2}|Z))\s+\S+\s+(?P<proc>[^:]+):\s(?P<msg>.*)$"
)

_STARTUP_CHECKPOINT_RE = re.compile(r"Startup: \+(?P<offset>\d+(?:\.\d+)?)s (?P<label>.+)$")

# Milestone classification for journal messages that are not Startup
# checkpoints. First match wins.
_JOURNAL_MILESTONES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Started reachy-app-autostart"), "service_started"),
    (re.compile(r"Gemini Live session connected"), "gemini_session_connected"),
    (re.compile(r"Lost connection with the server"), "sdk_connection_lost"),
)


def parse_sink_line(line: str) -> TimelineEvent | None:
    """Parse one log-sink line (``RCSPAN {json}``) into a timeline event."""
    if not line.startswith(_SINK_PREFIX):
        return None
    try:
        payload = json.loads(line[len(_SINK_PREFIX) :])
    except (json.JSONDecodeError, ValueError):
        return None
    name = payload.get("name")
    ts = payload.get("ts")
    if not isinstance(name, str) or not isinstance(ts, (int, float)):
        return None
    return {
        "source": "span",
        "label": name,
        "epoch_ms": int(ts),
        "detail": {
            "dur_ms": payload.get("dur_ms"),
            "attrs": payload.get("attrs") or {},
        },
    }


def parse_journal_line(line: str) -> TimelineEvent | None:
    """Parse one ``journalctl -o short-iso`` line into a timeline event.

    Returns None for lines that are not recognised milestones (systemd
    service start, ``Startup: +X.XXs`` checkpoints, or the patterns in
    ``_JOURNAL_MILESTONES``).
    """
    match = _JOURNAL_RE.match(line)
    if match is None:
        return None
    try:
        # journald's short-iso offset varies by version ("+0100" vs "+01:00");
        # fromisoformat accepts both on py3.11+.
        when = datetime.fromisoformat(match.group("ts").replace("Z", "+00:00"))
    except ValueError:
        return None
    epoch_ms = int(when.timestamp() * 1000)
    message = match.group("msg")

    checkpoint = _STARTUP_CHECKPOINT_RE.search(message)
    if checkpoint is not None:
        return {
            "source": "journal",
            "label": checkpoint.group("label").strip(),
            "epoch_ms": epoch_ms,
            "detail": {"startup_offset_s": float(checkpoint.group("offset"))},
        }

    for pattern, label in _JOURNAL_MILESTONES:
        if pattern.search(message):
            return {
                "source": "journal",
                "label": label,
                "epoch_ms": epoch_ms,
                "detail": {"message": message},
            }
    return None


def build_timeline(events: list[TimelineEvent], t0_epoch_ms: int) -> list[TimelineEvent]:
    """Sort events, drop anything before t0, and add ``t_rel_s``."""
    timeline = [dict(e) for e in events if e["epoch_ms"] >= t0_epoch_ms]
    timeline.sort(key=lambda e: e["epoch_ms"])
    for event in timeline:
        event["t_rel_s"] = round((event["epoch_ms"] - t0_epoch_ms) / 1000.0, 3)
    return timeline


def rank_gaps(timeline: list[TimelineEvent]) -> list[dict[str, Any]]:
    """Consecutive inter-event gaps, largest first."""
    gaps = [
        {
            "gap_s": round(after["t_rel_s"] - before["t_rel_s"], 3),
            "from": before["label"],
            "to": after["label"],
        }
        for before, after in zip(timeline, timeline[1:])
    ]
    gaps.sort(key=lambda g: g["gap_s"], reverse=True)
    return gaps


def render_markdown(
    timeline: list[TimelineEvent],
    gaps: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> str:
    """Render the merged timeline + ranked gaps as a markdown report."""
    lines: list[str] = ["# Cold-start timeline", ""]
    for key, value in (meta or {}).items():
        lines.append(f"- **{key}**: {value}")
    if meta:
        lines.append("")

    lines += ["## Timeline (t=0 at service start)", "", "| t | source | event |", "|---|---|---|"]
    for event in timeline:
        lines.append(f"| +{event['t_rel_s']:.2f}s | {event['source']} | {event['label']} |")

    lines += ["", "## Gaps, largest first", ""]
    for i, gap in enumerate(gaps, start=1):
        lines.append(f"{i}. **{gap['gap_s']:.2f}s** — `{gap['from']}` → `{gap['to']}`")
    return "\n".join(lines) + "\n"
