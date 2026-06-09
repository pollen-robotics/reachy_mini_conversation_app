"""Tier-0 observer MCP server.

Exposes the robot event log to an autonomous agent (Claude Code / Hermes) as an
MCP tool, so it can verify the robot actually did something after a trigger.

Run it where the agent runs (your coding laptop), pointed at the logs the app
and the audio witness write (``ROBOT_EVENT_LOG`` for Tier 0,
``ROBOT_AUDIO_LOG`` for the Tier-1 witness)::

    ROBOT_EVENT_LOG=/path/to/events.jsonl python -m robot_comic.observer.mcp_server

Requires the ``mcp`` package (``uv pip install mcp``). The import is lazy — the
rest of ``robot_comic.observer`` stays usable without it, matching how the repo
treats other optional deps (e.g. ``moonshine_voice``).

See ``docs/closing-the-loop.md`` for the full design and the planned Tier 2
tools (visual witness, ``robot_play_prompt`` actuator).
"""

from __future__ import annotations
import os
from typing import Any

from robot_comic.observer.events import recent_events, summarize_events
from robot_comic.observer.audio_activity import summarize_activity, recent_audio_activity


def _log_path() -> str:
    """Resolve the event-log path, failing loudly if it isn't configured."""
    path = os.getenv("ROBOT_EVENT_LOG", "").strip()
    if not path:
        raise RuntimeError(
            "ROBOT_EVENT_LOG is not set — point it at the app's event JSONL "
            "(the same path passed to the app via ROBOT_EVENT_LOG)."
        )
    return path


def _audio_log_path() -> str:
    """Resolve the audio-witness log path, failing loudly if unconfigured."""
    path = os.getenv("ROBOT_AUDIO_LOG", "").strip()
    if not path:
        raise RuntimeError(
            "ROBOT_AUDIO_LOG is not set — point it at the audio-witness JSONL "
            "(the same path passed to robot_comic.observer.audio_witness)."
        )
    return path


def build_server() -> Any:
    """Construct the FastMCP server. Imports ``mcp`` lazily (optional dep)."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("robot-comic-observer")

    # FastMCP is Any here (``mcp`` is an optional dep, not type-checked in CI),
    # so its decorator reads as untyped under strict mypy (code ``misc`` on the
    # pinned mypy 1.18.2).
    @server.tool()  # type: ignore[misc]
    def robot_get_recent_events(window_s: float = 30.0) -> dict[str, Any]:
        """Robot events from the last ``window_s`` seconds.

        Returns a ``summary`` (did it speak? which tools fired? latest
        timestamp) plus the raw ``events`` (turn outcomes, TTS, tool calls)
        so the agent can confirm a behaviour ran.
        """
        events = recent_events(_log_path(), window_s=window_s)
        return {
            "window_s": window_s,
            "summary": summarize_events(events),
            "events": events,
        }

    @server.tool()  # type: ignore[misc]
    def robot_get_audio_activity(window_s: float = 30.0) -> dict[str, Any]:
        """Independent audio-witness view of recently-heard sound.

        Reads the Tier-1 witness log and returns a ``summary`` (was sound
        present? total active ms, peak dBFS) plus the raw ``intervals``. Pair
        with ``robot_get_recent_events`` to confirm the robot's speaker actually
        produced the audio the app claims it played.
        """
        intervals = recent_audio_activity(_audio_log_path(), window_s=window_s)
        return {
            "window_s": window_s,
            "summary": summarize_activity(intervals),
            "intervals": intervals,
        }

    return server


def main() -> None:
    """Console entry point — runs the server over stdio."""
    build_server().run()


if __name__ == "__main__":
    main()
