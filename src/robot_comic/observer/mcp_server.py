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

    @server.tool()  # type: ignore[misc]
    def robot_play_prompt(wav: str | None = None, text: str | None = None) -> dict[str, Any]:
        """Play a trigger prompt at the robot from the laptop speaker (Tier 1.5).

        Provide exactly one of ``wav`` (a WAV file path) or ``text`` (synthesized
        with piper). This is the autonomous trigger — it makes the robot's mic
        hear audio so hear-and-respond behaviours can be exercised without a
        human. Guarded: opt-in via ``ROBOT_PLAY_PROMPT_ENABLED=1``, duration-
        capped, and rate-limited (cooldown + per-session cap). On refusal it
        returns ``{"played": false, "error": ...}`` so the agent can react rather
        than crash. See ``docs/closing-the-loop.md``.
        """
        # Lazy import: keeps the optional audio/TTS deps off the server's import
        # path (matching how the witness treats sounddevice).
        from robot_comic.observer.play_prompt import PlayPromptError, play_prompt

        try:
            return play_prompt(wav=wav, text=text)
        except PlayPromptError as exc:
            return {"played": False, "error": str(exc)}

    @server.tool()  # type: ignore[misc]
    def robot_run_loop_check(
        text: str | None = None,
        wav: str | None = None,
        window_s: float = 12.0,
        expect_tool: str | None = None,
    ) -> dict[str, Any]:
        """Run one closing-the-loop trigger->verify cycle and return a verdict.

        Plays a prompt at the robot (exactly one of ``text``/``wav``), listens
        ``window_s`` on the laptop mic, reads the robot's new Tier-0 events, and
        returns ``{passed, robot_spoke, heard_response, excerpt, tools_fired,
        ...}``. Pass ``expect_tool`` (e.g. ``play_emotion``) to require a tool
        fired. Honors the play_prompt guards (opt-in/rate-limit/duration). On a
        guard refusal it returns ``{"passed": false, "error": ...}``.
        """
        from robot_comic.observer.loop_check import run_loop_check
        from robot_comic.observer.play_prompt import PlayPromptError

        try:
            return run_loop_check(text=text, wav=wav, window_s=window_s, expect_tool=expect_tool)
        except PlayPromptError as exc:
            return {"passed": False, "error": str(exc)}

    @server.tool()  # type: ignore[misc]
    def robot_capture_frame(device: int | None = None) -> dict[str, Any]:
        """Tier-2: grab one still from the laptop camera aimed at the robot.

        Saves a JPEG and returns ``{captured, path, width, height}`` (or
        ``{captured: false, error}`` if no frame could be read). On-demand only.
        ``device`` defaults to ``VISUAL_WITNESS_CAMERA_DEVICE``.
        """
        from robot_comic.observer.visual_witness import VisualWitnessError, capture_frame

        try:
            return capture_frame(device=device) if device is not None else capture_frame()
        except VisualWitnessError as exc:
            return {"captured": False, "error": str(exc)}

    @server.tool()  # type: ignore[misc]
    def robot_describe_scene(prompt: str, device: int | None = None, image_path: str | None = None) -> dict[str, Any]:
        """Tier-2: vision-model answer about the robot scene (e.g. "did it move?").

        Grabs a fresh frame from the laptop camera (or describes ``image_path``)
        and runs the local SmolVLM2 model. Returns ``{prompt, description,
        frame_path}``. Expensive/laggy — use only when a motion needs confirming.
        ``device`` defaults to ``VISUAL_WITNESS_CAMERA_DEVICE``.
        """
        from robot_comic.observer.visual_witness import VisualWitnessError, describe_scene

        try:
            if device is not None:
                return describe_scene(prompt, device=device, image_path=image_path)
            return describe_scene(prompt, image_path=image_path)
        except VisualWitnessError as exc:
            return {"error": str(exc)}

    @server.tool()  # type: ignore[misc]
    def robot_measure_motion(seconds: float = 30.0, device: int | None = None) -> dict[str, Any]:
        """Tier-2: record the witness camera and quantify movement smoothness.

        Reduces the burst to a localized motion-energy series, detects jerk
        spikes, and attributes each spike to the tool.execute span rendering
        at that moment (via ``ROBOT_EVENT_LOG``). Returns ``{mean_energy,
        max_energy, max_jerk, spikes, attributions, fps, samples}``. Blocks
        for the full ``seconds`` — keep windows short.
        ``device`` defaults to ``VISUAL_WITNESS_CAMERA_DEVICE``.
        """
        from robot_comic.observer.motion_witness import (
            load_events,
            record_motion,
            analyze_motion,
            correlate_spikes_with_events,
        )

        try:
            start_wall_ts, fps, energies = record_motion(seconds, device=device)
        except Exception as exc:  # camera missing/busy must not kill the server
            return {"error": str(exc)}
        report = analyze_motion(energies, fps, start_wall_ts=start_wall_ts)
        report["fps"] = fps
        events_path = os.getenv("ROBOT_EVENT_LOG", "")
        if events_path and report["spikes"]:
            report["attributions"] = correlate_spikes_with_events(report["spikes"], load_events(events_path))
        return report

    @server.tool()  # type: ignore[misc]
    def robot_watch_bonks(seconds: float = 60.0) -> dict[str, Any]:
        """Tier-1: listen for cowling-impact signatures and grow the caution list.

        Detects short broadband transients (peak jumps over the rolling speech
        background with high-frequency content), attributes each to the
        move/emotion rendering at that moment (via ``ROBOT_EVENT_LOG``), and
        appends entries to the persistent bonk caution list (#522). Blocks for
        the full ``seconds``.
        """
        from robot_comic.observer.bonk_detector import watch

        try:
            return watch(seconds, events_path=os.getenv("ROBOT_EVENT_LOG", "") or None)
        except Exception as exc:  # no mic / sounddevice missing
            return {"error": str(exc)}

    return server


def main() -> None:
    """Console entry point — runs the server over stdio."""
    # Pre-import numpy/scipy extension modules before the anyio event loop
    # starts. On Windows, loading them lazily inside a tool call deadlocks in
    # the extension-module DLL load (numpy 2.2.5 / scipy / mcp 1.27.2 /
    # CPython 3.12) once the stdio server's stdin-reader thread is running —
    # the call never returns and the agent sees a hung tool. MSVC-built stacks
    # (cv2, torch) lazy-load fine; the gfortran/OpenBLAS-runtime ones do not.
    # All optional deps: the event-log tools work without any of this.
    try:
        import numpy  # noqa: F401
        import scipy.linalg  # noqa: F401
    except ImportError:
        pass
    # The describe_scene chain (transformers -> scipy submodules) hits the same
    # deadlock, but pulling torch+transformers in costs ~15s and real RAM on
    # every server start, so it is opt-in: set VISUAL_WITNESS_PRELOAD=1 where
    # robot_describe_scene will actually be used (the model weights still load
    # lazily on the first describe call).
    if os.getenv("VISUAL_WITNESS_PRELOAD", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            import scipy.signal  # noqa: F401
            import scipy.sparse  # noqa: F401
            import scipy.ndimage  # noqa: F401
            import scipy.special  # noqa: F401
            import scipy.interpolate  # noqa: F401

            import robot_comic.vision.local_vision  # noqa: F401
        except ImportError:
            pass
    build_server().run()


if __name__ == "__main__":
    main()
