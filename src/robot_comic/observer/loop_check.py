"""Closing-the-loop harness: one trigger -> verify cycle in a single call.

Codifies the "Example autonomous loop" from ``docs/closing-the-loop.md``: play a
prompt at the robot (Tier 1.5 actuator), listen on the laptop mic for a
response (Tier 1 witness), read the robot's Tier-0 events the turn produced, and
return a pass/fail verdict an agent can branch on — all unattended.

To avoid depending on synchronized clocks (the robot's clock can run seconds
ahead of the laptop's), the harness diffs the robot event log by **line count**:
snapshot before firing, read again after, and treat the appended lines as this
turn's events — no timestamp-window guesswork.

Side effects are injectable so the verdict logic stays pure and unit-tested:

* ``trigger``       — fires the prompt (default: the ``play_prompt`` actuator).
* ``listener``      — captures the laptop mic (default: ``sounddevice``, lazy).
* ``event_fetcher`` — reads the robot's event log (default: ``cat`` over SSH).

Topology via env:

* ``LOOP_CHECK_SSH_HOST``         (default ``reachy-mini-ip``)
* ``LOOP_CHECK_REMOTE_EVENT_LOG`` (default ``/home/pollen/.robot_comic/events.jsonl``)
* ``LOOP_CHECK_WINDOW_S``         (default ``12``)

The prompt actuator still enforces its own guards (opt-in via
``ROBOT_PLAY_PROMPT_ENABLED=1``, duration cap, rate limit) — see
``robot_comic.observer.play_prompt``.
"""

from __future__ import annotations
import os
import json
import time
import shlex
from typing import Any, Callable, Optional

from robot_comic.observer.events import summarize_events
from robot_comic.observer.play_prompt import play_prompt
from robot_comic.observer.audio_activity import summarize_activity


DEFAULT_SSH_HOST = os.getenv("LOOP_CHECK_SSH_HOST", "reachy-mini-ip")
DEFAULT_REMOTE_EVENT_LOG = os.getenv("LOOP_CHECK_REMOTE_EVENT_LOG", "/home/pollen/.robot_comic/events.jsonl")
DEFAULT_WINDOW_S = float(os.getenv("LOOP_CHECK_WINDOW_S", "12"))


def _parse_events(lines: list[str]) -> list[dict[str, Any]]:
    """Parse JSONL strings into event dicts, skipping blank/torn lines."""
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _latest_excerpt(events: list[dict[str, Any]]) -> Optional[str]:
    """Most recent ``turn.excerpt`` attribute across ``events`` (or None)."""
    for event in reversed(events):
        excerpt = (event.get("attrs") or {}).get("turn.excerpt")
        if isinstance(excerpt, str) and excerpt:
            return excerpt
    return None


def fetch_remote_event_lines(host: str, path: str, *, timeout_s: float = 15.0) -> list[str]:
    """Return the robot event-log lines via ``ssh <host> cat <path>``.

    Tolerant: returns ``[]`` on connection failure, non-zero exit, or timeout, so
    a transient SSH hiccup degrades to "no events seen" rather than raising.
    """
    import subprocess

    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, f"cat {shlex.quote(path)}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.SubprocessError, OSError):
        return []
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


def listen_for_intervals(window_s: float, *, samplerate: int = 16_000, block_ms: int = 100) -> list[dict[str, Any]]:
    """Capture the laptop mic for ``window_s`` and return sound-present intervals.

    Reuses the Tier-1 witness dBFS + hysteresis logic. Lazy audio imports
    (``sounddevice``/``numpy``); not unit-tested — needs a real mic.
    """
    import sounddevice as sd

    from robot_comic.observer.audio_witness import IntervalDetector, dbfs

    frames = sd.rec(int(samplerate * window_s), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    arr = frames.ravel()
    detector = IntervalDetector()
    block = max(1, int(samplerate * block_ms / 1000))
    base_ms = int(time.time() * 1000) - int(window_s * 1000)
    intervals: list[dict[str, Any]] = []
    for i in range(0, len(arr) - block, block):
        interval = detector.update(base_ms + int(i / samplerate * 1000), dbfs(arr[i : i + block]))
        if interval is not None:
            intervals.append(interval)
    tail = detector.flush()
    if tail is not None:
        intervals.append(tail)
    return intervals


def evaluate_loop(
    new_events: list[dict[str, Any]],
    audio_intervals: list[dict[str, Any]],
    *,
    expect_tool: Optional[str] = None,
) -> dict[str, Any]:
    """Turn a turn's new events + heard audio into a pass/fail verdict (pure).

    Passes when the robot's Tier-0 log shows it spoke, the laptop mic
    independently heard a response, and (if given) ``expect_tool`` fired.
    """
    summary = summarize_events(new_events)
    activity = summarize_activity(audio_intervals)
    failures: list[str] = []
    if not summary["spoke"]:
        failures.append("no robot TTS span in the new events")
    if not activity["sound_present"]:
        failures.append("laptop mic heard no response")
    if expect_tool and expect_tool not in summary["tools_fired"]:
        failures.append(f"expected tool {expect_tool!r} did not fire")
    return {
        "passed": not failures,
        "robot_spoke": summary["spoke"],
        "heard_response": activity["sound_present"],
        "excerpt": _latest_excerpt(new_events),
        "tools_fired": summary["tools_fired"],
        "new_event_count": len(new_events),
        "audio": activity,
        "failures": failures,
    }


def run_loop_check(
    *,
    text: Optional[str] = None,
    wav: Optional[str] = None,
    window_s: float = DEFAULT_WINDOW_S,
    expect_tool: Optional[str] = None,
    ssh_host: Optional[str] = None,
    remote_event_log: Optional[str] = None,
    trigger: Optional[Callable[[], dict[str, Any]]] = None,
    listener: Optional[Callable[[], list[dict[str, Any]]]] = None,
    event_fetcher: Optional[Callable[[], list[str]]] = None,
) -> dict[str, Any]:
    """Run one trigger->verify cycle and return a verdict dict.

    Snapshots the robot event log, fires the prompt, listens for ``window_s``,
    re-reads the log, and evaluates the *appended* events against the heard
    audio. The ``trigger``/``listener``/``event_fetcher`` seams are injectable so
    the orchestration is testable without audio or a robot.
    """
    ssh_host = ssh_host or DEFAULT_SSH_HOST
    remote_event_log = remote_event_log or DEFAULT_REMOTE_EVENT_LOG
    trigger = trigger or (lambda: play_prompt(text=text, wav=wav))
    listener = listener or (lambda: listen_for_intervals(window_s))
    event_fetcher = event_fetcher or (lambda: fetch_remote_event_lines(ssh_host, remote_event_log))

    before = event_fetcher()
    trigger_result = trigger()
    intervals = listener()
    after = event_fetcher()
    # Appended lines = this turn's events (line-count diff dodges clock skew).
    new_lines = after[len(before) :] if len(after) >= len(before) else after
    verdict = evaluate_loop(_parse_events(new_lines), intervals, expect_tool=expect_tool)
    verdict["trigger"] = trigger_result
    return verdict


def main() -> None:
    """CLI: fire one closing-the-loop check and print the verdict as JSON.

    Exit status is 0 on pass, 1 on fail, so it slots into shell/CI gates.
    Requires ``ROBOT_PLAY_PROMPT_ENABLED=1`` (and a piper model for ``--text``).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Closing-the-loop trigger->verify harness")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="prompt text to synthesize + play at the robot")
    source.add_argument("--wav", help="WAV file to play at the robot")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S, help="seconds to listen for a response")
    parser.add_argument("--expect-tool", default=None, help="fail unless this tool fired (e.g. play_emotion)")
    parser.add_argument("--ssh-host", default=None, help=f"robot SSH host (default {DEFAULT_SSH_HOST})")
    args = parser.parse_args()

    verdict = run_loop_check(
        text=args.text,
        wav=args.wav,
        window_s=args.window,
        expect_tool=args.expect_tool,
        ssh_host=args.ssh_host,
    )
    print(json.dumps(verdict, indent=2))
    raise SystemExit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()
