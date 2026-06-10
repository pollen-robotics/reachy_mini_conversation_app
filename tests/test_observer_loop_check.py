"""Tests for the closing-the-loop harness (loop_check).

Covers the pure verdict logic (``evaluate_loop``), the latest-excerpt helper,
and the ``run_loop_check`` orchestration via injected fakes — the line-count
diff that isolates a turn's events, and the trigger/listen/fetch seams. The real
mic capture and SSH read need hardware/a robot, so they're not exercised here.
"""

from __future__ import annotations

from robot_comic.observer import loop_check as lc
from robot_comic.observer.loop_check import evaluate_loop, run_loop_check


def _tts_event(excerpt: str = "") -> dict:
    return {"ts": 1, "name": "tts.synthesize", "attrs": {"turn.excerpt": excerpt} if excerpt else {}}


def _tool_event(name: str) -> dict:
    return {"ts": 2, "name": "tool.execute", "attrs": {"tool.name": name}}


def _interval(dur_ms: int = 300, peak: float = -30.0) -> dict:
    return {"start_ms": 0, "end_ms": dur_ms, "dur_ms": dur_ms, "peak_dbfs": peak}


# ---------------------------------------------------------------------------
# wait_for_quiet (pure via injected sampler/clock)
# ---------------------------------------------------------------------------
def _fake_clock(step: float = 0.1):
    state = {"t": 0.0}

    def clock() -> float:
        return state["t"]

    def sleeper(_s: float) -> None:
        state["t"] += step

    return clock, sleeper


def test_wait_for_quiet_waits_out_robot_speech() -> None:
    """Trigger must not fire while the robot is still talking (live finding
    2026-06-10: prompts played mid-speech barge in or get swallowed)."""
    levels = iter([-20.0, -20.0, -20.0] + [-65.0] * 50)  # loud, then quiet
    clock, sleeper = _fake_clock()
    result = lc.wait_for_quiet(
        lambda: next(levels), quiet_dbfs=-55.0, hold_s=1.0, timeout_s=15.0, clock=clock, sleeper=sleeper
    )
    assert result["timed_out"] is False
    assert result["waited_s"] >= 1.0  # at least the hold time


def test_wait_for_quiet_times_out_and_says_so() -> None:
    clock, sleeper = _fake_clock()
    result = lc.wait_for_quiet(
        lambda: -10.0, quiet_dbfs=-55.0, hold_s=1.0, timeout_s=3.0, clock=clock, sleeper=sleeper
    )
    assert result["timed_out"] is True
    assert result["waited_s"] >= 3.0


def test_run_loop_check_reports_quiet_wait_in_verdict() -> None:
    quiet_calls = []

    def fake_quiet_sampler() -> float:
        quiet_calls.append(1)
        return -70.0

    verdict = run_loop_check(
        text="hi",
        trigger=lambda: {"played": True},
        listener=lambda: [_interval()],
        event_fetcher=lambda: [],
        quiet_sampler=fake_quiet_sampler,
    )
    assert quiet_calls, "quiet gate must sample the mic before triggering"
    assert verdict["quiet_wait"]["timed_out"] is False


# ---------------------------------------------------------------------------
# evaluate_loop (pure)
# ---------------------------------------------------------------------------
def test_evaluate_pass_spoke_heard_and_tool() -> None:
    events = [_tts_event("Hey Richie, tell me a joke."), _tool_event("play_emotion")]
    verdict = evaluate_loop(events, [_interval()], expect_tool="play_emotion")
    assert verdict["passed"] is True
    assert verdict["robot_spoke"] is True
    assert verdict["heard_response"] is True
    assert verdict["tools_fired"] == ["play_emotion"]
    assert verdict["excerpt"] == "Hey Richie, tell me a joke."
    assert verdict["failures"] == []


def test_evaluate_fail_when_robot_silent() -> None:
    verdict = evaluate_loop([_tool_event("play_emotion")], [_interval()])
    assert verdict["passed"] is False
    assert verdict["robot_spoke"] is False
    assert any("TTS" in f for f in verdict["failures"])


def test_evaluate_fail_when_no_audio_heard() -> None:
    verdict = evaluate_loop([_tts_event("x")], [])
    assert verdict["passed"] is False
    assert verdict["heard_response"] is False
    assert any("mic heard no response" in f for f in verdict["failures"])


def test_evaluate_fail_when_expected_tool_missing() -> None:
    verdict = evaluate_loop([_tts_event("x")], [_interval()], expect_tool="dance")
    assert verdict["passed"] is False
    assert any("dance" in f for f in verdict["failures"])


def test_latest_excerpt_picks_most_recent() -> None:
    events = [_tts_event("first"), _tts_event("second")]
    assert lc._latest_excerpt(events) == "second"
    assert lc._latest_excerpt([_tool_event("play_emotion")]) is None


# ---------------------------------------------------------------------------
# run_loop_check orchestration (injected fakes — no audio, no SSH)
# ---------------------------------------------------------------------------
def test_run_loop_check_isolates_new_events_by_line_diff() -> None:
    # Two pre-existing log lines; the turn appends one tts + one tool line.
    pre = ['{"ts":1,"name":"tool.execute","attrs":{"tool.name":"play_emotion"}}', "{}"]
    post = pre + [
        '{"ts":9,"name":"tts.synthesize","attrs":{"turn.excerpt":"a joke"}}',
        '{"ts":9,"name":"tool.execute","attrs":{"tool.name":"play_emotion"}}',
    ]
    calls = {"n": 0}

    def fake_fetcher() -> list[str]:
        # First call returns pre-state, second returns post-state.
        calls["n"] += 1
        return pre if calls["n"] == 1 else post

    fired = {"hit": False}

    def fake_trigger() -> dict:
        fired["hit"] = True
        return {"played": True, "source": "text"}

    verdict = run_loop_check(
        text="tell me a joke",
        expect_tool="play_emotion",
        trigger=fake_trigger,
        listener=lambda: [_interval()],
        event_fetcher=fake_fetcher,
    )
    assert fired["hit"] is True
    assert verdict["passed"] is True
    # Only the two appended lines count — the pre-existing play_emotion is excluded.
    assert verdict["new_event_count"] == 2
    assert verdict["excerpt"] == "a joke"
    assert verdict["trigger"] == {"played": True, "source": "text"}


def test_run_loop_check_fails_when_no_new_events() -> None:
    same = ['{"ts":1,"name":"welcome.wav.completed","attrs":{}}']
    verdict = run_loop_check(
        text="hi",
        trigger=lambda: {"played": True},
        listener=lambda: [_interval()],
        event_fetcher=lambda: list(same),
    )
    assert verdict["new_event_count"] == 0
    assert verdict["passed"] is False
    assert verdict["robot_spoke"] is False
