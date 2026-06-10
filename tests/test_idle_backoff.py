"""Tests for the exponential idle-signal backoff (unprompted riff pacing).

Issue: unprompted riffs in Gemini Live mode fire on a fixed 15 s idle cadence,
making the robot riff back-to-back with no growing interjection window. The
IdleSignalBackoff grows the idle threshold exponentially after each idle
signal and resets it when the user actually speaks.
"""

from __future__ import annotations
import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from robot_comic.idle_backoff import IdleSignalBackoff


# ---------------------------------------------------------------------------
# IdleSignalBackoff unit tests
# ---------------------------------------------------------------------------


class TestIdleSignalBackoff:
    def test_initial_threshold_is_first_s(self) -> None:
        backoff = IdleSignalBackoff(first_s=15.0, factor=2.0, max_s=600.0)
        assert backoff.threshold_s == 15.0

    def test_threshold_grows_by_factor_after_each_signal(self) -> None:
        backoff = IdleSignalBackoff(first_s=15.0, factor=2.0, max_s=600.0)
        backoff.on_signal_sent()
        assert backoff.threshold_s == 30.0
        backoff.on_signal_sent()
        assert backoff.threshold_s == 60.0
        backoff.on_signal_sent()
        assert backoff.threshold_s == 120.0

    def test_threshold_is_capped_at_max_s(self) -> None:
        backoff = IdleSignalBackoff(first_s=100.0, factor=10.0, max_s=300.0)
        backoff.on_signal_sent()
        assert backoff.threshold_s == 300.0
        backoff.on_signal_sent()
        assert backoff.threshold_s == 300.0

    def test_user_activity_resets_threshold(self) -> None:
        backoff = IdleSignalBackoff(first_s=15.0, factor=2.0, max_s=600.0)
        backoff.on_signal_sent()
        backoff.on_signal_sent()
        assert backoff.threshold_s == 60.0
        backoff.on_user_activity()
        assert backoff.threshold_s == 15.0

    def test_factor_one_keeps_threshold_constant(self) -> None:
        """factor=1.0 is the escape hatch back to the old fixed cadence."""
        backoff = IdleSignalBackoff(first_s=15.0, factor=1.0, max_s=600.0)
        backoff.on_signal_sent()
        backoff.on_signal_sent()
        assert backoff.threshold_s == 15.0


# ---------------------------------------------------------------------------
# Config knob tests
# ---------------------------------------------------------------------------


class TestIdleBackoffConfig:
    def test_defaults(self) -> None:
        from robot_comic.config import config

        assert config.GEMINI_LIVE_IDLE_FIRST_S == 15.0
        assert config.GEMINI_LIVE_IDLE_BACKOFF_FACTOR == 2.0
        assert config.GEMINI_LIVE_IDLE_MAX_S == 600.0

    def test_refresh_picks_up_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from robot_comic import config as config_mod

        monkeypatch.setenv("GEMINI_LIVE_IDLE_FIRST_S", "20")
        monkeypatch.setenv("GEMINI_LIVE_IDLE_BACKOFF_FACTOR", "1.5")
        monkeypatch.setenv("GEMINI_LIVE_IDLE_MAX_S", "240")
        config_mod.refresh_runtime_config_from_env()
        try:
            assert config_mod.config.GEMINI_LIVE_IDLE_FIRST_S == 20.0
            assert config_mod.config.GEMINI_LIVE_IDLE_BACKOFF_FACTOR == 1.5
            assert config_mod.config.GEMINI_LIVE_IDLE_MAX_S == 240.0
        finally:
            monkeypatch.delenv("GEMINI_LIVE_IDLE_FIRST_S")
            monkeypatch.delenv("GEMINI_LIVE_IDLE_BACKOFF_FACTOR")
            monkeypatch.delenv("GEMINI_LIVE_IDLE_MAX_S")
            config_mod.refresh_runtime_config_from_env()


# ---------------------------------------------------------------------------
# GeminiLiveHandler integration
# ---------------------------------------------------------------------------


class _RecordingSession:
    """Minimal session stub that records realtime inputs."""

    def __init__(self) -> None:
        self.realtime_inputs: list[dict[str, Any]] = []

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime_inputs.append(kwargs)


def _make_handler() -> Any:
    from robot_comic.gemini_live import GeminiLiveHandler
    from robot_comic.tools.core_tools import ToolDependencies

    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True
    deps = ToolDependencies(
        reachy_mini=SimpleNamespace(media=SimpleNamespace(audio=None)),
        movement_manager=movement_manager,
    )
    return GeminiLiveHandler(deps)


def _idle_signals(session: _RecordingSession) -> list[dict[str, Any]]:
    return [k for k in session.realtime_inputs if "text" in k]


@pytest.mark.asyncio
async def test_emit_idle_threshold_grows_after_each_signal() -> None:
    """Each idle signal raises the bar for the next one (15 → 30 → 60 …)."""
    handler = _make_handler()
    session = _RecordingSession()
    handler.session = session
    loop = asyncio.get_running_loop()

    # First gap: 16 s of idle exceeds the 15 s default — signal fires.
    handler.output_queue.put_nowait((24000, None))
    handler.last_activity_time = loop.time() - 16.0
    await handler.emit()
    assert len(_idle_signals(session)) == 1

    # Same 16 s gap again: below the doubled (30 s) threshold — no signal.
    handler.output_queue.put_nowait((24000, None))
    handler.last_activity_time = loop.time() - 16.0
    await handler.emit()
    assert len(_idle_signals(session)) == 1

    # 31 s gap exceeds the doubled threshold — second signal fires.
    handler.output_queue.put_nowait((24000, None))
    handler.last_activity_time = loop.time() - 31.0
    await handler.emit()
    assert len(_idle_signals(session)) == 2

    # And now the bar is 60 s: a 31 s gap no longer fires.
    handler.output_queue.put_nowait((24000, None))
    handler.last_activity_time = loop.time() - 31.0
    await handler.emit()
    assert len(_idle_signals(session)) == 2


@pytest.mark.asyncio
async def test_user_speech_resets_idle_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real user transcript resets the idle threshold back to first_s."""
    import robot_comic.gemini_live as gemini_mod

    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda default=None, backend=None: "Kore")
    monkeypatch.setattr(gemini_mod, "get_active_tool_specs", lambda _: [])

    handler = _make_handler()
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    # Pre-grow the backoff as if two unprompted riffs already happened.
    handler._idle_backoff.on_signal_sent()
    handler._idle_backoff.on_signal_sent()
    assert handler._idle_backoff.threshold_s == 60.0

    class _FakeSession:
        def __init__(self, batches: list[list[SimpleNamespace]], stop_event: asyncio.Event) -> None:
            self._batches = list(batches)
            self._stop_event = stop_event

        async def close(self) -> None:
            self._stop_event.set()

        async def send_realtime_input(self, **kwargs: Any) -> None:
            return None

        async def send_tool_response(self, **kwargs: Any) -> None:
            return None

        async def receive(self) -> AsyncIterator[SimpleNamespace]:
            if self._batches:
                for response in self._batches.pop(0):
                    yield response
                return
            await self._stop_event.wait()
            return
            yield

    content = SimpleNamespace(
        model_turn=None,
        turn_complete=None,
        interrupted=None,
        grounding_metadata=None,
        generation_complete=None,
        input_transcription=SimpleNamespace(text="hold on, my turn"),
        output_transcription=None,
        url_context_metadata=None,
        turn_complete_reason=None,
        waiting_for_input=None,
    )

    class _FakeConnectContext:
        def __init__(self, inner: _FakeSession) -> None:
            self._inner = inner

        async def __aenter__(self) -> _FakeSession:
            return self._inner

        async def __aexit__(self, *_args: object) -> bool:
            return False

    session = _FakeSession(
        batches=[[SimpleNamespace(server_content=content, tool_call=None)]],
        stop_event=handler._stop_event,
    )
    handler.client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=lambda **_kwargs: _FakeConnectContext(session)))
    )

    task = asyncio.create_task(handler._run_live_session())
    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        if handler._idle_backoff.threshold_s == 15.0:
            break
        await asyncio.sleep(0.01)

    handler._stop_event.set()
    await asyncio.wait_for(task, timeout=5.0)
    assert handler._idle_backoff.threshold_s == 15.0
