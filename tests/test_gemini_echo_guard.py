"""Handler-level tests for the Gemini Live self-echo guard (#527).

The robot hears its own riffs through the chassis mic; Gemini transcribes the
echo as user input, the handler opens a real turn, and — before this guard —
reset the idle-riff backoff (#519) every time, undoing the riff pacing. The
guard compares input transcriptions against the robot's recent speech and
skips the pacing reset, closing the turn as ``outcome=self_echo``.
"""

from __future__ import annotations
import asyncio
from types import SimpleNamespace
from typing import Any, Callable, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

import robot_comic.gemini_live as gemini_mod
from robot_comic.gemini_live import GeminiLiveHandler
from robot_comic.tools.core_tools import ToolDependencies


RIFF = (
    "This hotel, I tell ya. You know, I feel like I should just call the "
    "front desk and ask for a room with fewer existential crises."
)
ECHO = "you know I feel like I should just call the…"
GENUINE = "What time does the show start tonight?"


def _server_content(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "model_turn": None,
        "turn_complete": None,
        "interrupted": None,
        "grounding_metadata": None,
        "generation_complete": None,
        "input_transcription": None,
        "output_transcription": None,
        "url_context_metadata": None,
        "turn_complete_reason": None,
        "waiting_for_input": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _response(server_content: Any = None, tool_call: Any = None) -> SimpleNamespace:
    return SimpleNamespace(server_content=server_content, tool_call=tool_call)


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


class _FakeConnectContext:
    def __init__(self, session: _FakeSession):
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *_args: object) -> bool:
        return False


class _FakeLiveClient:
    def __init__(self, session: _FakeSession) -> None:
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=lambda **_kwargs: _FakeConnectContext(session)))


async def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


def _riff_then_input_batch(user_text: str) -> list[SimpleNamespace]:
    """One robot riff turn, then one 'user' input turn with ``user_text``."""
    return [
        _response(_server_content(output_transcription=SimpleNamespace(text=RIFF))),
        _response(_server_content(turn_complete=True)),
        _response(_server_content(input_transcription=SimpleNamespace(text=user_text))),
        _response(_server_content(turn_complete=True)),
    ]


async def _run_session(monkeypatch: pytest.MonkeyPatch, user_text: str) -> tuple[GeminiLiveHandler, list[str]]:
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda default=None, backend=None: "Kore")
    monkeypatch.setattr(gemini_mod, "get_active_tool_specs", lambda _: [])

    deps = ToolDependencies(
        reachy_mini=SimpleNamespace(media=SimpleNamespace(audio=None)),
        movement_manager=MagicMock(),
    )
    handler = GeminiLiveHandler(deps)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    handler._idle_backoff = MagicMock()

    outcomes: list[str] = []
    original_close = handler._close_turn_span

    def _spy_close(outcome: str) -> None:
        outcomes.append(outcome)
        original_close(outcome)

    monkeypatch.setattr(handler, "_close_turn_span", _spy_close)

    session = _FakeSession(batches=[_riff_then_input_batch(user_text)], stop_event=handler._stop_event)
    handler.client = _FakeLiveClient(session)

    task = asyncio.create_task(handler._run_live_session())
    await _wait_for(lambda: len(outcomes) >= 2)
    handler._stop_event.set()
    await asyncio.wait_for(task, timeout=5.0)
    return handler, outcomes


@pytest.mark.asyncio
async def test_echoed_riff_does_not_reset_riff_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, outcomes = await _run_session(monkeypatch, ECHO)
    handler._idle_backoff.on_user_activity.assert_not_called()
    assert "self_echo" in outcomes
    assert outcomes[-1] == "self_echo"


@pytest.mark.asyncio
async def test_genuine_user_input_still_resets_riff_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, outcomes = await _run_session(monkeypatch, GENUINE)
    handler._idle_backoff.on_user_activity.assert_called()
    assert "self_echo" not in outcomes
    assert outcomes[-1] == "success"


@pytest.mark.asyncio
async def test_guard_disabled_restores_old_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini_mod.config, "GEMINI_LIVE_ECHO_GUARD_ENABLED", False)
    handler, outcomes = await _run_session(monkeypatch, ECHO)
    handler._idle_backoff.on_user_activity.assert_called()
    assert "self_echo" not in outcomes
