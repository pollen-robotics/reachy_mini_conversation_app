"""End-to-end integration smoke test — GeminiLiveHandler lifecycle.

Boots the handler in sim mode, feeds a scripted Gemini Live session (audio
model_turn + turn_complete), drains the output queue, and asserts that at
least one PCM audio frame was produced.

Network boundary mocked:
  * ``client.aio.live.connect`` → replaced with a :class:`_FakeConnectContext`
    that yields a :class:`_FakeSession` delivering canned server responses.

Run with::

    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations
import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import robot_comic.gemini_live as gemini_mod
from .conftest import drain_queue, make_tool_deps
from robot_comic.gemini_live import GeminiLiveHandler


# ---------------------------------------------------------------------------
# Fake Gemini Live session (mirrors the pattern in tests/test_gemini_live.py)
# ---------------------------------------------------------------------------


def _server_content(**kwargs: Any) -> SimpleNamespace:
    """Build a server_content SimpleNamespace with sensible None defaults."""
    defaults: dict[str, Any] = {
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
    """Minimal stand-in for a google.genai Live session."""

    def __init__(self, batches: list[list[SimpleNamespace]], stop_event: asyncio.Event) -> None:
        self._batches = list(batches)
        self._stop_event = stop_event
        self.realtime_inputs: list[dict[str, Any]] = []

    async def close(self) -> None:
        self._stop_event.set()

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime_inputs.append(kwargs)

    async def receive(self) -> AsyncIterator[SimpleNamespace]:
        if self._batches:
            for response in self._batches.pop(0):
                yield response
            return
        await self._stop_event.wait()
        return
        yield  # make this an async generator


class _FakeConnectContext:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *_args: object) -> bool:
        return False


class _FakeLiveClient:
    def __init__(self, session: _FakeSession) -> None:
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=lambda **_kwargs: _FakeConnectContext(session)))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


async def _wait_for(predicate: Any, timeout: float = 2.0) -> None:
    """Poll *predicate* until it returns True or *timeout* seconds elapse."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_live_audio_frame_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler lifecycle: feed canned audio turn → assert PCM frame in queue.

    This test exercises:
      1. GeminiLiveHandler construction in sim mode.
      2. ``_run_live_session`` wiring — the session boundary is replaced by a
         fake that delivers one model audio turn followed by turn_complete.
      3. Audio bytes flowing from server_content → output_queue as PCM tuples.
    """
    # Patch module-level callables that would reach external services.
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "Be funny.")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda default=None, backend=None: "Kore")
    monkeypatch.setattr(gemini_mod, "get_active_tool_specs", lambda _: [])

    deps = make_tool_deps()
    handler = GeminiLiveHandler(deps, sim_mode=True)

    # Disable tool-manager side-effects in this smoke test.
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    # Build canned audio — 4800 samples (200 ms) of silence at 24 kHz int16.
    audio_bytes = np.zeros(4800, dtype=np.int16).tobytes()

    session = _FakeSession(
        batches=[
            [
                # Audio model turn
                _response(
                    _server_content(
                        model_turn=SimpleNamespace(
                            parts=[SimpleNamespace(inline_data=SimpleNamespace(data=audio_bytes))]
                        )
                    )
                ),
                # Turn complete to trigger _handle_turn_complete
                _response(_server_content(turn_complete=True)),
            ]
        ],
        stop_event=handler._stop_event,
    )

    handler.client = _FakeLiveClient(session)  # type: ignore[attr-defined]

    # Run the session loop in a background task; wait until at least one audio
    # frame lands in the queue, then signal stop.
    task = asyncio.create_task(handler._run_live_session())

    await _wait_for(lambda: handler.output_queue.qsize() >= 1)

    handler._stop_event.set()
    await asyncio.wait_for(task, timeout=2.0)

    # Drain the queue and assert.
    all_items = drain_queue(handler.output_queue)
    audio_frames = [item for item in all_items if isinstance(item, tuple)]

    assert len(audio_frames) >= 1, (
        f"Expected at least one PCM audio frame in output_queue, got {len(audio_frames)}. "
        f"All item types: {[type(i).__name__ for i in all_items]}"
    )

    sample_rate, pcm_array = audio_frames[0]
    assert sample_rate == 24000, f"Expected Gemini output sample rate 24000, got {sample_rate}"
    assert isinstance(pcm_array, np.ndarray), "PCM data must be a numpy array"
    assert pcm_array.dtype == np.int16, f"PCM dtype should be int16, got {pcm_array.dtype}"
