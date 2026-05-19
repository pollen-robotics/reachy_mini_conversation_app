"""End-to-end integration smoke test — HuggingFaceRealtimeHandler lifecycle.

Boots the handler in sim mode, feeds a scripted realtime session
(``response.output_audio.delta`` + ``response.done``), drains the output queue,
and asserts that at least one PCM audio frame was produced.

Network boundary mocked:
  * ``client.realtime.connect`` → replaced with a :class:`_FakeConn` that
    delivers canned server events without touching the HF allocator endpoint.

Run with::

    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations
import base64
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import robot_comic.huggingface_realtime as hf_rt_mod
from .conftest import drain_queue, make_tool_deps
from robot_comic.huggingface_realtime import HuggingFaceRealtimeHandler


# ---------------------------------------------------------------------------
# Fake realtime connection (same shape as test_openai_realtime_smoke.py)
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int = 2400) -> bytes:
    """Return *n_samples* of silence as int16 PCM bytes."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _audio_delta_event(pcm_bytes: bytes) -> SimpleNamespace:
    """Build a ``response.output_audio.delta`` event."""
    return SimpleNamespace(
        type="response.output_audio.delta",
        delta=base64.b64encode(pcm_bytes).decode(),
    )


def _response_done_event() -> SimpleNamespace:
    """Build a ``response.done`` event with minimal usage."""
    usage = SimpleNamespace(
        input_token_details=None,
        output_token_details=None,
        input_tokens=0,
        output_tokens=0,
    )
    response = SimpleNamespace(usage=usage)
    return SimpleNamespace(type="response.done", response=response)


class _FakeSession:
    async def update(self, **_kw: Any) -> None:
        pass


class _FakeInputAudioBuffer:
    async def append(self, **_kw: Any) -> None:
        pass


class _FakeConversationItem:
    async def create(self, **_kw: Any) -> None:
        pass


class _FakeConversation:
    item = _FakeConversationItem()


class _FakeResponseResource:
    async def create(self, **_kw: Any) -> None:
        pass

    async def cancel(self, **_kw: Any) -> None:
        pass


class _FakeConn:
    """Minimal stand-in for an OpenAI-compatible realtime connection."""

    def __init__(self, events: list[Any]) -> None:
        self._events = iter(events)
        self.session = _FakeSession()
        self.input_audio_buffer = _FakeInputAudioBuffer()
        self.conversation = _FakeConversation()
        self.response = _FakeResponseResource()

    async def __aenter__(self) -> "_FakeConn":
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def close(self) -> None:
        pass

    def __aiter__(self) -> "_FakeConn":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration


class _FakeRealtime:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def connect(self, **_kw: Any) -> _FakeConn:
        return _FakeConn(self._events)


class _FakeHFClient:
    def __init__(self, events: list[Any]) -> None:
        self.realtime = _FakeRealtime(events)


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
async def test_hf_realtime_audio_frame_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler lifecycle: feed canned audio delta → assert PCM frame in queue.

    This test exercises:
      1. HuggingFaceRealtimeHandler construction in sim mode.
      2. ``_run_realtime_session`` wiring — the realtime.connect boundary is
         replaced by a fake that delivers one audio delta and a response.done.
      3. Audio bytes flowing from the delta event → output_queue as PCM tuples.
    """
    monkeypatch.setattr(hf_rt_mod, "get_session_instructions", lambda: "Be funny.")
    monkeypatch.setattr(hf_rt_mod, "get_session_voice", lambda default=None, backend=None: "default")
    monkeypatch.setattr(hf_rt_mod, "get_active_tool_specs", lambda _: [])

    deps = make_tool_deps()
    handler = HuggingFaceRealtimeHandler(deps, sim_mode=True)

    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    # Build canned audio — 2400 int16 samples (100 ms) of silence at 16 kHz.
    audio_bytes = _pcm_bytes(2400)

    events = [
        _audio_delta_event(audio_bytes),
        _response_done_event(),
    ]

    handler.client = _FakeHFClient(events)

    task = asyncio.create_task(handler._run_realtime_session())

    await _wait_for(lambda: handler.output_queue.qsize() >= 1)
    await asyncio.wait_for(task, timeout=3.0)

    all_items = drain_queue(handler.output_queue)
    audio_frames = [item for item in all_items if isinstance(item, tuple)]

    assert len(audio_frames) >= 1, (
        f"Expected at least one PCM audio frame in output_queue, got {len(audio_frames)}. "
        f"All item types: {[type(i).__name__ for i in all_items]}"
    )

    sample_rate, pcm_array = audio_frames[0]
    # HuggingFace handler uses 16 kHz (SAMPLE_RATE = 16000).
    assert sample_rate == 16000, f"Expected HF output sample rate 16000, got {sample_rate}"
    assert isinstance(pcm_array, np.ndarray), "PCM data must be a numpy array"
    assert pcm_array.dtype == np.int16, f"PCM dtype should be int16, got {pcm_array.dtype}"
