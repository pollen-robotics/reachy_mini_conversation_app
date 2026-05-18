"""Tests for ``XttsTTSAdapter`` — the LAN xtts-v2 TTSBackend (#438).

The adapter is a *native* TTSBackend (not a legacy-handler wrapper) so the
tests stub ``httpx.AsyncClient`` directly using the same fake-stream
pattern as ``tests/test_llama_elevenlabs_tts.py``.

Coverage:
- ``synthesize`` emits aligned 2400-sample int16 PCM frames @ 24 kHz from
  mixed-size server chunks.
- ``first_audio_marker`` is appended exactly once on the first frame.
- Non-empty ``tags`` are dropped with a DEBUG log (xtts has no native
  delivery cues — same gap as chatterbox today).
- ``get_available_voices`` hits ``GET /speakers``; falls back to
  ``[current_voice]`` when the server is unreachable.
- ``change_voice`` is lenient (no validation, returns confirmation).
- ``shutdown`` closes the httpx client; safe to call twice.
- HTTP error from ``POST /tts_stream`` propagates as a clean exception
  (no half-yielded frames).
"""

from __future__ import annotations
import logging
from typing import Any

import httpx
import numpy as np
import pytest

from robot_comic.backends import AudioFrame, TTSBackend
from robot_comic.adapters.xtts_tts_adapter import XttsTTSAdapter


# ---------------------------------------------------------------------------
# Fake httpx streaming helpers (mirrors tests/test_llama_elevenlabs_tts.py)
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    """Stand-in for ``httpx.Response`` returned by an ``AsyncClient.stream``."""

    def __init__(self, chunks: list[bytes], status_code: int = 200) -> None:
        self._chunks = chunks
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("POST", "http://example/")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=req, response=resp)

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamCM:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeAsyncClient:
    """Minimal httpx-ish stand-in covering ``stream``, ``get``, ``aclose``."""

    def __init__(
        self,
        stream_response: _FakeStreamResponse | None = None,
        get_payload: Any = None,
        get_status: int = 200,
        raise_on_get: Exception | None = None,
    ) -> None:
        self._stream_response = stream_response
        self._get_payload = get_payload
        self._get_status = get_status
        self._raise_on_get = raise_on_get
        self.posts: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self.aclose_calls = 0

    def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamCM:
        self.posts.append({"method": method, "url": url, **kwargs})
        assert self._stream_response is not None, "stream() called but no _stream_response set"
        return _FakeStreamCM(self._stream_response)

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.gets.append(url)
        if self._raise_on_get is not None:
            raise self._raise_on_get
        req = httpx.Request("GET", url)
        # httpx.Response.json() parses JSON bytes; encode the payload.
        import json as _json

        body = _json.dumps(self._get_payload).encode("utf-8")
        return httpx.Response(self._get_status, request=req, content=body)

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _silent_pcm_bytes(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# Construction / Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_tts_backend_protocol() -> None:
    """Structural Protocol check — adapter exposes the full TTSBackend surface."""
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles")
    assert isinstance(adapter, TTSBackend)


def test_current_voice_defaults_to_constructor_arg() -> None:
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles")
    assert adapter.get_current_voice() == "rickles"


# ---------------------------------------------------------------------------
# synthesize() streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_emits_aligned_2400_sample_frames() -> None:
    """Mixed-size server chunks (in bytes) are re-chunked into 2400-sample int16 frames."""
    # 1000 + 3800 = 4800 samples → exactly two 2400-sample frames.
    response = _FakeStreamResponse([_silent_pcm_bytes(1000), _silent_pcm_bytes(3800)])
    client = _FakeAsyncClient(stream_response=response)
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)

    frames: list[AudioFrame] = []
    async for frame in adapter.synthesize("hello"):
        frames.append(frame)

    assert len(frames) == 2
    for f in frames:
        assert isinstance(f.samples, np.ndarray)
        assert f.samples.dtype == np.int16
        assert f.samples.shape == (2400,)
        assert f.sample_rate == 24000


@pytest.mark.asyncio
async def test_synthesize_flushes_partial_remainder_as_final_frame() -> None:
    """When server bytes don't divide evenly, the leftover yields as a final short frame."""
    # 2400 + 500 = 2900 samples → one full 2400 frame + one partial 500-sample frame.
    response = _FakeStreamResponse([_silent_pcm_bytes(2900)])
    client = _FakeAsyncClient(stream_response=response)
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)

    frames = [f async for f in adapter.synthesize("hi")]
    assert len(frames) == 2
    assert frames[0].samples.shape == (2400,)
    assert frames[1].samples.shape == (500,)


@pytest.mark.asyncio
async def test_synthesize_posts_to_tts_stream_with_speaker_and_language() -> None:
    response = _FakeStreamResponse([_silent_pcm_bytes(2400)])
    client = _FakeAsyncClient(stream_response=response)
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", language="en", _client=client)

    async for _ in adapter.synthesize("hello world"):
        pass

    assert len(client.posts) == 1
    call = client.posts[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/tts_stream")
    body = call.get("json")
    assert body == {"text": "hello world", "speaker": "rickles", "language": "en"}


@pytest.mark.asyncio
async def test_synthesize_uses_current_voice_after_change_voice() -> None:
    response = _FakeStreamResponse([_silent_pcm_bytes(2400)])
    client = _FakeAsyncClient(stream_response=response)
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)
    await adapter.change_voice("hicks")

    async for _ in adapter.synthesize("test"):
        pass

    assert client.posts[0]["json"]["speaker"] == "hicks"


# ---------------------------------------------------------------------------
# first_audio_marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_audio_marker_appended_once_on_first_frame() -> None:
    response = _FakeStreamResponse([_silent_pcm_bytes(2400), _silent_pcm_bytes(2400), _silent_pcm_bytes(2400)])
    client = _FakeAsyncClient(stream_response=response)
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)

    marker: list[float] = []
    async for _ in adapter.synthesize("hi", first_audio_marker=marker):
        pass

    assert len(marker) == 1
    assert marker[0] > 0  # monotonic timestamp


@pytest.mark.asyncio
async def test_first_audio_marker_none_does_not_fire() -> None:
    response = _FakeStreamResponse([_silent_pcm_bytes(2400)])
    client = _FakeAsyncClient(stream_response=response)
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)

    # Should simply not raise.
    async for _ in adapter.synthesize("hi", first_audio_marker=None):
        pass


# ---------------------------------------------------------------------------
# Tag handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_empty_tags_are_logged_at_debug_and_dropped(caplog: pytest.LogCaptureFixture) -> None:
    response = _FakeStreamResponse([_silent_pcm_bytes(2400)])
    client = _FakeAsyncClient(stream_response=response)
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)

    with caplog.at_level(logging.DEBUG, logger="robot_comic.adapters.xtts_tts_adapter"):
        async for _ in adapter.synthesize("hi", tags=("fast", "annoyance")):
            pass

    drop_msgs = [r for r in caplog.records if "dropping delivery tags" in r.message]
    assert len(drop_msgs) == 1


# ---------------------------------------------------------------------------
# Voice methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_available_voices_hits_speakers_endpoint() -> None:
    client = _FakeAsyncClient(get_payload=["rickles", "hicks", "default"])
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)

    voices = await adapter.get_available_voices()

    assert voices == ["rickles", "hicks", "default"]
    assert any(url.endswith("/speakers") for url in client.gets)


@pytest.mark.asyncio
async def test_get_available_voices_falls_back_to_current_voice_on_error() -> None:
    client = _FakeAsyncClient(raise_on_get=httpx.ConnectError("nope"))
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)

    voices = await adapter.get_available_voices()

    assert voices == ["rickles"]


@pytest.mark.asyncio
async def test_change_voice_is_lenient_and_idempotent() -> None:
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles")
    msg = await adapter.change_voice("brand_new_voice")
    assert "brand_new_voice" in msg
    assert adapter.get_current_voice() == "brand_new_voice"
    # Setting the same voice again is a no-op success.
    msg2 = await adapter.change_voice("brand_new_voice")
    assert "brand_new_voice" in msg2


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_is_idempotent() -> None:
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles")
    await adapter.prepare()
    await adapter.prepare()
    # Client built once and reused.
    assert adapter._client is not None


@pytest.mark.asyncio
async def test_shutdown_closes_client_and_is_safe_to_call_twice() -> None:
    client = _FakeAsyncClient()
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)
    await adapter.shutdown()
    await adapter.shutdown()
    assert client.aclose_calls == 1  # only the first call hits aclose


@pytest.mark.asyncio
async def test_reset_per_session_state_is_a_no_op() -> None:
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles")
    # Should simply not raise and return None — there's no per-session
    # echo-guard state on a stateless HTTP-streaming adapter.
    assert await adapter.reset_per_session_state() is None


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_error_propagates_cleanly() -> None:
    response = _FakeStreamResponse([], status_code=500)
    client = _FakeAsyncClient(stream_response=response)
    adapter = XttsTTSAdapter(base_url="http://x:1", default_speaker="rickles", _client=client)

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in adapter.synthesize("boom"):
            pass
