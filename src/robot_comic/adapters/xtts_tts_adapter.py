"""XttsTTSAdapter: native ``TTSBackend`` for the LAN xtts-v2 service (#438).

Unlike :class:`ChatterboxTTSAdapter` / :class:`ElevenLabsTTSAdapter` /
:class:`GeminiTTSAdapter` — all of which wrap an existing legacy handler
class — this adapter is a **native** implementation. There is no
"XttsResponseHandler" legacy class because xtts-v2 was added after the
composable pipeline (Phase 5) landed, so the adapter talks to the LAN
service directly over HTTP.

## Protocol contract (decided in #438)

The adapter speaks a thin custom HTTP contract that the service-side
scripts (#436) will implement to match. We chose raw-PCM-over-chunked-HTTP
over the off-the-shelf alternatives (xtts-streaming-server WebSocket,
xtts-api-server WAV-streaming HTTP) because it gives the lowest
time-to-first-audio with the smallest adapter footprint.

- ``POST /tts_stream`` — JSON body ``{"text": str, "speaker": str,
  "language": str}``. Server responds with ``Content-Type:
  application/octet-stream`` and ``Transfer-Encoding: chunked``,
  streaming raw little-endian int16 PCM at 24 kHz mono. No WAV header.
- ``GET /speakers`` — JSON list of speaker keys the server can resolve to
  reference WAVs on its own filesystem. The adapter sends the speaker
  *key*, never the WAV bytes — keeps per-request payload < 1 KB and
  matches how the other TTS backends in this repo name voices.

## Frame chunking

The composable pipeline expects ``AudioFrame.samples`` to be
``np.ndarray[int16]`` at 24 kHz. We re-chunk the server's mixed-size byte
stream into fixed 2400-sample (100 ms) frames so downstream consumers see
the same shape as the elevenlabs path. The trailing remainder — server
sends a count not divisible by 2400 — is yielded as a final short frame
on stream end.

## Known gaps

- **No tag forwarding.** xtts-v2 has no native delivery cues (fast /
  slow / annoyance) the way ElevenLabs voice-settings does. The adapter
  accepts ``tags`` for Protocol compliance and drops them with a DEBUG
  log — mirrors :class:`ChatterboxTTSAdapter`. A future PR may strip
  ``[fast]``-style markers out of ``text`` if it surfaces as a real
  problem in operator audio.
- **No per-session state to reset.** Unlike legacy-handler-wrapping
  adapters which carry echo-guard accumulators (``_speaking_until`` etc.)
  on the wrapped handler, this adapter is stateless between calls —
  :meth:`reset_per_session_state` is a clean no-op.
- **Hardware E2E is gated on the service running.** Issue #436 builds
  the start/stop scripts. Until that lands the adapter is testable only
  via stubbed HTTP.
"""

from __future__ import annotations
import time
import logging
from typing import Any, AsyncIterator

import httpx
import numpy as np

from robot_comic.backends import AudioFrame


logger = logging.getLogger(__name__)


# Fixed frame size emitted to the orchestrator (100 ms @ 24 kHz mono).
# Matches the chunking the elevenlabs path uses so downstream queue +
# echo-guard accounting see consistent frame shapes across TTS backends.
_FRAME_SAMPLES = 2400
_SAMPLE_RATE_HZ = 24000
_BYTES_PER_SAMPLE = 2  # int16 little-endian
_FRAME_BYTES = _FRAME_SAMPLES * _BYTES_PER_SAMPLE


class XttsTTSAdapter:
    """Native ``TTSBackend`` implementation for the LAN xtts-v2 service."""

    def __init__(
        self,
        base_url: str,
        default_speaker: str,
        language: str = "en",
        timeout_s: float = 30.0,
        *,
        _client: Any | None = None,
    ) -> None:
        """Construct an adapter pointed at the LAN xtts service.

        ``_client`` is a test seam: pass a stub satisfying the
        ``stream``/``get``/``aclose`` subset of :class:`httpx.AsyncClient`
        to avoid real HTTP. Production code leaves it unset and lets
        :meth:`prepare` build a real client on demand.
        """
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._timeout_s = timeout_s
        self._current_voice = default_speaker
        self._client: Any | None = _client

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def prepare(self) -> None:
        """Lazily build the httpx client. Idempotent."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)

    async def shutdown(self) -> None:
        """Close the httpx client. Safe to call twice."""
        client = self._client
        if client is None:
            return
        try:
            await client.aclose()
        except Exception as exc:  # pragma: no cover — best-effort cleanup
            logger.warning("XttsTTSAdapter shutdown: aclose() raised: %s", exc)
        self._client = None

    async def reset_per_session_state(self) -> None:
        """No-op — the adapter is stateless between calls.

        See module docstring "Known gaps" for why this differs from the
        legacy-handler-wrapping adapters' implementations.
        """
        return None

    # ------------------------------------------------------------------ #
    # Synthesis                                                          #
    # ------------------------------------------------------------------ #

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        """Stream PCM frames for *text* from the LAN xtts service.

        See the module docstring for the HTTP contract. Re-chunks the
        server's raw int16 byte stream into fixed 2400-sample frames so
        the orchestrator sees the same frame shape as other TTS
        backends. Yields any trailing partial-frame remainder on stream
        end.
        """
        if tags:
            logger.debug(
                "XttsTTSAdapter: dropping delivery tags %r; xtts-v2 has no native "
                "delivery-cue channel (parity with ChatterboxTTSAdapter)",
                tags,
            )

        await self.prepare()
        assert self._client is not None  # prepare() guarantees this

        url = f"{self._base_url}/tts_stream"
        payload = {
            "text": text,
            "speaker": self._current_voice,
            "language": self._language,
        }

        buffer = bytearray()
        marker_appended = False

        async with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                buffer.extend(chunk)
                while len(buffer) >= _FRAME_BYTES:
                    frame_bytes = bytes(buffer[:_FRAME_BYTES])
                    del buffer[:_FRAME_BYTES]
                    samples = np.frombuffer(frame_bytes, dtype=np.int16)
                    if first_audio_marker is not None and not marker_appended:
                        first_audio_marker.append(time.monotonic())
                        marker_appended = True
                    yield AudioFrame(samples=samples, sample_rate=_SAMPLE_RATE_HZ)

        # Flush any trailing remainder as a final short frame. Pad to an
        # even byte count first — frombuffer requires int16-aligned bytes.
        if buffer:
            if len(buffer) % _BYTES_PER_SAMPLE != 0:
                buffer.append(0)
            samples = np.frombuffer(bytes(buffer), dtype=np.int16)
            if first_audio_marker is not None and not marker_appended:
                first_audio_marker.append(time.monotonic())
                marker_appended = True
            yield AudioFrame(samples=samples, sample_rate=_SAMPLE_RATE_HZ)

    # ------------------------------------------------------------------ #
    # Voice methods                                                      #
    # ------------------------------------------------------------------ #

    async def get_available_voices(self) -> list[str]:
        """Return speaker keys from the server, or ``[current_voice]`` on error.

        Falls back gracefully — the admin UI calls this on backend
        selection and we don't want an unreachable server to break the
        picker; the user can still type a custom voice key.
        """
        await self.prepare()
        assert self._client is not None
        url = f"{self._base_url}/speakers"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return [str(v) for v in payload]
            logger.warning(
                "XttsTTSAdapter: /speakers returned non-list payload %r; falling back to [current_voice]",
                payload,
            )
        except Exception as exc:
            logger.warning(
                "XttsTTSAdapter: GET /speakers failed (%s); falling back to [current_voice]=%r",
                exc,
                self._current_voice,
            )
        return [self._current_voice]

    def get_current_voice(self) -> str:
        """Return the active speaker key."""
        return self._current_voice

    async def change_voice(self, voice: str) -> str:
        """Set the active speaker key. Lenient — no server-side validation.

        Matches the elevenlabs/chatterbox contract: store the value as-is
        and let the server raise at synthesis time if it can't resolve.
        Admin-UI free-text inputs work.
        """
        self._current_voice = voice
        return f"XTTS voice set to {voice!r}"
