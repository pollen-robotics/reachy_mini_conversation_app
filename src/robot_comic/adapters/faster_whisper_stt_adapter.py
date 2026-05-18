"""FasterWhisperSTTAdapter: faster-whisper batch STT as :class:`STTBackend`.

Phase 5f of the pipeline refactor — alternate to :class:`MoonshineSTTAdapter`
for the on-robot STT slot. Issue #387 and the
``2026-05-16-moonshine-reliability-alternates-sibling-daemon.md`` memo
recommend faster-whisper ``tiny.en`` (CTranslate2 int8) as the lowest-risk
alternate to Moonshine: ~2s cold-load (vs Moonshine's ~20s), mature C++
runtime with no equivalent of Moonshine's "rearm-N-then-die" stall pattern
(#314), and a clean batch-call API.

Phase 5f.1 — VAD swap
---------------------

The original 5f adapter used ``silero-vad`` for utterance chunking.
``silero-vad`` transitively depends on ``torch`` (~2 GB unpacked) which
does NOT fit on the chassis eMMC (14 GB total, ~1.4 GB free after the
base install). 5f.1 swaps in ``webrtcvad`` — Google's WebRTC VAD wrapped
as a Python C extension. ~50 KB wheel, no torch, instant construction.

webrtcvad's contract differs from silero's:

- Constructed with an aggressiveness mode 0-3 (we default to 2).
- Accepts **bytes** (int16 PCM), not float32.
- Frame must be **exactly** 10/20/30 ms at 8/16/32/48 kHz. We pick
  16 kHz / 30 ms = 480 samples = 960 bytes — the longest webrtcvad
  accepts, which keeps per-frame overhead low.
- Returns a plain bool per frame, not start/end events.

We synthesise the silero-style start/end semantics with a small state
machine — see :class:`_WebRTCVadGate` below.

Architecture
------------

Faster-whisper is batch-only — it transcribes a complete utterance buffer
in one call, not a streaming pipeline. The adapter chunks streaming audio
frames into utterances via webrtcvad:

1. ``feed_audio`` receives an :class:`AudioFrame` (int16 PCM, any sample
   rate). The adapter converts to int16 mono @ 16 kHz.
2. The audio is fed into a fixed-size chunk buffer; each 480-sample chunk
   is passed through webrtcvad.
3. After ``_VAD_START_FRAMES`` consecutive speech frames, the adapter
   fires ``on_speech_started()`` (if registered) and begins accumulating
   audio into an utterance buffer.
4. After ``_VAD_END_SILENCE_FRAMES`` consecutive non-speech frames during
   an utterance, the accumulated buffer is submitted to
   ``WhisperModel.transcribe`` on a background thread (via
   :func:`asyncio.to_thread` so the asyncio loop keeps draining), and the
   joined segment text fires ``on_completed(text)``.

Partial transcripts
-------------------

Faster-whisper is batch — it does not emit partial transcripts mid-utterance.
``on_partial`` is therefore **never fired** by this adapter. This is a
documented limitation; the only consumer of ``on_partial`` today is the
orchestrator's decorative ``user_partial`` output-queue publish, which
already tolerates an adapter that opts out. A future revision could
sliding-window the buffer if partials become load-bearing.

Echo guard
----------

Mirrors :class:`MoonshineSTTAdapter`'s ``should_drop_frame`` callback shape
— when truthy, ``feed_audio`` short-circuits BEFORE any conversion / VAD
processing, so TTS-playback frames don't pollute the utterance buffer.

Dependencies
------------

This adapter requires the optional ``faster_whisper_stt`` extra:

.. code-block:: bash

    uv pip install -e .[faster_whisper_stt]

If the deps aren't installed, ``start()`` raises a clean
:class:`FasterWhisperSTTDependencyError` pointing the operator at the
install command — mirrors :class:`LocalSTTDependencyError` from
``local_stt_realtime.py``.
"""

from __future__ import annotations
import os
import asyncio
import logging
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from robot_comic.backends import (
    AudioFrame,
    TranscriptCallback,
    SpeechStartedCallback,
)


logger = logging.getLogger(__name__)


# webrtcvad accepts 10/20/30 ms frames at 8/16/32/48 kHz. We pick 16 kHz
# (faster-whisper's preferred sample rate) and 30 ms (the longest, which
# minimises per-frame call overhead).
_VAD_SAMPLE_RATE = 16000
_VAD_FRAME_MS = 30
_VAD_CHUNK_SAMPLES = _VAD_SAMPLE_RATE * _VAD_FRAME_MS // 1000  # 480

# ---------------------------------------------------------------------------
# Env-configurable VAD knobs
# ---------------------------------------------------------------------------
# Aggressiveness mode 0-3 (webrtcvad contract). Lower values are less
# aggressive at filtering non-speech; higher values reject more non-speech
# but risk clipping real speech. Default changed from 2 → 1 to reduce
# spurious VAD triggers from chassis-mic echo that drove the self-echo
# cascade (#460 root-cause follow-up). Override via env:
#   REACHY_MINI_FASTER_WHISPER_VAD_AGGRESSIVENESS=0..3
_VAD_AGGRESSIVENESS_DEFAULT = 1
_VAD_AGGRESSIVENESS_ENV = int(
    os.getenv("REACHY_MINI_FASTER_WHISPER_VAD_AGGRESSIVENESS", str(_VAD_AGGRESSIVENESS_DEFAULT))
)
if not (0 <= _VAD_AGGRESSIVENESS_ENV <= 3):
    logging.getLogger(__name__).warning(
        "REACHY_MINI_FASTER_WHISPER_VAD_AGGRESSIVENESS=%d is out of range [0, 3]; falling back to default %d.",
        _VAD_AGGRESSIVENESS_ENV,
        _VAD_AGGRESSIVENESS_DEFAULT,
    )
    _VAD_AGGRESSIVENESS_ENV = _VAD_AGGRESSIVENESS_DEFAULT
_VAD_AGGRESSIVENESS = _VAD_AGGRESSIVENESS_ENV

# Debounce: N consecutive speech frames before we declare an utterance
# started. 3 × 30 ms ≈ 90 ms — filters single-frame noise spikes.
_VAD_START_FRAMES = 3

# Trailing silence: N consecutive non-speech frames before we declare
# the utterance ended. Default changed from 17 → 25 (750 ms) to reduce
# premature utterance-end triggers that compounded the echo cascade.
# Override via env:
#   REACHY_MINI_FASTER_WHISPER_VAD_END_SILENCE_FRAMES=5..100
_VAD_END_SILENCE_FRAMES_DEFAULT = 25
_VAD_END_SILENCE_FRAMES_ENV = int(
    os.getenv(
        "REACHY_MINI_FASTER_WHISPER_VAD_END_SILENCE_FRAMES",
        str(_VAD_END_SILENCE_FRAMES_DEFAULT),
    )
)
if not (5 <= _VAD_END_SILENCE_FRAMES_ENV <= 100):
    logging.getLogger(__name__).warning(
        "REACHY_MINI_FASTER_WHISPER_VAD_END_SILENCE_FRAMES=%d is out of range [5, 100]; falling back to default %d.",
        _VAD_END_SILENCE_FRAMES_ENV,
        _VAD_END_SILENCE_FRAMES_DEFAULT,
    )
    _VAD_END_SILENCE_FRAMES_ENV = _VAD_END_SILENCE_FRAMES_DEFAULT
_VAD_END_SILENCE_FRAMES = _VAD_END_SILENCE_FRAMES_ENV


class FasterWhisperSTTDependencyError(RuntimeError):
    """Raised when the optional ``faster_whisper_stt`` extras aren't installed."""


class _WebRTCVadGate:
    """State machine wrapping ``webrtcvad.Vad`` with start/end semantics.

    webrtcvad gives one bool per frame; silero-vad gave us explicit
    ``start`` / ``end`` events. This class bridges the two:

    - After ``_VAD_START_FRAMES`` consecutive speech frames, ``feed()``
      returns ``"start"`` on the frame that crosses the boundary and
      transitions to the in-utterance state.
    - After ``_VAD_END_SILENCE_FRAMES`` consecutive non-speech frames
      while in an utterance, ``feed()`` returns ``"end"`` on the frame
      that crosses the boundary and transitions back to idle.
    - Otherwise ``feed()`` returns ``None``.

    The class is intentionally inert with regard to the audio payload —
    callers handle the buffer themselves. The gate only owns the
    is-speech bool + counter state.
    """

    def __init__(self, vad: Any) -> None:
        self._vad = vad
        self._in_utterance: bool = False
        self._speech_run: int = 0
        self._silence_run: int = 0

    def feed(self, frame_bytes: bytes) -> str | None:
        """Push one VAD-frame-sized chunk; return ``"start"``, ``"end"``, or ``None``.

        ``frame_bytes`` must be exactly ``_VAD_CHUNK_SAMPLES * 2`` bytes
        of int16 PCM at ``_VAD_SAMPLE_RATE``. webrtcvad will raise
        otherwise — that's a programming error, surface it.
        """
        try:
            is_speech = bool(self._vad.is_speech(frame_bytes, _VAD_SAMPLE_RATE))
        except Exception as e:
            logger.debug("webrtcvad.is_speech raised on frame: %s", e)
            return None

        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        if not self._in_utterance:
            if self._speech_run >= _VAD_START_FRAMES:
                self._in_utterance = True
                # Reset silence counter for the end-of-utterance check.
                self._silence_run = 0
                return "start"
            return None

        # In an utterance.
        if self._silence_run >= _VAD_END_SILENCE_FRAMES:
            self._in_utterance = False
            self._speech_run = 0
            self._silence_run = 0
            return "end"
        return None

    def reset(self) -> None:
        """Drop accumulated counters. Used on stop()."""
        self._in_utterance = False
        self._speech_run = 0
        self._silence_run = 0


class FasterWhisperSTTAdapter:
    """Adapter exposing faster-whisper batch STT as :class:`STTBackend`.

    Standalone-only — there is no host-coupled mode (unlike
    :class:`MoonshineSTTAdapter` which carried a transition shape for
    Phase 5e). Future STT backends follow this shape.
    """

    def __init__(
        self,
        *,
        should_drop_frame: Callable[[], bool] | None = None,
        model_name: str = "tiny.en",
        compute_type: str = "int8",
    ) -> None:
        """Capture echo-guard + model overrides; defer all model loads to ``start()``.

        ``should_drop_frame`` — when provided, called before every
        ``feed_audio`` frame is processed. Truthy return drops the frame.
        Intended to wire the same TTS-playback echo guard
        :class:`MoonshineSTTAdapter` uses (closure over
        ``host._speaking_until``).

        ``model_name`` / ``compute_type`` — passed straight through to
        :class:`faster_whisper.WhisperModel`. Defaults match the memo's
        recommendation (``tiny.en`` int8 quantised — ~75 MB, ~1-2s cold
        load on Pi 5).
        """
        self._should_drop_frame = should_drop_frame
        self._model_name = model_name
        self._compute_type = compute_type

        self._on_completed: TranscriptCallback | None = None
        self._on_speech_started: SpeechStartedCallback | None = None
        # on_partial intentionally NOT stored — faster-whisper is batch.

        self._model: Any = None
        self._vad_gate: _WebRTCVadGate | None = None
        # Pending audio that has not yet been split into VAD-frame-sized
        # chunks. int16 PCM @ 16 kHz mono.
        self._chunk_remainder: NDArray[np.int16] = np.zeros(0, dtype=np.int16)
        # Audio accumulated between VAD start and end events (int16 @ 16 kHz).
        self._utterance_buffer: list[NDArray[np.int16]] = []
        self._in_utterance: bool = False

    async def start(
        self,
        on_completed: TranscriptCallback,
        on_partial: TranscriptCallback | None = None,
        on_speech_started: SpeechStartedCallback | None = None,
    ) -> None:
        """Load the model + VAD and bind transcript callbacks.

        ``on_partial`` is accepted for Protocol compatibility but is
        **ignored** — faster-whisper is batch-only and does not produce
        partial transcripts. See the module docstring.
        """
        self._on_completed = on_completed
        self._on_speech_started = on_speech_started
        if on_partial is not None:
            logger.debug(
                "FasterWhisperSTTAdapter: on_partial provided but faster-whisper "
                "is batch-only; partial transcripts will never fire."
            )

        if self._model is not None:
            # Already started — rebind callbacks but skip the load.
            return

        try:
            await asyncio.to_thread(self._load_model_and_vad)
        except Exception:
            # Roll back state so a re-attempt isn't blocked.
            self._on_completed = None
            self._on_speech_started = None
            self._model = None
            self._vad_gate = None
            raise

    def _load_model_and_vad(self) -> None:
        """Load faster-whisper + webrtcvad. Synchronous — runs in a thread."""
        try:
            from faster_whisper import WhisperModel
        except Exception as e:  # pragma: no cover — covered via stubbed tests
            raise FasterWhisperSTTDependencyError(
                "faster-whisper STT requires the optional dependency group: "
                "install with `uv pip install -e .[faster_whisper_stt]`."
            ) from e

        try:
            import webrtcvad
        except Exception as e:  # pragma: no cover — covered via stubbed tests
            raise FasterWhisperSTTDependencyError(
                "faster-whisper STT requires webrtcvad: install with `uv pip install -e .[faster_whisper_stt]`."
            ) from e

        logger.info(
            "Loading faster-whisper STT: model=%s compute_type=%s",
            self._model_name,
            self._compute_type,
        )
        self._model = WhisperModel(
            self._model_name,
            device="cpu",
            compute_type=self._compute_type,
        )
        vad = webrtcvad.Vad(_VAD_AGGRESSIVENESS)
        self._vad_gate = _WebRTCVadGate(vad)

    async def feed_audio(self, frame: AudioFrame) -> None:
        """Push one audio frame through the VAD chunker.

        Consults ``should_drop_frame`` first (echo-guard) and short-circuits
        before any conversion if truthy. Otherwise: convert to int16 mono
        @ 16 kHz, append to the chunk-remainder buffer, and process every
        full 480-sample chunk through webrtcvad.
        """
        if self._should_drop_frame is not None and self._should_drop_frame():
            return
        if self._vad_gate is None or self._model is None:
            # start() not called or load failed — drop silently.
            return

        audio_i16 = _to_int16_mono(frame.samples, frame.sample_rate)
        if audio_i16.size == 0:
            return

        # Append + slice into VAD-frame-sized chunks.
        self._chunk_remainder = np.concatenate([self._chunk_remainder, audio_i16])
        while self._chunk_remainder.size >= _VAD_CHUNK_SAMPLES:
            chunk = self._chunk_remainder[:_VAD_CHUNK_SAMPLES]
            self._chunk_remainder = self._chunk_remainder[_VAD_CHUNK_SAMPLES:]
            await self._process_vad_chunk(chunk)

    async def _process_vad_chunk(self, chunk: NDArray[np.int16]) -> None:
        """Push one VAD-frame-sized chunk through webrtcvad; dispatch on start/end."""
        gate = self._vad_gate
        if gate is None:  # pragma: no cover — guarded by feed_audio
            return
        event = gate.feed(chunk.tobytes())

        if event == "start":
            # Echo guard at VAD-start boundary: if TTS is still playing when
            # webrtcvad declares speech started, the trigger is almost certainly
            # acoustic echo — discard this detection cycle entirely.
            # The per-frame guard in feed_audio fires only BEFORE audio is fed
            # to webrtcvad; it cannot catch the case where TTS ends mid-utterance
            # or where _speaking_until was already cleared before the VAD state
            # machine crosses the start threshold. Rechecking here closes that gap.
            if self._should_drop_frame is not None and self._should_drop_frame():
                return
            self._in_utterance = True
            # Restart the buffer with this chunk (speech began here, so
            # include the leading audio in the transcription buffer).
            self._utterance_buffer = [chunk]
            await self._fire_speech_started()
            return

        if self._in_utterance:
            self._utterance_buffer.append(chunk)

        if event == "end":
            # The "end" event lands after the trailing silence so the
            # buffer already contains the closing samples. Flush.
            buffer = self._utterance_buffer
            self._utterance_buffer = []
            self._in_utterance = False
            await self._transcribe_and_dispatch(buffer)

    async def _fire_speech_started(self) -> None:
        cb = self._on_speech_started
        if cb is None:
            return
        try:
            await cb()
        except Exception:
            logger.exception("FasterWhisperSTTAdapter on_speech_started callback raised")

    async def _transcribe_and_dispatch(self, buffer: list[NDArray[np.int16]]) -> None:
        """Run transcribe on a thread and fire ``on_completed`` with the text."""
        if not buffer:
            return
        cb = self._on_completed
        if cb is None:
            return
        audio_i16 = np.concatenate(buffer)
        # faster-whisper accepts float32 in [-1, 1]; convert from int16.
        audio = (audio_i16.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
        try:
            text = await asyncio.to_thread(self._transcribe_sync, audio)
        except Exception:
            logger.exception("faster-whisper transcribe raised")
            return
        text = text.strip()
        if not text:
            # Mirror the moonshine partial-empty-drop behaviour: empty
            # transcripts are uninteresting noise (e.g. webrtcvad fired
            # on a brief noise spike with no speech content).
            return
        try:
            await cb(text)
        except Exception:
            logger.exception("FasterWhisperSTTAdapter on_completed callback raised")

    def _transcribe_sync(self, audio: NDArray[np.float32]) -> str:
        """Run a synchronous transcribe call; invoked from the to_thread worker."""
        assert self._model is not None, "_transcribe_sync called before start()"
        # faster-whisper's transcribe returns ``(segments_iter, info)``.
        # segments_iter is a generator — iterating it triggers the actual
        # decode work. We join the .text fields with spaces; that mirrors
        # what an operator would expect to see as the final transcript.
        segments, _info = self._model.transcribe(audio, language="en")
        parts: list[str] = []
        for segment in segments:
            text = getattr(segment, "text", "")
            if isinstance(text, str) and text:
                parts.append(text.strip())
        return " ".join(parts).strip()

    async def stop(self) -> None:
        """Release model + VAD state. Idempotent."""
        model = self._model
        gate = self._vad_gate
        self._model = None
        self._vad_gate = None
        self._on_completed = None
        self._on_speech_started = None
        self._utterance_buffer = []
        self._chunk_remainder = np.zeros(0, dtype=np.int16)
        self._in_utterance = False

        def _close() -> None:
            if gate is not None:
                try:
                    gate.reset()
                except Exception as e:  # pragma: no cover — best-effort
                    logger.debug("FasterWhisperSTTAdapter VAD reset failed: %s", e)
            if model is not None:
                close = getattr(model, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as e:  # pragma: no cover — best-effort
                        logger.debug("FasterWhisperSTTAdapter model close failed: %s", e)

        if model is not None or gate is not None:
            await asyncio.to_thread(_close)

    async def reset_per_session_state(self) -> None:
        """No per-session accumulator state in this adapter — no-op.

        Phase 5c.2 hook the orchestrator fires on persona switch. The
        :class:`MoonshineSTTAdapter` analogue also relies on the
        :class:`STTBackend` Protocol default (no-op). We declare the
        method explicitly so structural :class:`STTBackend` conformance
        is unambiguous.
        """
        return None


def _to_int16_mono(samples: Any, sample_rate: int) -> NDArray[np.int16]:
    """Convert a frame's samples to int16 mono @ 16 kHz.

    webrtcvad expects int16 PCM bytes, so we keep the audio in int16 the
    whole way through the chunker and only float-convert for the actual
    transcribe call.
    """
    arr = samples
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.int16)

    # Downmix stereo / multi-channel to mono.
    if arr.ndim == 2:
        if arr.shape[1] > arr.shape[0]:
            arr = arr.T
        if arr.shape[1] > 1:
            arr = arr[:, 0]

    # Coerce to int16. AudioFrame's contract is int16 PCM but defend
    # against float inputs (e.g. tests passing float arrays).
    if arr.dtype != np.int16:
        if np.issubdtype(arr.dtype, np.floating):
            # Assume float in [-1, 1].
            arr = np.clip(arr, -1.0, 1.0)
            arr = (arr * 32767.0).astype(np.int16)
        else:
            arr = arr.astype(np.int16)

    if sample_rate != _VAD_SAMPLE_RATE and arr.size > 0:
        # Deferred import: scipy is heavy and the unit tests stub the model
        # path so they don't reach this branch in the common case.
        from scipy.signal import resample

        target_len = int(round(arr.size * _VAD_SAMPLE_RATE / sample_rate))
        # scipy.signal.resample returns float; clip + cast back to int16.
        resampled = resample(arr.astype(np.float32), target_len)
        resampled = np.clip(resampled, -32768.0, 32767.0)
        arr = resampled.astype(np.int16, copy=False)

    out: NDArray[np.int16] = arr.astype(np.int16, copy=False)
    return out
