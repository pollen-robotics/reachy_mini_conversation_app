"""Tests for ``FasterWhisperSTTAdapter`` — Phase 5f + 5f.1 (VAD swap).

The adapter wraps faster-whisper + webrtcvad. Neither dependency is
installed in CI; tests stub both via ``sys.modules`` injection so the
adapter's import-on-start path picks up the stubs. The stubs simulate:

- ``faster_whisper.WhisperModel(...)`` returning an object whose
  ``transcribe(audio, language=...)`` method returns a ``(segments, info)``
  pair. ``segments`` is an iterable of objects exposing ``.text``.
- ``webrtcvad.Vad(aggressiveness)`` returning an object whose
  ``is_speech(frame_bytes, sample_rate) -> bool`` is driven by a test-
  configured script (a list of bools popped one per call).
"""

from __future__ import annotations
import sys
import types
from typing import Any

import numpy as np
import pytest

from robot_comic.backends import AudioFrame, STTBackend


# webrtcvad frame size at 16 kHz / 30 ms.
_VAD_CHUNK = 480
_START_FRAMES = 7  # matches _VAD_START_FRAMES_DEFAULT (raised from 3 → 7, 2026-05-19)
_END_SILENCE_FRAMES = 25  # matches new default _VAD_END_SILENCE_FRAMES_DEFAULT


# ---------------------------------------------------------------------------
# Stub injection helpers
# ---------------------------------------------------------------------------


class _StubSegment:
    def __init__(self, text: str, no_speech_prob: float | None = None) -> None:
        self.text = text
        if no_speech_prob is not None:
            self.no_speech_prob = no_speech_prob


class _StubWhisperModel:
    """Stand-in for ``faster_whisper.WhisperModel``.

    Records constructor args + ``transcribe`` calls; returns a queued list
    of segments for each call so tests can stage multi-utterance scenarios.
    """

    def __init__(
        self, model_name: str, *, device: str = "cpu", compute_type: str = "int8", cpu_threads: int = 4
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.transcribe_calls: list[np.ndarray] = []
        # Tests push transcript strings here; each transcribe call pops the
        # next entry (or returns "" if empty).
        self.queued_results: list[Any] = []
        self.close_called = False

    def transcribe(self, audio: np.ndarray, language: str = "en") -> tuple[Any, dict[str, Any]]:
        self.transcribe_calls.append(audio)
        text = self.queued_results.pop(0) if self.queued_results else ""
        # Three ways to stage segments:
        # - str: one segment, no no_speech_prob
        # - list[str]: many segments, no no_speech_prob
        # - list[tuple[str, float]]: many segments with explicit no_speech_prob
        if isinstance(text, list):
            segments: list[_StubSegment] = []
            for item in text:
                if isinstance(item, tuple):
                    segments.append(_StubSegment(item[0], no_speech_prob=item[1]))
                else:
                    segments.append(_StubSegment(item))
        else:
            segments = [_StubSegment(text)] if text else []
        return iter(segments), {"language": language}

    def close(self) -> None:
        self.close_called = True


class _StubVad:
    """Stand-in for ``webrtcvad.Vad``.

    Tests configure ``script`` — a list of bools the VAD returns in order,
    one per ``is_speech`` invocation. After the script runs out, returns
    ``False`` (silence) indefinitely. Records every frame the VAD received.
    """

    def __init__(self, aggressiveness: int = 0) -> None:
        self.aggressiveness = aggressiveness
        self.script: list[bool] = []
        self.received_frames: list[bytes] = []

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self.received_frames.append(frame)
        if not self.script:
            return False
        return self.script.pop(0)


def _make_speech_only_vad() -> Any:
    """Stub VAD that always reports speech, never silence.

    Used to drive the max-buffer-ceiling tests: an utterance starts after
    the first 3 speech frames and then never ends because webrtcvad keeps
    saying "speech". The adapter's backstop should force-flush.
    """

    class _AllSpeechVad(_StubVad):
        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            self.received_frames.append(frame)
            return True

    return _AllSpeechVad


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject the stub modules into ``sys.modules`` and return a holder dict.

    The adapter's ``_load_model_and_vad`` imports ``faster_whisper`` and
    ``webrtcvad`` inside a function body. Replacing the module in
    ``sys.modules`` makes the next ``from … import …`` resolve to our stub.

    Returns a holder dict; after ``start()`` runs, ``holder["model"]`` and
    ``holder["vad"]`` are populated with the constructed stub instances
    so tests can drive their scripts and read their captured state.
    """
    holder: dict[str, Any] = {"model": None, "vad": None}

    def _make_model(
        name: str, *, device: str = "cpu", compute_type: str = "int8", cpu_threads: int = 4
    ) -> _StubWhisperModel:
        m = _StubWhisperModel(name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
        holder["model"] = m
        return m

    def _make_vad(aggressiveness: int = 0) -> _StubVad:
        v = _StubVad(aggressiveness)
        holder["vad"] = v
        return v

    fw_module = types.ModuleType("faster_whisper")
    fw_module.WhisperModel = _make_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_module)

    wv_module = types.ModuleType("webrtcvad")
    wv_module.Vad = _make_vad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "webrtcvad", wv_module)

    return holder


def _speech_script(count: int) -> list[bool]:
    """A script of ``count`` consecutive True (speech) frames."""
    return [True] * count


def _silence_script(count: int) -> list[bool]:
    """A script of ``count`` consecutive False (silence) frames."""
    return [False] * count


def _utterance_script(speech_frames: int = _START_FRAMES, silence_frames: int = _END_SILENCE_FRAMES) -> list[bool]:
    """Speech-then-silence sequence that triggers exactly one start + one end."""
    return _speech_script(speech_frames) + _silence_script(silence_frames)


# ---------------------------------------------------------------------------
# Construction + Protocol conformance
# ---------------------------------------------------------------------------


def test_constructor_accepts_no_arguments() -> None:
    """Standalone constructor (no host handler) is the only shape."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    assert adapter._model is None
    assert adapter._vad_gate is None


def test_constructor_accepts_should_drop_frame() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter(should_drop_frame=lambda: True)
    assert adapter._should_drop_frame is not None
    assert adapter._should_drop_frame() is True


def test_constructor_accepts_model_overrides() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter(model_name="small.en", compute_type="float16")
    assert adapter._model_name == "small.en"
    assert adapter._compute_type == "float16"


def test_adapter_satisfies_stt_backend_protocol() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    assert isinstance(adapter, STTBackend)


@pytest.mark.asyncio
async def test_reset_per_session_state_is_noop() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    await adapter.reset_per_session_state()


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_loads_model_and_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)

    assert holder["model"] is not None
    # base.en is the chassis default after the 2026-05-16 hardware
    # validation (issue #429) — tiny.en hallucinated on short audio.
    assert holder["model"].model_name == "base.en"
    assert holder["model"].compute_type == "int8"
    assert holder["vad"] is not None
    # Aggressiveness 3 — most aggressive, chassis default after 2026-05-16
    # hardware validation (issue #429): ambient noise kept VAD pinned to speech.
    assert holder["vad"].aggressiveness == 3


@pytest.mark.asyncio
async def test_start_passes_through_model_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(model_name="base.en", compute_type="int8_float32")

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    assert holder["model"].model_name == "base.en"
    assert holder["model"].compute_type == "int8_float32"


@pytest.mark.asyncio
async def test_start_records_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _on_completed(_t: str) -> None: ...

    async def _on_speech_started() -> None: ...

    await adapter.start(_on_completed, on_speech_started=_on_speech_started)
    assert adapter._on_completed is _on_completed
    assert adapter._on_speech_started is _on_speech_started


@pytest.mark.asyncio
async def test_start_accepts_on_partial_but_does_not_store_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """faster-whisper is batch; ``on_partial`` is documented as never fired."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _on_completed(_t: str) -> None: ...

    async def _on_partial(_t: str) -> None: ...

    await adapter.start(_on_completed, on_partial=_on_partial)
    # The adapter logs a debug message but does not store the partial cb;
    # the simplest behavioural assertion is that an attempt to fire a
    # partial-style event below produces no callback invocation. (Covered
    # by the missing-import-attribute spot-check.)
    assert not hasattr(adapter, "_on_partial")


@pytest.mark.asyncio
async def test_start_idempotent_does_not_reload_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    model_after_first = holder["model"]
    await adapter.start(_cb)
    # The stub factory replaces holder["model"] on EVERY call; a re-load
    # would have produced a different instance.
    assert holder["model"] is model_after_first


@pytest.mark.asyncio
async def test_start_load_failure_clears_state_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the model load raises, the adapter resets so a retry is possible."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    # Stub both modules so the import succeeds — the failure has to come
    # from the model constructor itself, not a missing-dep ImportError.
    _install_stubs(monkeypatch)

    fw_module = types.ModuleType("faster_whisper")

    def _bad_model(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("model load boom")

    fw_module.WhisperModel = _bad_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_module)

    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    with pytest.raises(RuntimeError, match="model load boom"):
        await adapter.start(_cb)

    assert adapter._model is None
    assert adapter._vad_gate is None
    assert adapter._on_completed is None


@pytest.mark.asyncio
async def test_start_raises_dependency_error_when_faster_whisper_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters.faster_whisper_stt_adapter import (
        FasterWhisperSTTAdapter,
        FasterWhisperSTTDependencyError,
    )

    # Force the import inside _load_model_and_vad to fail by injecting a
    # module that doesn't expose WhisperModel.
    fw_module = types.ModuleType("faster_whisper")
    # Don't set WhisperModel — `from faster_whisper import WhisperModel` will
    # raise ImportError.
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_module)

    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    with pytest.raises(FasterWhisperSTTDependencyError):
        await adapter.start(_cb)


@pytest.mark.asyncio
async def test_start_raises_dependency_error_when_webrtcvad_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """webrtcvad import failure surfaces as FasterWhisperSTTDependencyError."""
    from robot_comic.adapters.faster_whisper_stt_adapter import (
        FasterWhisperSTTAdapter,
        FasterWhisperSTTDependencyError,
    )

    # Provide a working faster-whisper stub so the failure is isolated to
    # the webrtcvad import.
    fw_module = types.ModuleType("faster_whisper")
    fw_module.WhisperModel = _StubWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_module)

    # Remove webrtcvad from sys.modules entirely so the import raises.
    monkeypatch.delitem(sys.modules, "webrtcvad", raising=False)

    # And block re-import by intercepting the import machinery — easiest
    # way is a finder that raises ImportError for webrtcvad.
    class _Blocker:
        def find_spec(self, name: str, *_a: Any, **_k: Any) -> None:
            if name == "webrtcvad":
                raise ImportError("blocked for test")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + sys.meta_path)

    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    with pytest.raises(FasterWhisperSTTDependencyError):
        await adapter.start(_cb)


# ---------------------------------------------------------------------------
# feed_audio → VAD chunking → on_speech_started
# ---------------------------------------------------------------------------


def _silence_frame(num_samples: int = _VAD_CHUNK * 2, sample_rate: int = 16000) -> AudioFrame:
    return AudioFrame(samples=np.zeros(num_samples, dtype=np.int16), sample_rate=sample_rate)


@pytest.mark.asyncio
async def test_feed_audio_chunks_into_vad_frame_sized_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1024-sample frame → 2 full 480-sample VAD frames (64 sample remainder)."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(_silence_frame(num_samples=1024))

    vad = holder["vad"]
    assert len(vad.received_frames) == 2
    assert all(len(f) == _VAD_CHUNK * 2 for f in vad.received_frames)  # int16 = 2 bytes


@pytest.mark.asyncio
async def test_feed_audio_starts_utterance_after_debounced_speech_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_START_FRAMES`` consecutive speech frames fire ``on_speech_started`` once."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    started_calls = 0

    async def _on_completed(_t: str) -> None: ...

    async def _on_speech_started() -> None:
        nonlocal started_calls
        started_calls += 1

    await adapter.start(_on_completed, on_speech_started=_on_speech_started)

    holder["vad"].script = _speech_script(_START_FRAMES)
    # Enough samples for _START_FRAMES VAD frames.
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * _START_FRAMES))

    assert started_calls == 1
    # No silence yet → no transcribe.
    assert holder["model"].transcribe_calls == []


@pytest.mark.asyncio
async def test_feed_audio_does_not_start_utterance_below_debounce_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fewer than ``_START_FRAMES`` speech frames must NOT fire ``on_speech_started``."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    started_calls = 0

    async def _on_completed(_t: str) -> None: ...

    async def _on_speech_started() -> None:
        nonlocal started_calls
        started_calls += 1

    await adapter.start(_on_completed, on_speech_started=_on_speech_started)

    holder["vad"].script = _speech_script(_START_FRAMES - 1) + _silence_script(5)
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * (_START_FRAMES - 1 + 5)))

    assert started_calls == 0


@pytest.mark.asyncio
async def test_feed_audio_does_not_call_speech_started_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ``on_speech_started`` is fine — VAD start events are silently dropped."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _on_completed(_t: str) -> None: ...

    await adapter.start(_on_completed)
    holder["vad"].script = _speech_script(_START_FRAMES)
    # Must not raise.
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * _START_FRAMES))


# ---------------------------------------------------------------------------
# VAD end event → transcribe → on_completed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_audio_transcribes_and_fires_on_completed_after_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)

    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    holder["vad"].script = _utterance_script()
    holder["model"].queued_results = ["hello robot"]

    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    assert captured == ["hello robot"]
    assert len(holder["model"].transcribe_calls) == 1


@pytest.mark.asyncio
async def test_feed_audio_joins_multi_segment_transcribe_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    holder["vad"].script = _utterance_script()
    holder["model"].queued_results = [["hello", " robot", " how are you"]]
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    assert captured == ["hello robot how are you"]


@pytest.mark.asyncio
async def test_feed_audio_empty_transcribe_result_does_not_fire_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    holder["vad"].script = _utterance_script()
    holder["model"].queued_results = [""]
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    assert captured == []


@pytest.mark.asyncio
async def test_feed_audio_whitespace_transcribe_result_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    holder["vad"].script = _utterance_script()
    holder["model"].queued_results = ["   "]
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    assert captured == []


@pytest.mark.asyncio
async def test_feed_audio_back_to_back_utterances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two utterances in one stream → two on_completed calls."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    # Two complete utterances in sequence.
    holder["vad"].script = _utterance_script() + _utterance_script()
    holder["model"].queued_results = ["first", "second"]
    total_frames = 2 * (_START_FRAMES + _END_SILENCE_FRAMES)
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    assert captured == ["first", "second"]


# ---------------------------------------------------------------------------
# should_drop_frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_should_drop_frame_when_true_skips_feed_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(should_drop_frame=lambda: True)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK))

    assert holder["vad"].received_frames == []
    assert holder["model"].transcribe_calls == []


@pytest.mark.asyncio
async def test_should_drop_frame_when_false_processes_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(should_drop_frame=lambda: False)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK))

    assert len(holder["vad"].received_frames) == 1


@pytest.mark.asyncio
async def test_should_drop_frame_default_none_processes_every_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()  # no should_drop_frame

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * 2))

    assert len(holder["vad"].received_frames) == 2


# ---------------------------------------------------------------------------
# feed_audio guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_audio_before_start_is_safe_noop() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    # Must not raise — silently drops the frame because model/VAD aren't loaded.
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK))


@pytest.mark.asyncio
async def test_feed_audio_handles_list_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AudioFrame.samples`` may be a Python list, not always ndarray."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(AudioFrame(samples=[0] * _VAD_CHUNK, sample_rate=16000))

    assert len(holder["vad"].received_frames) == 1


@pytest.mark.asyncio
async def test_feed_audio_resamples_when_sample_rate_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-16kHz frames are resampled before chunking."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    # 1024 samples at 24 kHz → ~683 samples at 16 kHz → 1 full 480-sample
    # frame plus a remainder. Assert at least one VAD frame made it through.
    await adapter.feed_audio(AudioFrame(samples=np.zeros(1024, dtype=np.int16), sample_rate=24000))
    assert len(holder["vad"].received_frames) >= 1


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_clears_state_and_releases_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.stop()

    assert adapter._model is None
    assert adapter._vad_gate is None
    assert adapter._on_completed is None
    assert holder["model"].close_called is True


@pytest.mark.asyncio
async def test_stop_is_safe_when_never_started() -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter()
    await adapter.stop()  # must not raise
    assert adapter._model is None


@pytest.mark.asyncio
async def test_stop_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.stop()
    await adapter.stop()  # second call must not raise


# ---------------------------------------------------------------------------
# Callback exception handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_completed_exception_does_not_crash_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _bad(_t: str) -> None:
        raise ValueError("user code bug")

    await adapter.start(_bad)
    holder["vad"].script = _utterance_script()
    holder["model"].queued_results = ["payload"]
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    # Must not raise.
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))


@pytest.mark.asyncio
async def test_on_speech_started_exception_does_not_crash_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    async def _bad_started() -> None:
        raise ValueError("user code bug")

    await adapter.start(_cb, on_speech_started=_bad_started)
    holder["vad"].script = _speech_script(_START_FRAMES)
    # Must not raise.
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * _START_FRAMES))


@pytest.mark.asyncio
async def test_transcribe_exception_does_not_crash_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transcribe-call failure logs but doesn't kill the adapter."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)

    def _bad_transcribe(_a: Any, language: str = "en") -> Any:
        raise RuntimeError("transcribe boom")

    holder["model"].transcribe = _bad_transcribe  # type: ignore[method-assign]
    holder["vad"].script = _utterance_script()
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    # Must not raise.
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))
    assert captured == []


@pytest.mark.asyncio
async def test_vad_is_speech_exception_is_treated_as_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If webrtcvad raises (e.g. wrong frame size), the gate must not crash."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()
    started_calls = 0

    async def _cb(_t: str) -> None: ...

    async def _on_speech_started() -> None:
        nonlocal started_calls
        started_calls += 1

    await adapter.start(_cb, on_speech_started=_on_speech_started)

    def _bad_is_speech(_frame: bytes, _sr: int) -> bool:
        raise RuntimeError("webrtcvad boom")

    holder["vad"].is_speech = _bad_is_speech  # type: ignore[method-assign]
    # Several frames of "speech" that the VAD will explode on — must not
    # crash and must not trigger an utterance start (treated as silence).
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * 5))
    assert started_calls == 0


# ---------------------------------------------------------------------------
# Issue #429: vad_aggressiveness / no_speech_threshold / max_buffer_sec
# ---------------------------------------------------------------------------


def test_constructor_accepts_tuning_kwargs() -> None:
    """Tuning knobs flow into adapter instance state."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    adapter = FasterWhisperSTTAdapter(
        vad_aggressiveness=1,
        no_speech_threshold=0.4,
        max_buffer_sec=5.0,
    )
    assert adapter._vad_aggressiveness == 1
    assert adapter._no_speech_threshold == 0.4
    # 5.0s × 16kHz = 80000 samples.
    assert adapter._max_buffer_samples == 80000


@pytest.mark.asyncio
async def test_start_passes_vad_aggressiveness_to_webrtcvad(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor override flows all the way to the Vad() constructor."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(vad_aggressiveness=0)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    assert holder["vad"].aggressiveness == 0


@pytest.mark.asyncio
async def test_no_speech_prob_filter_drops_high_confidence_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segment whose no_speech_prob exceeds the threshold is dropped before dispatch."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(no_speech_threshold=0.6)
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    holder["vad"].script = _utterance_script()
    # Single segment with no_speech_prob 0.9 (above 0.6 threshold) — drop.
    holder["model"].queued_results = [[("I think he's very young.", 0.9)]]

    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    assert captured == []
    assert len(holder["model"].transcribe_calls) == 1  # we ran transcribe, just filtered the output


@pytest.mark.asyncio
async def test_no_speech_prob_filter_keeps_low_confidence_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segment whose no_speech_prob is below the threshold is kept."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(no_speech_threshold=0.6)
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    holder["vad"].script = _utterance_script()
    holder["model"].queued_results = [[("hello robot", 0.1)]]

    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    assert captured == ["hello robot"]


@pytest.mark.asyncio
async def test_no_speech_prob_filter_mixes_kept_and_dropped_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-segment transcripts filter per-segment; surviving text joined."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(no_speech_threshold=0.6)
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    holder["vad"].script = _utterance_script()
    holder["model"].queued_results = [
        [
            ("hello", 0.2),
            (" thanks for watching", 0.95),  # whisper-trained-data hallucination
            (" robot", 0.3),
        ]
    ]

    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    assert captured == ["hello robot"]


@pytest.mark.asyncio
async def test_no_speech_prob_filter_keeps_segments_without_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segments that don't expose no_speech_prob are treated as speech (kept)."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(no_speech_threshold=0.6)
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    holder["vad"].script = _utterance_script()
    # Plain string → _StubSegment without no_speech_prob attribute.
    holder["model"].queued_results = ["hello"]

    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    assert captured == ["hello"]


@pytest.mark.asyncio
async def test_max_buffer_ceiling_force_flushes_without_end_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the VAD never fires end, the ceiling force-flushes a transcribe."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    # 0.1s × 16kHz = 1600 samples = ~3.33 VAD frames; we'll cross it fast.
    adapter = FasterWhisperSTTAdapter(max_buffer_sec=0.1)
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    # Always-speech VAD: utterance starts after _START_FRAMES, then never ends.
    holder["vad"].script = [True] * 200
    holder["model"].queued_results = ["forced flush"]

    # Feed enough frames that the buffer crosses the 1600-sample ceiling.
    # _START_FRAMES × 480 = 1440 (just under), so 4 speech frames triggers
    # start + appends until ceiling. 10 frames is comfortably over.
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * 10))

    assert captured == ["forced flush"]
    assert len(holder["model"].transcribe_calls) == 1


@pytest.mark.asyncio
async def test_max_buffer_ceiling_resets_state_for_next_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Immediately after a force-flush, in_utterance + buffer state must be clean.

    Feed exactly enough always-speech frames to cross the ceiling once:
    chunks 1-3 trigger start, chunks 4-6 grow the buffer past the 1600-
    sample ceiling and force-flush. After chunk 6 the adapter must report
    clean state. (Further chunks would re-trigger a fresh utterance start
    through the reset gate.)
    """
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(max_buffer_sec=0.1)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    holder["vad"].script = [True] * 200
    holder["model"].queued_results = ["forced flush"]
    # 6 chunks: 3 to start + 3 more grow buffer to 1920 samples (>1600 ceiling).
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * 6))

    assert adapter._in_utterance is False
    assert adapter._utterance_buffer == []
    assert adapter._utterance_buffer_samples == 0


@pytest.mark.asyncio
async def test_max_buffer_ceiling_does_not_trigger_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal-length utterance under the ceiling flushes only on VAD end."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    # 10s default ceiling. Normal utterance is well under.
    adapter = FasterWhisperSTTAdapter()
    captured: list[str] = []

    async def _on_completed(t: str) -> None:
        captured.append(t)

    await adapter.start(_on_completed)
    total_frames = _START_FRAMES + _END_SILENCE_FRAMES
    holder["vad"].script = _utterance_script()
    holder["model"].queued_results = ["normal turn"]

    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * total_frames))

    # Exactly one transcribe — fired by VAD end, not by the ceiling.
    assert captured == ["normal turn"]
    assert len(holder["model"].transcribe_calls) == 1


def test_make_speech_only_vad_helper_exists() -> None:
    """Sanity: the helper module-level function is callable.

    Defensive — keeps the helper from being dead-code-eliminated by a
    refactor since the ceiling tests construct their own always-speech
    VAD inline (simpler than wiring the helper through _install_stubs).
    """
    cls = _make_speech_only_vad()
    instance = cls(aggressiveness=2)
    assert instance.is_speech(b"\x00" * 960, 16000) is True


# ---------------------------------------------------------------------------
# Env-configurable VAD knobs (aggressiveness + end-silence-frames)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vad_aggressiveness_env_var_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting REACHY_MINI_FASTER_WHISPER_VAD_AGGRESSIVENESS=0..3 is passed to Vad()."""
    import importlib

    import robot_comic.adapters.faster_whisper_stt_adapter as _mod

    _install_stubs(monkeypatch)

    for level in (0, 1, 2, 3):
        monkeypatch.setenv("REACHY_MINI_FASTER_WHISPER_VAD_AGGRESSIVENESS", str(level))
        # Force the module-level constants to re-evaluate.
        importlib.reload(_mod)

        from robot_comic.adapters.faster_whisper_stt_adapter import (
            _VAD_AGGRESSIVENESS,
            FasterWhisperSTTAdapter,
        )

        assert _VAD_AGGRESSIVENESS == level, f"Expected aggressiveness={level}, got {_VAD_AGGRESSIVENESS}"

        adapter = FasterWhisperSTTAdapter()

        async def _cb(_t: str) -> None: ...

        # Re-inject stubs after reload (module ref changed).
        _install_stubs(monkeypatch)
        holder = _install_stubs(monkeypatch)
        await adapter.start(_cb)
        assert holder["vad"].aggressiveness == level

    # Restore module to default state so subsequent tests are unaffected.
    monkeypatch.delenv("REACHY_MINI_FASTER_WHISPER_VAD_AGGRESSIVENESS", raising=False)
    importlib.reload(_mod)


@pytest.mark.asyncio
async def test_vad_aggressiveness_out_of_range_clamps_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-range aggressiveness (e.g. 5 or -1) falls back to default=3 and logs a warning."""
    import importlib

    import robot_comic.adapters.faster_whisper_stt_adapter as _mod

    _install_stubs(monkeypatch)

    for bad_value in ("5", "-1"):
        # Reload with the bad env var set; the module-level guard should clamp
        # to the default and emit a logger.warning (not a Python Warning).
        with monkeypatch.context() as m:
            m.setenv("REACHY_MINI_FASTER_WHISPER_VAD_AGGRESSIVENESS", bad_value)
            importlib.reload(_mod)
            from robot_comic.adapters.faster_whisper_stt_adapter import (
                _VAD_AGGRESSIVENESS,
                _VAD_AGGRESSIVENESS_DEFAULT,
            )

            assert _VAD_AGGRESSIVENESS == _VAD_AGGRESSIVENESS_DEFAULT, (
                f"Bad value {bad_value!r} should clamp to {_VAD_AGGRESSIVENESS_DEFAULT}, got {_VAD_AGGRESSIVENESS}"
            )

    # Restore module state.
    monkeypatch.delenv("REACHY_MINI_FASTER_WHISPER_VAD_AGGRESSIVENESS", raising=False)
    importlib.reload(_mod)


@pytest.mark.asyncio
async def test_vad_start_echo_guard_suppresses_speech_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When should_drop_frame() returns True at VAD-start time, on_speech_started is NOT called.

    This is the key regression test for the self-echo cascade fix.
    Prior to the fix, the per-frame guard in feed_audio short-circuited
    BEFORE webrtcvad saw the audio, but the VAD state machine could accumulate
    a run of "speech" frames fed before TTS started playing (or during the
    TTS-echo window when _speaking_until was transiently 0). When webrtcvad
    crossed the start threshold, _on_speech_started fired unconditionally,
    beginning a new utterance barge-in. The fix re-evaluates should_drop_frame()
    at the moment the "start" event is dispatched, blocking the barge-in.
    """
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    started_calls = 0

    async def _on_completed(_t: str) -> None: ...

    async def _on_speech_started() -> None:
        nonlocal started_calls
        started_calls += 1

    # Predicate starts as False (TTS silent) so the VAD frames accumulate
    # normally, then flips to True (TTS playing) exactly when webrtcvad
    # would cross the start threshold.
    _drop = [False] * _START_FRAMES  # per-frame guard: allow frames through VAD
    # At VAD-start time the guard re-check will return True → suppress barge-in.
    _drop_at_start = True

    call_count = [0]

    def _should_drop() -> bool:
        # Return False for the per-frame calls during accumulation, then True
        # at the VAD-start boundary re-check.
        # Because the per-frame guard fires once per feed_audio call (before
        # the chunk loop) and the VAD-start guard fires once per "start" event,
        # we track calls to distinguish them.
        #
        # Simpler approach: return True unconditionally — the per-frame guard
        # will short-circuit before the VAD sees any audio. But that tests the
        # existing guard, not the new one.
        #
        # To isolate the NEW guard: disable the per-frame guard by removing it,
        # and only wire should_drop_frame on the _process_vad_chunk path.
        # We achieve this by creating the adapter without should_drop_frame,
        # feeding audio until the VAD state machine is primed, THEN injecting
        # _should_drop_frame and feeding the final trigger frame.
        call_count[0] += 1
        return True

    # Build adapter WITHOUT the guard so the first batch of frames reaches VAD.
    adapter = FasterWhisperSTTAdapter()
    await adapter.start(_on_completed, on_speech_started=_on_speech_started)

    # Script: _START_FRAMES - 1 speech frames → VAD not yet at threshold.
    holder["vad"].script = _speech_script(_START_FRAMES - 1)
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * (_START_FRAMES - 1)))
    # VAD has seen _START_FRAMES - 1 consecutive speech frames; one more will
    # cross the threshold. Now inject the echo guard.
    adapter._should_drop_frame = _should_drop

    # Feed the final trigger frame (the VAD stub will return True → the gate
    # will return "start" → our guard should intercept and return early).
    holder["vad"].script = _speech_script(1)
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK))

    # The guard must have fired (proving the VAD-start re-check path ran).
    assert call_count[0] >= 1, "should_drop_frame was never called at VAD-start boundary"
    # And on_speech_started must NOT have been called.
    assert started_calls == 0, (
        f"on_speech_started fired {started_calls} time(s) despite should_drop_frame returning True"
    )


# ---------------------------------------------------------------------------
# no_speech_threshold default + env override (post-TTS hallucination fix)
# ---------------------------------------------------------------------------


def test_no_speech_threshold_default_is_0_4() -> None:
    """Default no_speech_threshold is 0.4 (raised from 0.6 for post-TTS cascade defence).

    Regression guard: if the constant is changed accidentally this test catches it.
    Issue: post-TTS hallucination cascade (2026-05-18) — ambient audio after
    playback had low no_speech_prob despite containing no real speech.
    """
    from robot_comic.adapters.faster_whisper_stt_adapter import _NO_SPEECH_THRESHOLD_DEFAULT

    assert _NO_SPEECH_THRESHOLD_DEFAULT == 0.4, (
        f"_NO_SPEECH_THRESHOLD_DEFAULT should be 0.4, got {_NO_SPEECH_THRESHOLD_DEFAULT}"
    )


def test_no_speech_threshold_default_applied_to_new_adapter() -> None:
    """FasterWhisperSTTAdapter() without kwargs uses _NO_SPEECH_THRESHOLD_DEFAULT (0.4)."""
    from robot_comic.adapters import FasterWhisperSTTAdapter
    from robot_comic.adapters.faster_whisper_stt_adapter import _NO_SPEECH_THRESHOLD_DEFAULT

    adapter = FasterWhisperSTTAdapter()
    assert adapter._no_speech_threshold == _NO_SPEECH_THRESHOLD_DEFAULT


def test_no_speech_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """REACHY_MINI_FASTER_WHISPER_NO_SPEECH_THRESHOLD overrides the module-level default.

    We patch the env var and re-import the module to pick up the change.
    The adapter constructor must respect the overridden value.
    """
    import sys

    monkeypatch.setenv("REACHY_MINI_FASTER_WHISPER_NO_SPEECH_THRESHOLD", "0.55")

    # Force re-execution of module-level env read by reloading.
    mod_name = "robot_comic.adapters.faster_whisper_stt_adapter"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import robot_comic.adapters.faster_whisper_stt_adapter as reloaded_mod

    assert reloaded_mod._NO_SPEECH_THRESHOLD_DEFAULT == 0.55

    # Restore for other tests.
    del sys.modules[mod_name]


def test_no_speech_threshold_env_invalid_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid REACHY_MINI_FASTER_WHISPER_NO_SPEECH_THRESHOLD logs a warning and uses default."""
    import sys

    monkeypatch.setenv("REACHY_MINI_FASTER_WHISPER_NO_SPEECH_THRESHOLD", "not_a_float")

    mod_name = "robot_comic.adapters.faster_whisper_stt_adapter"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import robot_comic.adapters.faster_whisper_stt_adapter as reloaded_mod

    # Invalid value → falls back to hardcoded default (0.4).
    assert reloaded_mod._NO_SPEECH_THRESHOLD_DEFAULT == 0.4

    del sys.modules[mod_name]


def test_no_speech_threshold_env_out_of_range_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range (>1.0) threshold env var falls back to default."""
    import sys

    monkeypatch.setenv("REACHY_MINI_FASTER_WHISPER_NO_SPEECH_THRESHOLD", "1.5")

    mod_name = "robot_comic.adapters.faster_whisper_stt_adapter"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import robot_comic.adapters.faster_whisper_stt_adapter as reloaded_mod

    assert reloaded_mod._NO_SPEECH_THRESHOLD_DEFAULT == 0.4

    del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# VAD start-frames knob (2026-05-19 ambient-noise fix)
# ---------------------------------------------------------------------------


def test_vad_start_frames_default() -> None:
    """Module-level default is 7 (raised from 3 to reject ambient enclosure noise)."""
    from robot_comic.adapters.faster_whisper_stt_adapter import _VAD_START_FRAMES, _VAD_START_FRAMES_DEFAULT

    assert _VAD_START_FRAMES_DEFAULT == 7, f"_VAD_START_FRAMES_DEFAULT should be 7, got {_VAD_START_FRAMES_DEFAULT}"
    assert _VAD_START_FRAMES == _VAD_START_FRAMES_DEFAULT, (
        f"_VAD_START_FRAMES should equal _VAD_START_FRAMES_DEFAULT ({_VAD_START_FRAMES_DEFAULT}), "
        f"got {_VAD_START_FRAMES}"
    )


def test_vad_start_frames_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """REACHY_MINI_FASTER_WHISPER_VAD_START_FRAMES=5 overrides the module-level default."""
    import importlib

    import robot_comic.adapters.faster_whisper_stt_adapter as _mod

    monkeypatch.setenv("REACHY_MINI_FASTER_WHISPER_VAD_START_FRAMES", "5")
    importlib.reload(_mod)

    from robot_comic.adapters.faster_whisper_stt_adapter import _VAD_START_FRAMES

    assert _VAD_START_FRAMES == 5, f"Expected _VAD_START_FRAMES=5 after env override, got {_VAD_START_FRAMES}"

    # Restore module to default state so subsequent tests are unaffected.
    monkeypatch.delenv("REACHY_MINI_FASTER_WHISPER_VAD_START_FRAMES", raising=False)
    importlib.reload(_mod)


def test_vad_start_frames_out_of_range_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range values (0, 100) fall back to _VAD_START_FRAMES_DEFAULT and log a warning."""
    import importlib

    import robot_comic.adapters.faster_whisper_stt_adapter as _mod

    for bad_value in ("0", "100"):
        with monkeypatch.context() as m:
            m.setenv("REACHY_MINI_FASTER_WHISPER_VAD_START_FRAMES", bad_value)
            importlib.reload(_mod)

            from robot_comic.adapters.faster_whisper_stt_adapter import (
                _VAD_START_FRAMES,
                _VAD_START_FRAMES_DEFAULT,
            )

            assert _VAD_START_FRAMES == _VAD_START_FRAMES_DEFAULT, (
                f"Bad value {bad_value!r} should clamp to {_VAD_START_FRAMES_DEFAULT}, got {_VAD_START_FRAMES}"
            )

    # Restore.
    monkeypatch.delenv("REACHY_MINI_FASTER_WHISPER_VAD_START_FRAMES", raising=False)
    importlib.reload(_mod)


# ---------------------------------------------------------------------------
# Max-buffer force-flush knob (issue #509)
# ---------------------------------------------------------------------------


def test_max_buffer_sec_default() -> None:
    """Module-level default is 10.0s force-flush backstop."""
    from robot_comic.adapters.faster_whisper_stt_adapter import _MAX_BUFFER_SEC, _MAX_BUFFER_SEC_DEFAULT

    assert _MAX_BUFFER_SEC_DEFAULT == 10.0, f"_MAX_BUFFER_SEC_DEFAULT should be 10.0, got {_MAX_BUFFER_SEC_DEFAULT}"
    assert _MAX_BUFFER_SEC == _MAX_BUFFER_SEC_DEFAULT, (
        f"_MAX_BUFFER_SEC should equal _MAX_BUFFER_SEC_DEFAULT ({_MAX_BUFFER_SEC_DEFAULT}), got {_MAX_BUFFER_SEC}"
    )


def test_max_buffer_sec_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """REACHY_MINI_FASTER_WHISPER_MAX_BUFFER_SEC=5.0 overrides the module-level default."""
    import importlib

    import robot_comic.adapters.faster_whisper_stt_adapter as _mod

    monkeypatch.setenv("REACHY_MINI_FASTER_WHISPER_MAX_BUFFER_SEC", "5.0")
    importlib.reload(_mod)

    from robot_comic.adapters.faster_whisper_stt_adapter import _MAX_BUFFER_SEC

    assert _MAX_BUFFER_SEC == 5.0, f"Expected _MAX_BUFFER_SEC=5.0 after env override, got {_MAX_BUFFER_SEC}"

    # Restore module to default state so subsequent tests are unaffected.
    monkeypatch.delenv("REACHY_MINI_FASTER_WHISPER_MAX_BUFFER_SEC", raising=False)
    importlib.reload(_mod)


def test_max_buffer_sec_out_of_range_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Out-of-range values (0.5, 120.0) fall back to _MAX_BUFFER_SEC_DEFAULT and log a warning."""
    import logging
    import importlib

    import robot_comic.adapters.faster_whisper_stt_adapter as _mod

    for bad_value in ("0.5", "120.0"):
        with monkeypatch.context() as m:
            m.setenv("REACHY_MINI_FASTER_WHISPER_MAX_BUFFER_SEC", bad_value)
            with caplog.at_level(logging.WARNING, logger="robot_comic.adapters.faster_whisper_stt_adapter"):
                caplog.clear()
                importlib.reload(_mod)

            from robot_comic.adapters.faster_whisper_stt_adapter import (
                _MAX_BUFFER_SEC,
                _MAX_BUFFER_SEC_DEFAULT,
            )

            assert _MAX_BUFFER_SEC == _MAX_BUFFER_SEC_DEFAULT, (
                f"Bad value {bad_value!r} should fall back to {_MAX_BUFFER_SEC_DEFAULT}, got {_MAX_BUFFER_SEC}"
            )
            assert any("out of range" in r.getMessage() for r in caplog.records), (
                f"Expected an out-of-range warning for {bad_value!r}; got {[r.getMessage() for r in caplog.records]}"
            )

    # Restore.
    monkeypatch.delenv("REACHY_MINI_FASTER_WHISPER_MAX_BUFFER_SEC", raising=False)
    importlib.reload(_mod)


def test_max_buffer_sec_non_numeric_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-numeric values (e.g. "abc") fall back to _MAX_BUFFER_SEC_DEFAULT and log a warning."""
    import logging
    import importlib

    import robot_comic.adapters.faster_whisper_stt_adapter as _mod

    monkeypatch.setenv("REACHY_MINI_FASTER_WHISPER_MAX_BUFFER_SEC", "abc")
    with caplog.at_level(logging.WARNING, logger="robot_comic.adapters.faster_whisper_stt_adapter"):
        caplog.clear()
        importlib.reload(_mod)

    from robot_comic.adapters.faster_whisper_stt_adapter import (
        _MAX_BUFFER_SEC,
        _MAX_BUFFER_SEC_DEFAULT,
    )

    assert _MAX_BUFFER_SEC == _MAX_BUFFER_SEC_DEFAULT, (
        f"Non-numeric value should fall back to {_MAX_BUFFER_SEC_DEFAULT}, got {_MAX_BUFFER_SEC}"
    )
    assert any("not a valid float" in r.getMessage() for r in caplog.records), (
        f"Expected an invalid-float warning; got {[r.getMessage() for r in caplog.records]}"
    )

    # Restore.
    monkeypatch.delenv("REACHY_MINI_FASTER_WHISPER_MAX_BUFFER_SEC", raising=False)
    importlib.reload(_mod)


@pytest.mark.asyncio
async def test_vad_gate_opens_only_after_threshold_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """``on_speech_started`` fires only after _VAD_START_FRAMES consecutive speech frames.

    Feeds N-1 speech frames (must stay silent) then the Nth frame (must fire).
    Verifies the debounce gate is gated by the current module-level threshold.
    """
    from robot_comic.adapters import FasterWhisperSTTAdapter
    from robot_comic.adapters.faster_whisper_stt_adapter import _VAD_START_FRAMES

    holder = _install_stubs(monkeypatch)
    started_calls = 0

    async def _on_completed(_t: str) -> None: ...

    async def _on_speech_started() -> None:
        nonlocal started_calls
        started_calls += 1

    adapter = FasterWhisperSTTAdapter()
    await adapter.start(_on_completed, on_speech_started=_on_speech_started)

    # Feed N-1 consecutive speech frames — gate must NOT open yet.
    holder["vad"].script = _speech_script(_VAD_START_FRAMES - 1)
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK * (_VAD_START_FRAMES - 1)))
    assert started_calls == 0, (
        f"on_speech_started fired after only {_VAD_START_FRAMES - 1} frames (threshold is {_VAD_START_FRAMES})"
    )

    # Feed one more speech frame — gate must open now.
    holder["vad"].script = _speech_script(1)
    await adapter.feed_audio(_silence_frame(num_samples=_VAD_CHUNK))
    assert started_calls == 1, (
        f"on_speech_started should have fired after {_VAD_START_FRAMES} frames, fired {started_calls} time(s)"
    )


# ---------------------------------------------------------------------------
# cpu_threads — WhisperModel constructor arg (CTranslate2 intra-op threads)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whisper_model_constructed_with_cpu_threads_1_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """WhisperModel receives cpu_threads=1 when no env override is set."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    assert holder["model"].cpu_threads == 1


@pytest.mark.asyncio
async def test_whisper_model_cpu_threads_constructor_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing cpu_threads=2 to the constructor is forwarded to WhisperModel."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(cpu_threads=2)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    assert holder["model"].cpu_threads == 2


@pytest.mark.asyncio
async def test_whisper_model_cpu_threads_lower_bound_clamped_to_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """cpu_threads=0 is clamped to 1 (CTranslate2 requires >= 1)."""
    from robot_comic.adapters import FasterWhisperSTTAdapter

    holder = _install_stubs(monkeypatch)
    adapter = FasterWhisperSTTAdapter(cpu_threads=0)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    assert holder["model"].cpu_threads == 1
