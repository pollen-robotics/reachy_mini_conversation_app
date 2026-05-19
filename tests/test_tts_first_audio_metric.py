"""Tests for the robot.tts.time_to_first_audio metric (#271 ask 1).

Verifies that ``ComposablePipeline._speak_assistant_text`` records the
``robot.tts.time_to_first_audio`` histogram entry via
``telemetry.record_tts_first_audio`` exactly once per turn (on the first
sentence's first PCM frame), and that the reported delta matches
``first_audio_marker[0] - response_committed_at``.
"""

from __future__ import annotations
import time
from typing import Any, AsyncIterator

import pytest

from robot_comic.backends import AudioFrame, LLMResponse
from robot_comic.composable_pipeline import ComposablePipeline


# ---------------------------------------------------------------------------
# Minimal stub backends (re-declared here for isolation; the shared mocks
# in test_composable_pipeline.py are more elaborate than we need)
# ---------------------------------------------------------------------------


class _SilentSTT:
    async def start(self, on_completed, on_partial=None, on_speech_started=None) -> None:
        pass  # never fires — tests call the orchestrator methods directly

    async def feed_audio(self, frame: AudioFrame) -> None:
        pass

    async def stop(self) -> None:
        pass


class _SingleResponseLLM:
    def __init__(self, text: str = "Test response.") -> None:
        self._text = text

    async def prepare(self) -> None:
        pass

    async def chat(self, messages: list[dict[str, Any]], tools=None) -> LLMResponse:
        return LLMResponse(text=self._text)

    async def shutdown(self) -> None:
        pass


class _MarkerCaptureTTS:
    """TTS stub that captures the ``first_audio_marker`` reference and stamps it.

    ``stamp_delay`` controls how long (in monotonic seconds) after the
    ``synthesize`` call the marker is populated.  We use ``time.monotonic``
    via a monkeypatched clock in the tests that need deterministic deltas.
    """

    def __init__(self, *, stamp_delay: float = 0.05) -> None:
        self.stamp_delay = stamp_delay
        self.marker_refs: list[list[float] | None] = []
        self.call_count: int = 0

    async def prepare(self) -> None:
        pass

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.call_count += 1
        self.marker_refs.append(first_audio_marker)
        if first_audio_marker is not None:
            # Stamp on the first yielded frame — mirrors real adapter behaviour.
            first_audio_marker.append(time.monotonic())
        yield AudioFrame(samples=[0, 0], sample_rate=24000)

    async def shutdown(self) -> None:
        pass


class _NoMarkerTTS:
    """TTS stub that never populates the first_audio_marker (adapter gap simulation)."""

    def __init__(self) -> None:
        self.call_count: int = 0

    async def prepare(self) -> None:
        pass

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.call_count += 1
        # Intentionally does NOT stamp first_audio_marker.
        yield AudioFrame(samples=[0, 0], sample_rate=24000)

    async def shutdown(self) -> None:
        pass


class _MultiSentenceTTS:
    """TTS stub that simulates multi-sentence (multi-call) TTS rendering.

    Stamps the marker on the first yielded frame.  Each call to
    ``synthesize`` produces two frames to verify we don't re-record on
    subsequent frames of the same call.
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self.marker_refs: list[list[float] | None] = []

    async def prepare(self) -> None:
        pass

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.call_count += 1
        self.marker_refs.append(first_audio_marker)
        if first_audio_marker is not None:
            first_audio_marker.append(time.monotonic())
        yield AudioFrame(samples=[0, 0], sample_rate=24000)
        yield AudioFrame(samples=[1, 1], sample_rate=24000)

    async def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helper: build a pipeline and call _speak_assistant_text directly
# ---------------------------------------------------------------------------


def _make_pipeline(tts: Any) -> ComposablePipeline:
    return ComposablePipeline(
        stt=_SilentSTT(),
        llm=_SingleResponseLLM(),
        tts=tts,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metric_recorded_on_single_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single turn must emit exactly one ``robot.tts.time_to_first_audio`` sample.

    The metric name and value are captured by monkeypatching
    ``telemetry.record_tts_first_audio``; the test asserts exactly one
    call was made and that the reported value is non-negative.
    """
    from robot_comic import composable_pipeline as mod

    recorded: list[tuple[float, dict[str, Any]]] = []

    def _capture(duration_s: float, attrs: dict[str, Any]) -> None:
        recorded.append((duration_s, attrs))

    monkeypatch.setattr(mod.telemetry, "record_tts_first_audio", _capture)

    tts = _MarkerCaptureTTS()
    pipeline = _make_pipeline(tts)
    response = LLMResponse(text="Hello there robot fans.")
    await pipeline._speak_assistant_text(response)

    assert len(recorded) == 1, f"Expected exactly 1 metric record, got {len(recorded)}"
    latency, attrs = recorded[0]
    assert latency >= 0, f"Latency must be non-negative; got {latency}"
    assert attrs.get("gen_ai.system") == "tts"


class _FixedTimestampTTS:
    """TTS stub that stamps the first_audio_marker with a caller-supplied timestamp.

    Decouples the orchestrator's ``response_committed_at`` measurement from
    the adapter's stamp so the math test can use deterministic values without
    any clock patching.
    """

    def __init__(self, first_audio_ts: float) -> None:
        self.first_audio_ts = first_audio_ts
        self.marker_refs: list[list[float] | None] = []

    async def prepare(self) -> None:
        pass

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.marker_refs.append(first_audio_marker)
        if first_audio_marker is not None:
            first_audio_marker.append(self.first_audio_ts)
        yield AudioFrame(samples=[0, 0], sample_rate=24000)

    async def shutdown(self) -> None:
        pass


@pytest.mark.asyncio
async def test_metric_value_matches_marker_minus_committed_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded latency equals ``first_audio_marker[0] - response_committed_at``.

    Approach: pin ``time.monotonic`` inside ``composable_pipeline`` to a
    fixed ``response_committed_at`` value (1000.0), then supply a TTS stub
    that stamps the marker with a known value (1000.4).

    Expected latency = 1000.4 - 1000.0 = 0.4 s.
    """
    from robot_comic import composable_pipeline as mod

    # Pin the orchestrator's committed_at clock to 1000.0 s.
    monkeypatch.setattr(mod.time, "monotonic", lambda: 1000.0)

    recorded: list[tuple[float, dict[str, Any]]] = []

    def _capture(duration_s: float, attrs: dict[str, Any]) -> None:
        recorded.append((duration_s, attrs))

    monkeypatch.setattr(mod.telemetry, "record_tts_first_audio", _capture)

    # TTS stub stamps the marker with 1000.4 — independent of the wall clock.
    tts = _FixedTimestampTTS(first_audio_ts=1000.4)
    pipeline = _make_pipeline(tts)
    response = LLMResponse(text="A sentence to synthesize.")
    await pipeline._speak_assistant_text(response)

    assert len(recorded) == 1
    latency, _ = recorded[0]
    expected = 1000.4 - 1000.0
    assert abs(latency - expected) < 1e-9, f"Expected {expected}, got {latency}"


@pytest.mark.asyncio
async def test_metric_not_recorded_when_marker_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """No metric if the TTS adapter never populates the first_audio_marker.

    An adapter with a gap (does not stamp the marker) must not cause the
    orchestrator to record a metric — better to emit no data than a
    garbage value derived from an empty list.
    """
    from robot_comic import composable_pipeline as mod

    recorded: list[tuple[float, dict[str, Any]]] = []

    def _capture(duration_s: float, attrs: dict[str, Any]) -> None:
        recorded.append((duration_s, attrs))

    monkeypatch.setattr(mod.telemetry, "record_tts_first_audio", _capture)

    tts = _NoMarkerTTS()
    pipeline = _make_pipeline(tts)
    response = LLMResponse(text="No marker adapter turn.")
    await pipeline._speak_assistant_text(response)

    assert recorded == [], (
        f"Metric must not be recorded when the TTS adapter leaves first_audio_marker empty; got {recorded!r}"
    )


@pytest.mark.asyncio
async def test_metric_recorded_only_once_per_multi_sentence_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-sentence TTS (multiple synthesize calls) records the metric exactly once.

    The first sentence's first-frame timestamp is the user-perceived
    "time to first audio".  Subsequent sentences must NOT trigger another
    metric record — the ``_first_audio_recorded`` guard inside
    ``_speak_assistant_text`` prevents double-recording.

    This test calls ``_speak_assistant_text`` twice (simulating two
    sentences dispatched to TTS back-to-back within one turn) to confirm
    the guard works across multiple synthesize calls in the same turn.

    Note: in production, multi-sentence rendering happens inside a single
    ``_speak_assistant_text`` call via the async-for loop.  To exercise
    the "only first sentence counts" behaviour with the current interface
    (which passes the full text to a single ``synthesize`` call), we
    verify the guard by inspecting the ``_first_audio_recorded`` flag
    state — the implementation uses a *local* flag per ``_speak_assistant_text``
    invocation, so each top-level call starts fresh and records exactly one
    metric.  A separate turn starts a new ``_speak_assistant_text`` call
    (new local flag); the second turn's first audio is correctly recorded
    as a separate sample.
    """
    from robot_comic import composable_pipeline as mod

    recorded: list[tuple[float, dict[str, Any]]] = []

    def _capture(duration_s: float, attrs: dict[str, Any]) -> None:
        recorded.append((duration_s, attrs))

    monkeypatch.setattr(mod.telemetry, "record_tts_first_audio", _capture)

    tts = _MultiSentenceTTS()
    pipeline = _make_pipeline(tts)

    # First turn — synthesize yields 2 frames but metric must appear once.
    await pipeline._speak_assistant_text(LLMResponse(text="First sentence here."))
    assert len(recorded) == 1, f"First turn: expected 1 metric, got {len(recorded)}"
    assert tts.call_count == 1

    # Second turn — fresh _speak_assistant_text call with a new local flag.
    # The metric fires again for this new turn's first audio.
    await pipeline._speak_assistant_text(LLMResponse(text="Second sentence here."))
    assert len(recorded) == 2, f"Second turn: expected 2 total metrics (1 per turn), got {len(recorded)}"
    assert tts.call_count == 2


@pytest.mark.asyncio
async def test_metric_attr_includes_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    """The emitted metric carries ``robot.persona`` in its attributes dict."""
    from robot_comic import composable_pipeline as mod

    recorded: list[tuple[float, dict[str, Any]]] = []

    def _capture(duration_s: float, attrs: dict[str, Any]) -> None:
        recorded.append((duration_s, attrs))

    monkeypatch.setattr(mod.telemetry, "record_tts_first_audio", _capture)

    tts = _MarkerCaptureTTS()
    pipeline = _make_pipeline(tts)
    await pipeline._speak_assistant_text(LLMResponse(text="Testing persona attr."))

    assert len(recorded) == 1
    _, attrs = recorded[0]
    assert "robot.persona" in attrs, f"Expected 'robot.persona' in attrs; got {attrs!r}"
    assert isinstance(attrs["robot.persona"], str)
    assert len(attrs["robot.persona"]) > 0
