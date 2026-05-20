"""Tests for the composable pipeline's synthetic startup greeting (#504).

The composable pipeline is input-driven only — without a kick, the persona
stays silent until the user speaks. ``ComposablePipeline.start_up()`` fires
a synthetic ``[conversation started]`` transcript through the LLM so the
persona produces its opener, matching the legacy handlers' behaviour.
Gated by ``REACHY_MINI_STARTUP_GREETING`` (default 1; set 0 for chassis-safe
boots).
"""

from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator

import pytest

from robot_comic.config import config as cfg
from robot_comic.config import refresh_runtime_config_from_env
from robot_comic.backends import AudioFrame, LLMResponse
from robot_comic.composable_pipeline import ComposablePipeline


# ---------------------------------------------------------------------------
# Programmable mock backends — same shape as test_composable_pipeline.py
# ---------------------------------------------------------------------------


class _ProgrammableSTT:
    def __init__(self) -> None:
        self.on_completed = None
        self.on_partial = None
        self.on_speech_started = None
        self.audio_frames: list[AudioFrame] = []
        self.stopped = False
        self.started = False

    async def start(
        self,
        on_completed,
        on_partial=None,
        on_speech_started=None,
    ) -> None:
        self.started = True
        self.on_completed = on_completed
        self.on_partial = on_partial
        self.on_speech_started = on_speech_started

    async def feed_audio(self, frame: AudioFrame) -> None:
        self.audio_frames.append(frame)

    async def stop(self) -> None:
        self.stopped = True


class _ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.prepared = False
        self.shutdown_called = False

    async def prepare(self) -> None:
        self.prepared = True

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise RuntimeError("scripted LLM exhausted")
        return self._responses.pop(0)

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _RecordingTTS:
    def __init__(self) -> None:
        self.prepared = False
        self.shutdown_called = False
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def prepare(self) -> None:
        self.prepared = True

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.calls.append((text, tags))
        if first_audio_marker is not None:
            import time as _time

            first_audio_marker.append(_time.monotonic())
        yield AudioFrame(samples=[0, 0, 0], sample_rate=24000)

    async def shutdown(self) -> None:
        self.shutdown_called = True


async def _wait_for_started(pipeline: ComposablePipeline) -> None:
    """Yield the loop until the pipeline has finished its prep + scheduled
    any startup task. The pipeline sets ``_started = True`` after preparing
    backends and binding the STT callback, just before scheduling the
    greeting task and entering ``_stop_event.wait()``.
    """
    for _ in range(50):
        await asyncio.sleep(0)
        if pipeline._started:
            return
    raise AssertionError("pipeline.start_up never reached the ready state")


async def _wait_for_llm_call(llm: _ScriptedLLM, timeout: float = 1.0) -> None:
    """Yield the loop until the LLM has been called at least once."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0)
        if llm.calls:
            return
    raise AssertionError(f"LLM was never called within {timeout}s")


# ---------------------------------------------------------------------------
# 1. Default-enabled: greeting fires the synthetic transcript through the LLM.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_greeting_enabled_by_default_fires_synthetic_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the env unset, default ``STARTUP_GREETING_ENABLED=True`` must
    cause ``start_up()`` to fire ``_on_transcript_completed`` with the
    ``[conversation started]`` payload, which then drives the LLM.
    """
    monkeypatch.delenv("REACHY_MINI_STARTUP_GREETING", raising=False)
    refresh_runtime_config_from_env()
    assert cfg.STARTUP_GREETING_ENABLED is True

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Welcome to the show!")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    try:
        await _wait_for_started(pipeline)
        await _wait_for_llm_call(llm, timeout=2.0)

        # The LLM saw exactly one chat call whose last history entry is the
        # synthetic ``[conversation started]`` transcript.
        assert len(llm.calls) == 1, f"expected exactly one LLM call, got {len(llm.calls)}"
        history = llm.calls[0]
        assert history[-1] == {"role": "user", "content": "[conversation started]"}
    finally:
        await pipeline.shutdown()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()


# ---------------------------------------------------------------------------
# 2. Env-disabled: greeting suppressed; LLM never called.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_greeting_disabled_skips_synthetic_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``REACHY_MINI_STARTUP_GREETING=0``, ``start_up()`` must NOT
    fire the synthetic transcript and the LLM must remain untouched.
    """
    monkeypatch.setenv("REACHY_MINI_STARTUP_GREETING", "0")
    refresh_runtime_config_from_env()
    assert cfg.STARTUP_GREETING_ENABLED is False

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="should-not-fire")])
    tts = _RecordingTTS()
    system_prompt = "You are a comedian."
    pipeline = ComposablePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        system_prompt=system_prompt,
    )

    task = asyncio.create_task(pipeline.start_up())
    try:
        await _wait_for_started(pipeline)
        # Give the loop plenty of ticks — any greeting task would land here.
        for _ in range(20):
            await asyncio.sleep(0)

        assert llm.calls == [], f"LLM should never be called when greeting is disabled; got {llm.calls!r}"
        # Conversation history holds only the original system prompt.
        assert pipeline.conversation_history == [{"role": "system", "content": system_prompt}]
    finally:
        # Restore default so other tests aren't affected by ordering.
        monkeypatch.delenv("REACHY_MINI_STARTUP_GREETING", raising=False)
        refresh_runtime_config_from_env()
        await pipeline.shutdown()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()


# ---------------------------------------------------------------------------
# 3. Enabled path drives the TTS chain and produces audio frames downstream.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_greeting_produces_tts_audio_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthetic transcript should drive the LLM → TTS chain and emit
    at least one audio frame onto the output queue, exactly like a normal
    user-turn does.
    """
    monkeypatch.delenv("REACHY_MINI_STARTUP_GREETING", raising=False)
    refresh_runtime_config_from_env()
    assert cfg.STARTUP_GREETING_ENABLED is True

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hello world from the persona.")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    try:
        await _wait_for_started(pipeline)

        # The pipeline pushes frames as ``(sample_rate, samples)`` tuples
        # (see test_single_text_turn_produces_one_frame for the contract).
        # We just need to see at least one audio-ish item materialise.
        deadline = asyncio.get_event_loop().time() + 3.0
        audio_item: Any = None
        while asyncio.get_event_loop().time() < deadline:
            try:
                item = await asyncio.wait_for(pipeline.output_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if isinstance(item, tuple):
                audio_item = item
                break

        assert audio_item is not None, "Startup greeting should have produced a TTS audio frame on the output queue"
        sample_rate, _samples = audio_item
        assert sample_rate == 24000
        assert tts.calls and tts.calls[0][0] == "Hello world from the persona."
    finally:
        await pipeline.shutdown()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()


# ---------------------------------------------------------------------------
# 4. refresh_runtime_config_from_env keeps STARTUP_GREETING_ENABLED in sync.
# ---------------------------------------------------------------------------


def test_startup_greeting_env_refresh_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """Toggling ``REACHY_MINI_STARTUP_GREETING`` after import and calling
    ``refresh_runtime_config_from_env()`` must update the live attribute.
    """
    monkeypatch.setenv("REACHY_MINI_STARTUP_GREETING", "1")
    refresh_runtime_config_from_env()
    assert cfg.STARTUP_GREETING_ENABLED is True

    monkeypatch.setenv("REACHY_MINI_STARTUP_GREETING", "0")
    refresh_runtime_config_from_env()
    assert cfg.STARTUP_GREETING_ENABLED is False

    monkeypatch.setenv("REACHY_MINI_STARTUP_GREETING", "1")
    refresh_runtime_config_from_env()
    assert cfg.STARTUP_GREETING_ENABLED is True

    # Cleanup: leave the env in the same state as a fresh import.
    monkeypatch.delenv("REACHY_MINI_STARTUP_GREETING", raising=False)
    refresh_runtime_config_from_env()
