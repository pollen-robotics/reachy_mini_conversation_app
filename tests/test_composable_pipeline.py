"""Tests for ComposablePipeline (Phase 2 of pipeline refactor)."""

from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator

import pytest

from robot_comic.backends import ToolCall, AudioFrame, LLMResponse
from robot_comic.composable_pipeline import ComposablePipeline


# ---------------------------------------------------------------------------
# Programmable mock backends
# ---------------------------------------------------------------------------


class _ProgrammableSTT:
    """STT mock that holds the bound callbacks so tests can fire events."""

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
    """LLM mock that returns a sequence of pre-built LLMResponses."""

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
        # Snapshot the messages as the orchestrator saw them this round.
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise RuntimeError("scripted LLM exhausted")
        return self._responses.pop(0)

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _RecordingTTS:
    """TTS mock that yields one frame and records the text/tags it saw."""

    def __init__(self) -> None:
        self.prepared = False
        self.shutdown_called = False
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.marker_refs: list[list[float] | None] = []

    async def prepare(self) -> None:
        self.prepared = True

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.calls.append((text, tags))
        self.marker_refs.append(first_audio_marker)
        # Populate the marker like a real adapter would so the orchestrator
        # has a stamp to read out, in case future test cases want it.
        if first_audio_marker is not None:
            import time as _time

            first_audio_marker.append(_time.monotonic())
        yield AudioFrame(samples=[0, 0, 0], sample_rate=24000)

    async def shutdown(self) -> None:
        self.shutdown_called = True


async def _wait_for_callback(stt: _ProgrammableSTT) -> None:
    """Yield the loop until the pipeline binds the STT callback."""
    for _ in range(20):
        await asyncio.sleep(0)
        if stt.on_completed is not None:
            return
    raise AssertionError("STT.on_completed was never bound")


# ---------------------------------------------------------------------------
# Single-turn happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_text_turn_produces_one_frame() -> None:
    """One transcript → one LLM call → one TTS synth → one frame on the output queue."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hello there!")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)
    assert stt.started and llm.prepared and tts.prepared

    await stt.on_completed("hi robot")

    # Pipeline pushes the legacy ``(sample_rate, samples)`` tuple shape
    # so the downstream FastRTC playback consumer in ``console.py``
    # (``isinstance(handler_output, tuple)`` branch at :1466) picks it up.
    # Pre-hotfix the orchestrator was pushing the ``AudioFrame`` dataclass
    # directly and audio was silently dropped on hardware.
    item = await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    assert isinstance(item, tuple)
    sample_rate, samples = item
    assert sample_rate == 24000
    assert tts.calls == [("Hello there!", ())]
    assert pipeline.conversation_history == [
        {"role": "user", "content": "hi robot"},
        {"role": "assistant", "content": "Hello there!"},
    ]

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Tool-call loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_calls_dispatched_then_followup_text_spoken() -> None:
    """LLM requests a tool, orchestrator dispatches it, follow-up text is spoken."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(tool_calls=(ToolCall(id="t-1", name="dance", args={"name": "happy"}),)),
            LLMResponse(text="Done dancing!"),
        ]
    )
    tts = _RecordingTTS()

    dispatched: list[ToolCall] = []

    async def _dispatch(call: ToolCall) -> str:
        dispatched.append(call)
        return f"ok:{call.name}"

    pipeline = ComposablePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        tool_dispatcher=_dispatch,
        tools_spec=[{"name": "dance"}],
    )

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("dance for me")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(dispatched) == 1
    assert dispatched[0].name == "dance"
    assert dispatched[0].args == {"name": "happy"}
    assert len(llm.calls) == 2

    assert pipeline.conversation_history[0] == {"role": "user", "content": "dance for me"}
    assert pipeline.conversation_history[1]["role"] == "tool"
    assert pipeline.conversation_history[1]["tool_call_id"] == "t-1"
    assert pipeline.conversation_history[1]["content"] == "ok:dance"
    assert pipeline.conversation_history[2] == {"role": "assistant", "content": "Done dancing!"}

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_tool_call_without_dispatcher_short_circuits() -> None:
    """When no tool_dispatcher is configured, tool_calls are ignored gracefully."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(tool_calls=(ToolCall(id="t-1", name="x", args={}),))])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("do a thing")
    for _ in range(5):
        await asyncio.sleep(0)

    assert tts.calls == []
    assert pipeline.output_queue.empty()

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_tool_loop_respects_max_rounds() -> None:
    """An LLM that endlessly requests tools is bounded by max_tool_rounds."""
    stt = _ProgrammableSTT()
    looper = LLMResponse(tool_calls=(ToolCall(id="t-1", name="loop", args={}),))
    llm = _ScriptedLLM([looper] * 50)
    tts = _RecordingTTS()

    async def _dispatch(call: ToolCall) -> str:
        return "ok"

    pipeline = ComposablePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        tool_dispatcher=_dispatch,
        max_tool_rounds=3,
    )

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("loop please")
    for _ in range(10):
        await asyncio.sleep(0)

    assert len(llm.calls) == 3
    assert tts.calls == []

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_releases_start_up_and_closes_backends() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await pipeline.shutdown()
    await asyncio.wait_for(task, timeout=1.0)
    assert stt.stopped and llm.shutdown_called and tts.shutdown_called


@pytest.mark.asyncio
async def test_shutdown_before_start_up_is_safe() -> None:
    """Calling shutdown() before start_up() must not deadlock or crash."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    await pipeline.shutdown()
    assert not stt.stopped
    # Subsequent start_up returns immediately because stop_event is set,
    # AND it must not have prepared any backends — those would never get
    # torn down (shutdown's "not started" early-return already ran).
    await asyncio.wait_for(pipeline.start_up(), timeout=1.0)
    assert not stt.started
    assert not llm.prepared
    assert not tts.prepared


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_seeded_into_history() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="hi")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="You are Bill Hicks.")

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hello there")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert pipeline.conversation_history[0] == {"role": "system", "content": "You are Bill Hicks."}
    assert llm.calls[0][0] == {"role": "system", "content": "You are Bill Hicks."}
    assert llm.calls[0][1] == {"role": "user", "content": "hello there"}

    await pipeline.shutdown()
    await task


def test_reset_history_keeps_system_prompt_by_default() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="persona")
    pipeline.conversation_history.append({"role": "user", "content": "x"})
    pipeline.conversation_history.append({"role": "assistant", "content": "y"})
    pipeline.reset_history()
    assert pipeline.conversation_history == [{"role": "system", "content": "persona"}]


def test_reset_history_can_drop_system_prompt() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="persona")
    pipeline.conversation_history.append({"role": "user", "content": "x"})
    pipeline.reset_history(keep_system=False)
    assert pipeline.conversation_history == []


# ---------------------------------------------------------------------------
# Audio input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_audio_delegates_to_stt() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    frame = AudioFrame(samples=[1, 2], sample_rate=16000)
    await pipeline.feed_audio(frame)
    assert stt.audio_frames == [frame]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_transcript_is_ignored() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])  # would raise if called
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("")
    for _ in range(5):
        await asyncio.sleep(0)

    assert llm.calls == []
    assert pipeline.conversation_history == []

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_empty_assistant_text_substitutes_canned_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty-text LLMResponse triggers the #267 canned-line fallback.

    Pre-#267 behaviour was a single ``logger.warning`` followed by a
    dropped turn — the operator perceived the robot as silently hung
    whenever Gemini emitted ``finish_reason=STOP`` with zero parts (the
    short-input + tools-configured + comedian-style-prompt case).
    The new behaviour speaks a persona-appropriate canned line and
    records it in conversation_history so the LLM keeps continuity.
    """
    from robot_comic import composable_pipeline as mod

    # Pin the profile-dir resolver so the default-pool path is exercised
    # regardless of the test process's ambient REACHY_MINI_CUSTOM_PROFILE.
    monkeypatch.setattr(mod, "_resolve_empty_stop_profile_dir", lambda: None)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="   ")])  # whitespace-only
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi robot")
    item = await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    assert isinstance(item, tuple)  # one TTS frame emitted

    # The canned line is sampled from the default pool — check we got
    # one of those entries verbatim, both in TTS and in history.
    from robot_comic.empty_stop_fallbacks import get_default_pool

    assert len(tts.calls) == 1
    spoken_text, _tags = tts.calls[0]
    assert spoken_text in get_default_pool()
    assert pipeline.conversation_history == [
        {"role": "user", "content": "hi robot"},
        {"role": "assistant", "content": spoken_text},
    ]

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Phase 5f.2 — minimum-utterance filter (self-echo cascade defence)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_word_transcript_dropped_as_likely_echo() -> None:
    """Single-word transcripts (``"You"``) must not reach the LLM.

    Hardware finding 2026-05-16: faster-whisper transcribes speaker
    echo as short stock fragments (`"You"`, `"Thank you"`). Dropping
    them at the orchestrator boundary breaks the cascade.
    """
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])  # would raise if called
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("You")
    for _ in range(5):
        await asyncio.sleep(0)

    assert llm.calls == []
    assert pipeline.conversation_history == []
    assert pipeline.output_queue.empty()

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_short_char_transcript_dropped_as_likely_echo() -> None:
    """Two-character transcripts (``"hi"``) fail both filter arms and are dropped."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi")
    for _ in range(5):
        await asyncio.sleep(0)

    assert llm.calls == []
    assert pipeline.conversation_history == []

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_two_word_transcript_below_char_floor_dropped() -> None:
    """Two-word but ≤7-char transcripts (``"go on"``) drop via the char floor.

    Documents the deliberate trade-off: legitimate ultra-short prompts
    fail the filter. Operator-facing rationale lives in the 5f.2 spec.
    """
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("go on")  # 2 words, 5 chars — fails char floor
    for _ in range(5):
        await asyncio.sleep(0)

    assert llm.calls == []
    assert pipeline.conversation_history == []

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_normal_transcript_passes_filter() -> None:
    """Regression guard: ordinary user input clears both filter arms."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hello!")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi robot can you dance")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert llm.calls != []
    assert pipeline.conversation_history[0] == {
        "role": "user",
        "content": "hi robot can you dance",
    }

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_short_transcript_drop_logged_at_debug(caplog) -> None:
    """The short-drop path emits a DEBUG log for operator visibility."""
    import logging

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    with caplog.at_level(logging.DEBUG, logger="robot_comic.composable_pipeline"):
        await stt.on_completed("You")
        for _ in range(5):
            await asyncio.sleep(0)

    assert any("short transcript" in rec.getMessage() for rec in caplog.records)

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_short_drop_runs_before_duplicate_suppression() -> None:
    """The short-drop must precede duplicate-suppression cache writes.

    Proves ordering: the duplicate-suppression cache never sees a
    dropped value, so the same short fragment can be sent repeatedly
    and each is still rejected via the short-drop path (not the
    duplicate path) — i.e. the LLM still never runs.
    """
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("You")
    await stt.on_completed("You")
    for _ in range(5):
        await asyncio.sleep(0)

    assert llm.calls == []
    assert pipeline._last_completed_transcript == ""

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Phase 5f.3 — content-similarity echo filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_echo_of_assistant_text_dropped() -> None:
    """An incoming transcript identical to the last assistant utterance is dropped.

    Hardware finding 2026-05-16 (post-5f.2): multi-sentence echoes
    (e.g. "Okay, well I didn't vote for you.") clear 5f.2's length
    filter and still cascade. The similarity filter is the third line
    of defence.
    """
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hello there folks how are you")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    # First, a real user turn that elicits the assistant utterance.
    await stt.on_completed("hi robot can you greet")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    assert len(llm.calls) == 1

    # Now STT emits the assistant utterance back (acoustic self-echo).
    await stt.on_completed("Hello there folks how are you")
    for _ in range(5):
        await asyncio.sleep(0)

    # No second LLM call; history is unchanged by the echo.
    assert len(llm.calls) == 1
    assert pipeline.conversation_history[-1] == {
        "role": "assistant",
        "content": "Hello there folks how are you",
    }

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_minor_word_error_echo_dropped() -> None:
    """Echo with case folding + dropped punctuation + minor word errors is dropped.

    faster-whisper's transcription of its own speech often differs only
    in case + punctuation + contraction expansion. SequenceMatcher.ratio
    on normalized text should still clear the 0.65 threshold.
    """
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Okay, well I didn't vote for you.")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("who did you vote for")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    assert len(llm.calls) == 1

    # Echo with no punctuation, lowercase, contraction expanded.
    await stt.on_completed("okay well i did not vote for you")
    for _ in range(5):
        await asyncio.sleep(0)

    assert len(llm.calls) == 1  # echo dropped; LLM not re-invoked

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_paraphrase_not_dropped() -> None:
    """Paraphrase / response (low similarity) clears the filter."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(text="Okay, well I didn't vote for you."),
            LLMResponse(text="Sure, who did you go for?"),
        ]
    )
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("who did you vote for")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    # A genuine paraphrase response from the user — shares "voted for"
    # but is otherwise dissimilar.
    await stt.on_completed("yeah well I voted for the other guy")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(llm.calls) == 2  # paraphrase dispatched

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_echo_against_older_assistant_turn() -> None:
    """The ring buffer covers more than just the most-recent turn."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(text="First utterance about something specific"),
            LLMResponse(text="Second utterance about another topic"),
            LLMResponse(text="Third utterance about yet more things"),
        ]
    )
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    for prompt in ("tell me a thing", "another one", "and one more"):
        await stt.on_completed(prompt)
        await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(llm.calls) == 3

    # Echo of the FIRST assistant utterance (still within maxlen=5).
    await stt.on_completed("First utterance about something specific")
    for _ in range(5):
        await asyncio.sleep(0)

    assert len(llm.calls) == 3  # dropped

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_evicted_assistant_text_no_longer_blocks() -> None:
    """Once an assistant text falls out of the ring buffer it no longer matches."""
    from robot_comic import composable_pipeline as mod

    maxlen = mod.ECHO_HISTORY_MAXLEN

    # Each assistant line must be lexically distinct so that evicting
    # the first one actually frees the similarity check (otherwise the
    # remaining buffer entries would still match a paraphrase of #0).
    distinct_phrases = [
        "Aardvarks waddle through dense underbrush quietly",
        "Bicycles squeak when their chains run dry",
        "Cocoa beans ferment in wooden trays slowly",
        "Doppler radar reveals oncoming storm patterns clearly",
        "Eagles soar over alpine ridges silently",
        "Felonious badgers raid forgotten garden plots",
        "Glaciers carve fjords across millennia patiently",
        "Hummingbirds visit fuchsia blooms each morning",
    ]
    assistant_lines = distinct_phrases[: maxlen + 1]
    final_response = LLMResponse(text="acknowledged")
    llm = _ScriptedLLM([LLMResponse(text=line) for line in assistant_lines] + [final_response])
    stt = _ProgrammableSTT()
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    # Use highly-distinct prompt strings so the post-#425 dedup (similarity
    # >= DEDUP_SIMILARITY_THRESHOLD) doesn't merge consecutive turns. "prompt
    # number 0" / "prompt number 1" are ~87% similar by SequenceMatcher ratio,
    # which is above the 0.85 threshold — would dedupe even with distinct
    # echo-history slots, defeating the test's intent.
    distinct_prompts = [
        "alpha rhinoceros question",
        "beta whale opinion",
        "gamma toaster perspective",
        "delta marsupial proposal",
        "epsilon volcano analysis",
        "zeta xylophone hypothesis",
        "eta meteor recommendation",
    ]
    assert len(distinct_prompts) >= maxlen + 1
    for i in range(maxlen + 1):
        await stt.on_completed(distinct_prompts[i])
        await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(llm.calls) == maxlen + 1

    # The very first assistant utterance should have been evicted.
    await stt.on_completed(assistant_lines[0])
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(llm.calls) == maxlen + 2  # dispatched, not dropped

    await pipeline.shutdown()
    await task


def test_normalize_for_echo_check_strips_punctuation_and_case() -> None:
    """The normalizer must lowercase + strip punctuation + collapse whitespace."""
    from robot_comic.composable_pipeline import _normalize_for_echo_check

    # Apostrophe inside a word becomes a space, which is fine — both
    # sides of the similarity comparison get the same treatment so the
    # ratio remains high.
    assert _normalize_for_echo_check("Okay, well I didn't vote for you!") == "okay well i didn t vote for you"
    assert _normalize_for_echo_check("  Multiple   spaces\there\n") == "multiple spaces here"
    assert _normalize_for_echo_check("") == ""
    assert _normalize_for_echo_check("...") == ""


@pytest.mark.asyncio
async def test_similarity_filter_runs_after_length_filter(caplog) -> None:
    """A short transcript that also matches an assistant turn drops via the length filter.

    Proves ordering: the length filter (5f.2) runs before the similarity
    filter (5f.3), so a fragment that would trip both is rejected by
    the cheaper / more specific structural check first.
    """
    import logging

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="ok")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    # Get "ok" into the assistant-text ring buffer via a real user prompt.
    await stt.on_completed("are you ready")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    with caplog.at_level(logging.DEBUG, logger="robot_comic.composable_pipeline"):
        caplog.clear()
        await stt.on_completed("ok")
        for _ in range(5):
            await asyncio.sleep(0)

    messages = [r.getMessage() for r in caplog.records]
    assert any("short transcript" in m for m in messages)
    assert not any("likely echo of own speech" in m for m in messages)

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_similarity_filter_runs_before_duplicate_suppression() -> None:
    """The similarity drop must NOT poison the duplicate-suppression cache.

    A repeated echo should keep hitting the similarity arm — never the
    exact-match dup window — so ``_last_completed_transcript`` stays
    pinned to the last *real* user turn.
    """
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Alright alright alright that's enough")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("say something")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    assert pipeline._last_completed_transcript == "say something"

    # Two echoes in a row.
    await stt.on_completed("Alright alright alright that's enough")
    await stt.on_completed("Alright alright alright that's enough")
    for _ in range(5):
        await asyncio.sleep(0)

    # Dup-window state is untouched: still holds the real user turn.
    assert pipeline._last_completed_transcript == "say something"
    assert len(llm.calls) == 1  # neither echo dispatched

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Lifecycle Hook #4 — record_joke_history (#337)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_joke_history_called_for_non_empty_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must invoke ``record_joke_history`` after a final speak round.

    Legacy parity: ``llama_base.py:578-594`` and ``gemini_tts.py:380-394``
    capture the punchline + topic after the LLM produces the final
    assistant text. The composable path bypasses both legacy sites; the
    orchestrator must call ``record_joke_history`` itself.
    """
    from robot_comic import composable_pipeline as mod

    captured: list[str] = []

    async def _recorder(text: str) -> None:
        captured.append(text)

    monkeypatch.setattr(mod, "record_joke_history", _recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="That's the joke!")])
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("tell me one")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert captured == ["That's the joke!"]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_record_joke_history_not_called_for_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The #267 empty-STOP canned fallback must not pollute joke history.

    Pre-#267 the empty-text path simply dropped the turn, so joke-history
    capture obviously didn't fire. With the fallback path active the
    orchestrator does speak something — but the canned filler is not a
    comedy punchline, so it stays out of the cross-persona joke history.
    """
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "_resolve_empty_stop_profile_dir", lambda: None)

    captured: list[str] = []

    async def _recorder(text: str) -> None:
        captured.append(text)

    monkeypatch.setattr(mod, "record_joke_history", _recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="   ")])  # whitespace-only
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi robot")
    # Drain the queued canned-line audio so the pipeline can shut down.
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert captured == []

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_record_joke_history_not_called_on_tool_only_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-call-only LLM rounds must not record joke history.

    Legacy ``_run_turn`` captures after the assistant text is final, NOT on
    intermediate tool rounds. The orchestrator hook must respect that —
    fire exactly once per turn, on the final speak round.
    """
    from robot_comic import composable_pipeline as mod

    captured: list[str] = []

    async def _recorder(text: str) -> None:
        captured.append(text)

    monkeypatch.setattr(mod, "record_joke_history", _recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(tool_calls=(ToolCall(id="t-1", name="dance", args={"name": "happy"}),)),
            LLMResponse(text="Done dancing!"),
        ]
    )
    tts = _RecordingTTS()

    async def _dispatch(call: ToolCall) -> str:
        return f"ok:{call.name}"

    pipeline = mod.ComposablePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        tool_dispatcher=_dispatch,
        tools_spec=[{"name": "dance"}],
    )

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("dance for me")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    # Exactly one capture, with the final assistant text — not the tool round.
    assert captured == ["Done dancing!"]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_record_joke_history_exception_does_not_crash_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If record_joke_history somehow raises, the TTS still runs.

    The helper itself swallows exceptions, but if a future change ever
    leaks one, the orchestrator must not let it prevent speech.
    """
    from robot_comic import composable_pipeline as mod

    async def _bad_recorder(text: str) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(mod, "record_joke_history", _bad_recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hello!")])
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi there")
    # Yield several loop iterations so the turn either completes or fails.
    for _ in range(20):
        await asyncio.sleep(0)
        if not tts.calls == []:
            break

    # TTS must have run even though the hook raised.
    assert tts.calls == [("Hello!", ())]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_tool_dispatch_exception_records_error_in_history() -> None:
    """A tool that raises gets a stringified-error result back to the LLM."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(tool_calls=(ToolCall(id="t-1", name="bad", args={}),)),
            LLMResponse(text="I see the tool failed; carrying on."),
        ]
    )
    tts = _RecordingTTS()

    async def _dispatch(call: ToolCall) -> str:
        raise ValueError("oops")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, tool_dispatcher=_dispatch)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("try the bad tool")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    tool_msg = pipeline.conversation_history[1]
    assert tool_msg["role"] == "tool"
    assert "tool error" in tool_msg["content"]
    assert "ValueError" in tool_msg["content"]

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Lifecycle Hook #5 — trim_history_in_place (#337)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trim_history_called_once_per_user_turn_before_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must invoke ``trim_history_in_place`` once per user turn.

    Legacy parity: all three legacy ``_dispatch_completed_transcript``
    sites (``llama_base.py:506``, ``gemini_tts.py:365``,
    ``elevenlabs_tts.py:565``) trim once per user turn before the LLM
    loop. The composable path bypasses every one of those — the
    orchestrator must trim itself.
    """
    from robot_comic import composable_pipeline as mod

    call_log: list[str] = []

    def _trim_recorder(history: list[dict[str, Any]], **kwargs: Any) -> int:
        call_log.append("trim")
        return 0

    monkeypatch.setattr(mod, "trim_history_in_place", _trim_recorder)

    class _LoggingLLM(_ScriptedLLM):
        def __init__(self, responses: list[LLMResponse]) -> None:
            super().__init__(responses)

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
        ) -> LLMResponse:
            call_log.append("chat")
            return await super().chat(messages, tools)

    stt = _ProgrammableSTT()
    llm = _LoggingLLM([LLMResponse(text="Hello!")])
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi robot")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    # Exactly one trim, fired before the LLM chat() call.
    assert call_log == ["trim", "chat"]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_trim_history_uses_orchestrator_history_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trim must operate on the orchestrator's own history list, by identity.

    A passed-in copy would mean the cap doesn't shrink subsequent
    requests. Identity check guards against an accidental ``list(...)``
    or slice at the call site.
    """
    from robot_comic import composable_pipeline as mod

    seen: list[list[dict[str, Any]]] = []

    def _trim_recorder(history: list[dict[str, Any]], **kwargs: Any) -> int:
        seen.append(history)
        return 0

    monkeypatch.setattr(mod, "trim_history_in_place", _trim_recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="ok")])
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("test prompt")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(seen) == 1
    # Identity, not equality — the trim must mutate the orchestrator's list
    # in place. After the LLM call the orchestrator appends the assistant
    # turn, so equality wouldn't hold anyway, but ``is`` proves the contract.
    assert seen[0] is pipeline._conversation_history

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_trim_history_called_once_per_turn_not_per_tool_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-call rounds inside a single user turn must NOT re-trim.

    Legacy ``_dispatch_completed_transcript`` trims once before the LLM
    loop and never re-trims inside it. The orchestrator must respect
    that cadence — fire exactly once per user turn even when several
    LLM rounds run.
    """
    from robot_comic import composable_pipeline as mod

    trim_calls: list[int] = []

    def _trim_recorder(history: list[dict[str, Any]], **kwargs: Any) -> int:
        trim_calls.append(len(history))
        return 0

    monkeypatch.setattr(mod, "trim_history_in_place", _trim_recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(tool_calls=(ToolCall(id="t-1", name="dance", args={"name": "happy"}),)),
            LLMResponse(text="Done dancing!"),
        ]
    )
    tts = _RecordingTTS()

    async def _dispatch(call: ToolCall) -> str:
        return f"ok:{call.name}"

    pipeline = mod.ComposablePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        tool_dispatcher=_dispatch,
        tools_spec=[{"name": "dance"}],
    )

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("dance for me")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    # Two LLM rounds, but exactly ONE trim call.
    assert len(trim_calls) == 1
    assert len(llm.calls) == 2

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_trim_history_cap_respected_across_user_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: with ``REACHY_MINI_MAX_HISTORY_TURNS=2`` history shrinks to 2 user turns.

    Exercises the real ``trim_history_in_place`` helper (no monkeypatch
    on the trim itself) to prove the orchestrator → helper wiring
    delivers the bound legacy already provides.
    """
    monkeypatch.setenv("REACHY_MINI_MAX_HISTORY_TURNS", "2")

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(text="reply 1"),
            LLMResponse(text="reply 2"),
            LLMResponse(text="reply 3"),
        ]
    )
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    for transcript in ("turn one", "turn two", "turn three"):
        await stt.on_completed(transcript)
        await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    user_msgs = [m for m in pipeline.conversation_history if m["role"] == "user"]
    assert len(user_msgs) == 2
    assert user_msgs[0]["content"] == "turn two"
    assert user_msgs[1]["content"] == "turn three"

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Phase 5a.2 — delivery-tag + first-audio-marker plumbing through orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speak_assistant_text_threads_delivery_tags_to_tts() -> None:
    """Phase 5a.2: ``LLMResponse.delivery_tags`` flows into
    ``TTSBackend.synthesize(tags=...)``. The orchestrator passes the tuple
    through unchanged; adapters decide whether to consume from the param or
    fall back to text parsing."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hi!", delivery_tags=("fast",))])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("ping pong")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert tts.calls == [("Hi!", ("fast",))]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_speak_assistant_text_passes_empty_tags_when_response_unpopulated() -> None:
    """When the LLM response has the default empty ``delivery_tags``, the
    orchestrator still passes an empty tuple — adapters fall back to their
    text-parsing path. Pins the fallback contract."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Plain reply.")])  # delivery_tags=() default
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)
    await stt.on_completed("ping pong")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert tts.calls == [("Plain reply.", ())]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_speak_assistant_text_allocates_first_audio_marker() -> None:
    """Phase 5a.2: orchestrator allocates a fresh ``list[float] = []`` per
    turn and passes it to ``TTSBackend.synthesize(first_audio_marker=...)``.
    Population is the adapter's responsibility — the orchestrator only owns
    the allocation."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hi.")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)
    await stt.on_completed("ping pong")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(tts.marker_refs) == 1
    marker = tts.marker_refs[0]
    assert marker is not None
    # The _RecordingTTS mock populates the marker (mimicking real-adapter
    # behaviour) so the orchestrator could read a stamp out after iteration.
    assert len(marker) == 1
    assert isinstance(marker[0], float)

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_speak_assistant_text_allocates_fresh_marker_per_turn() -> None:
    """Each turn gets its own marker list — no cross-turn aliasing."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="One."), LLMResponse(text="Two.")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)
    await stt.on_completed("first turn")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    await stt.on_completed("second turn")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(tts.marker_refs) == 2
    assert tts.marker_refs[0] is not tts.marker_refs[1]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_speak_assistant_text_puts_tuple_shape_for_fastrtc_consumer() -> None:
    """Regression: TTS frames go onto ``output_queue`` as ``(sample_rate, samples)``
    tuples, not ``AudioFrame`` dataclasses.

    ``console.py``'s playback loop dispatches with
    ``isinstance(handler_output, tuple)`` to route to ALSA (see
    ``console.py:1466``). Pre-hotfix the orchestrator pushed the ``AudioFrame``
    dataclass straight from ``TTSBackend.synthesize`` onto the queue; the
    isinstance check failed silently and TTS audio never reached the
    speaker. Caught during 2026-05-16 hardware validation on ricci.
    """
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hi.")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)
    await stt.on_completed("ping pong")

    item = await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    assert isinstance(item, tuple), (
        f"output_queue must hold (sample_rate, samples) tuples for the "
        f"FastRTC playback consumer; got {type(item).__name__}"
    )
    assert len(item) == 2
    sample_rate, samples = item
    assert sample_rate == 24000
    # Samples is whatever the adapter put inside the AudioFrame — we don't
    # constrain the dtype here, just confirm the unwrap happened.
    assert samples is not None

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Phase 5c.2 — apply_personality moved onto ComposablePipeline
# ---------------------------------------------------------------------------


class _ResettableTTS(_RecordingTTS):
    """Recording TTS that also tracks ``reset_per_session_state`` awaits."""

    def __init__(self) -> None:
        super().__init__()
        self.reset_calls: int = 0

    async def reset_per_session_state(self) -> None:
        self.reset_calls += 1


@pytest.mark.asyncio
async def test_apply_personality_resets_history_and_reseeds_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pipeline.apply_personality`` wipes history (system + user) and
    reseeds with ``get_session_instructions()``.

    Phase 5c.2 moves the wrapper's logic onto the pipeline; this test pins
    the new owner's behaviour. The wrapper-level equivalent in
    ``test_composable_conversation_handler.py`` is the post-5c.2
    "forwards to pipeline" test.
    """
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "set_custom_profile", lambda profile: None)
    monkeypatch.setattr(mod, "get_session_instructions", lambda: "fresh instructions")

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _ResettableTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="old persona")
    # Pre-seed some history that should be wiped.
    pipeline._conversation_history.append({"role": "user", "content": "hi"})

    result = await pipeline.apply_personality("rodney")

    assert "Applied personality 'rodney'" in result
    assert pipeline.conversation_history == [
        {"role": "system", "content": "fresh instructions"},
    ]


@pytest.mark.asyncio
async def test_apply_personality_awaits_tts_reset_per_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pipeline.apply_personality`` awaits ``tts.reset_per_session_state``
    exactly once on the success path. Persona switch is a hard cut on
    listening state; the TTS adapter clears its wrapped handler's
    echo-guard accumulators in this call.
    """
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "set_custom_profile", lambda profile: None)
    monkeypatch.setattr(mod, "get_session_instructions", lambda: "fresh")

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _ResettableTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    await pipeline.apply_personality("rodney")

    assert tts.reset_calls == 1, (
        f"apply_personality must await tts.reset_per_session_state exactly once; got {tts.reset_calls}"
    )


@pytest.mark.asyncio
async def test_apply_personality_returns_failure_message_on_set_custom_profile_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``set_custom_profile`` raises, the pipeline returns a failure
    string and does NOT touch history or per-session state.

    Pins the partial-failure contract: a profile-name typo does not
    half-reset the session.
    """
    from robot_comic import composable_pipeline as mod

    def _boom(profile: str | None) -> None:
        raise RuntimeError("bad profile")

    monkeypatch.setattr(mod, "set_custom_profile", _boom)
    monkeypatch.setattr(mod, "get_session_instructions", lambda: "fresh")

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _ResettableTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="old")
    pipeline._conversation_history.append({"role": "user", "content": "untouched"})
    original_history = list(pipeline._conversation_history)

    result = await pipeline.apply_personality("broken")

    assert "Failed to apply personality" in result
    assert "bad profile" in result
    # TTS reset must NOT have fired on the failure path.
    assert tts.reset_calls == 0, "tts.reset_per_session_state must NOT run when set_custom_profile fails"
    # History must be untouched on the failure path.
    assert pipeline._conversation_history == original_history, (
        "history must be untouched when set_custom_profile fails"
    )


@pytest.mark.asyncio
async def test_apply_personality_orders_history_reset_before_tts_reset_and_reseed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering: ``reset_history`` → ``tts.reset_per_session_state`` →
    history-append. Pins the ordering so a future refactor that, say,
    appends the system prompt before resetting history (which would leak
    the new prompt into the old session's tail) gets caught.
    """
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "set_custom_profile", lambda profile: None)
    monkeypatch.setattr(mod, "get_session_instructions", lambda: "FRESH-MARKER")

    call_log: list[str] = []

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])

    class _SequencingTTS(_RecordingTTS):
        async def reset_per_session_state(self) -> None:
            call_log.append("tts_reset")

    tts = _SequencingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="old persona")
    pipeline._conversation_history.append({"role": "user", "content": "u"})

    original_reset_history = pipeline.reset_history

    def _spy_reset_history(*, keep_system: bool = True) -> None:
        call_log.append(f"reset_history(keep_system={keep_system})")
        original_reset_history(keep_system=keep_system)

    pipeline.reset_history = _spy_reset_history  # type: ignore[method-assign]

    await pipeline.apply_personality("rodney")

    # reset_history fires first, then tts_reset. The history-append is the
    # last step (visible in the final history shape).
    assert call_log == [
        "reset_history(keep_system=False)",
        "tts_reset",
    ], f"unexpected call ordering: {call_log!r}"
    # Final history reflects the post-append shape.
    assert pipeline.conversation_history == [
        {"role": "system", "content": "FRESH-MARKER"},
    ]


# ---------------------------------------------------------------------------
# Phase 5e.2 — STT host concerns landed on ComposablePipeline
# ---------------------------------------------------------------------------


def _make_mock_deps(
    *,
    head_wobbler: Any = None,
    pause_controller: Any = None,
    recent_user_transcripts: list[str] | None = None,
) -> Any:
    """Return a SimpleNamespace standing in for ``ToolDependencies``.

    The pipeline only reads ``movement_manager.set_listening``,
    ``head_wobbler.reset``, ``pause_controller.handle_transcript``, and
    ``recent_user_transcripts``; a SimpleNamespace with the relevant
    attributes is enough.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    return SimpleNamespace(
        movement_manager=MagicMock(),
        head_wobbler=head_wobbler,
        pause_controller=pause_controller,
        recent_user_transcripts=recent_user_transcripts if recent_user_transcripts is not None else [],
    )


@pytest.mark.asyncio
async def test_start_up_subscribes_partial_and_speech_started_callbacks() -> None:
    """The pipeline wires all three STT callbacks through ``stt.start``."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    assert stt.on_completed is not None
    assert stt.on_partial is not None
    assert stt.on_speech_started is not None

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_speech_started_callback_opens_turn_span() -> None:
    """``_on_speech_started`` opens a root ``turn`` span + child ``stt.infer`` span."""
    stt = _ProgrammableSTT()
    pipeline = ComposablePipeline(stt=stt, llm=_ScriptedLLM([]), tts=_RecordingTTS())

    assert pipeline._turn_span is None
    assert pipeline._stt_infer_span is None

    await pipeline._on_speech_started()

    assert pipeline._turn_span is not None
    assert pipeline._stt_infer_span is not None


@pytest.mark.asyncio
async def test_speech_started_callback_calls_set_listening_true_when_deps_provided() -> None:
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS(), deps=deps)

    await pipeline._on_speech_started()

    deps.movement_manager.set_listening.assert_called_once_with(True)


@pytest.mark.asyncio
async def test_speech_started_callback_calls_head_wobbler_reset_when_provided() -> None:
    from unittest.mock import MagicMock

    head_wobbler = MagicMock()
    deps = _make_mock_deps(head_wobbler=head_wobbler)
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS(), deps=deps)

    await pipeline._on_speech_started()

    head_wobbler.reset.assert_called_once()


@pytest.mark.asyncio
async def test_speech_started_callback_calls_clear_queue_when_set() -> None:
    from unittest.mock import MagicMock

    clear_cb = MagicMock()
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())
    pipeline._clear_queue = clear_cb

    await pipeline._on_speech_started()

    clear_cb.assert_called_once()


@pytest.mark.asyncio
async def test_speech_started_callback_no_deps_does_not_raise() -> None:
    """When ``deps=None`` the callback opens the span but skips movement/wobbler."""
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())

    await pipeline._on_speech_started()  # must not raise

    assert pipeline._turn_span is not None  # span opened
    # No deps means no movement/wobbler calls to assert; the test passes
    # if no exception was raised.


@pytest.mark.asyncio
async def test_partial_callback_publishes_user_partial_to_output_queue() -> None:
    from fastrtc import AdditionalOutputs

    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())

    await pipeline._on_partial_transcript("hello rob")

    item = pipeline.output_queue.get_nowait()
    assert isinstance(item, AdditionalOutputs)
    assert item.args[0] == {"role": "user_partial", "content": "hello rob"}


@pytest.mark.asyncio
async def test_partial_callback_does_not_publish_empty_string() -> None:
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())

    await pipeline._on_partial_transcript("")

    assert pipeline.output_queue.empty()


@pytest.mark.asyncio
async def test_completed_callback_publishes_user_to_output_queue_when_deps_provided() -> None:
    from fastrtc import AdditionalOutputs

    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="reply")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    await pipeline._on_transcript_completed("hi there")

    # First item on the queue is the user-role transcript marker; the TTS
    # frame follows.
    item = pipeline.output_queue.get_nowait()
    assert isinstance(item, AdditionalOutputs)
    assert item.args[0] == {"role": "user", "content": "hi there"}


@pytest.mark.asyncio
async def test_completed_callback_records_user_transcript_when_deps_provided() -> None:
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="reply")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    await pipeline._on_transcript_completed("hello robot")

    assert "hello robot" in deps.recent_user_transcripts


@pytest.mark.asyncio
async def test_completed_callback_calls_set_listening_false_when_deps_provided() -> None:
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="reply")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    await pipeline._on_transcript_completed("hi there")

    deps.movement_manager.set_listening.assert_called_with(False)


@pytest.mark.asyncio
async def test_completed_callback_suppresses_duplicate_within_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact-match duplicate suppression within the dedup window.

    Pins the perf-counter window to a deterministic sequence so the
    test is independent of test-runner wallclock delays (e.g.
    pytest-xdist worker overhead, joke-history file I/O on the first
    LLM round-trip).
    """
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="r1"), LLMResponse(text="r2")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    # First call sets the dedup baseline at t=100.0; second call at
    # t=100.1 is well inside the ``DEDUP_WINDOW_S`` and must be dropped.
    # We also intercept the post-LLM perf_counter reads (in
    # telemetry.record_stt timing) by stamping a stable sequence.
    times = iter([100.0, 100.0, 100.05, 100.1, 100.1, 100.15])

    def _fake_perf_counter() -> float:
        try:
            return next(times)
        except StopIteration:
            return 100.2

    monkeypatch.setattr("robot_comic.composable_pipeline.time.perf_counter", _fake_perf_counter)

    await pipeline._on_transcript_completed("repeat me")
    # Immediate duplicate within the dedup window must be ignored.
    await pipeline._on_transcript_completed("repeat me")

    user_items = []
    while not pipeline.output_queue.empty():
        item = pipeline.output_queue.get_nowait()
        from fastrtc import AdditionalOutputs as AO

        if isinstance(item, AO) and item.args[0].get("role") == "user":
            user_items.append(item)
    assert len(user_items) == 1, "duplicate completion within dedup window must be dropped"


# ---------------------------------------------------------------------------
# Near-duplicate (similarity) dedup — duplicate-tts-storm fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_near_duplicate_trailing_punctuation_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two transcripts that differ only by trailing punctuation collapse to one LLM call.

    The 2026-05-16 16:58 hardware storm fingerprint: Moonshine fires
    two near-identical hypotheses for the same utterance. With strict
    text-equality dedup, both reach the LLM (which returns identical
    responses → two ``tts.synthesize`` spans of equal char_count).
    """
    deps = _make_mock_deps()
    llm = _ScriptedLLM([LLMResponse(text="r1"), LLMResponse(text="r2")])
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=llm, tts=_RecordingTTS(), deps=deps)

    times = iter([100.0, 100.0, 100.05, 100.1, 100.1, 100.15])

    def _fake_perf_counter() -> float:
        try:
            return next(times)
        except StopIteration:
            return 100.2

    monkeypatch.setattr("robot_comic.composable_pipeline.time.perf_counter", _fake_perf_counter)

    await pipeline._on_transcript_completed("Hey Rickles")
    await pipeline._on_transcript_completed("Hey Rickles.")

    assert len(llm.calls) == 1, "near-duplicate within dedup window must collapse to one LLM call"


@pytest.mark.asyncio
async def test_near_duplicate_extra_trailing_word_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late-arriving hypothesis with one extra trailing word is still suppressed."""
    deps = _make_mock_deps()
    llm = _ScriptedLLM([LLMResponse(text="r1"), LLMResponse(text="r2")])
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=llm, tts=_RecordingTTS(), deps=deps)

    times = iter([100.0, 100.0, 100.05, 100.4, 100.4, 100.45])

    def _fake_perf_counter() -> float:
        try:
            return next(times)
        except StopIteration:
            return 100.5

    monkeypatch.setattr("robot_comic.composable_pipeline.time.perf_counter", _fake_perf_counter)

    await pipeline._on_transcript_completed("how are you doing today")
    await pipeline._on_transcript_completed("how are you doing today buddy")

    assert len(llm.calls) == 1, "near-duplicate with extra trailing word must be suppressed"


@pytest.mark.asyncio
async def test_distinct_transcripts_within_window_both_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Genuinely different transcripts within the window must both dispatch.

    Guards against over-aggressive similarity dedup: a fast user
    follow-up like "wait, no" → "do the dance" must not collapse.
    """
    deps = _make_mock_deps()
    llm = _ScriptedLLM([LLMResponse(text="r1"), LLMResponse(text="r2")])
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=llm, tts=_RecordingTTS(), deps=deps)

    times = iter([100.0, 100.0, 100.05, 100.5, 100.5, 100.55])

    def _fake_perf_counter() -> float:
        try:
            return next(times)
        except StopIteration:
            return 100.6

    monkeypatch.setattr("robot_comic.composable_pipeline.time.perf_counter", _fake_perf_counter)

    await pipeline._on_transcript_completed("hello there friend")
    await pipeline._on_transcript_completed("do the dance please")

    assert len(llm.calls) == 2, "distinct transcripts within window must both dispatch"


@pytest.mark.asyncio
async def test_near_duplicate_outside_window_both_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Near-duplicates separated by more than the dedup window both dispatch.

    Operator may legitimately repeat themselves a few seconds later;
    the similarity dedup is window-bounded, not unconditional.
    """
    deps = _make_mock_deps()
    llm = _ScriptedLLM([LLMResponse(text="r1"), LLMResponse(text="r2")])
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=llm, tts=_RecordingTTS(), deps=deps)

    # Each _on_transcript_completed currently consumes ~2 perf_counter
    # reads on the deps-non-None path (one at the dedup line, one inside
    # the deps closure block). Second call's *first* read (= ``now``)
    # must land at t=105.0 to fall outside the 2.0s DEDUP_WINDOW_S.
    times = iter([100.0, 100.0, 105.0, 105.0])

    def _fake_perf_counter() -> float:
        try:
            return next(times)
        except StopIteration:
            return 105.1

    monkeypatch.setattr("robot_comic.composable_pipeline.time.perf_counter", _fake_perf_counter)

    await pipeline._on_transcript_completed("Hey Rickles")
    await pipeline._on_transcript_completed("Hey Rickles.")

    assert len(llm.calls) == 2, "near-duplicate outside dedup window must both dispatch"


@pytest.mark.asyncio
async def test_near_duplicate_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Near-duplicate dedup hits log at WARNING; exact-match stays DEBUG.

    Hardware operators need WARNING-level visibility to confirm the
    fix is firing on the next on-robot session; exact-match is the
    high-volume legacy case and would drown the journal at WARNING.
    """
    import logging

    deps = _make_mock_deps()
    llm = _ScriptedLLM([LLMResponse(text="r1"), LLMResponse(text="r2"), LLMResponse(text="r3")])
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=llm, tts=_RecordingTTS(), deps=deps)

    times = iter([100.0, 100.0, 100.05, 100.1, 100.1, 100.15, 100.2, 100.2, 100.25])

    def _fake_perf_counter() -> float:
        try:
            return next(times)
        except StopIteration:
            return 100.3

    monkeypatch.setattr("robot_comic.composable_pipeline.time.perf_counter", _fake_perf_counter)

    with caplog.at_level(logging.DEBUG, logger="robot_comic.composable_pipeline"):
        await pipeline._on_transcript_completed("Hey Rickles")
        # Exact-match — DEBUG only.
        await pipeline._on_transcript_completed("Hey Rickles")
        # Near-duplicate (trailing punctuation) — WARNING.
        await pipeline._on_transcript_completed("Hey Rickles.")

    exact_drops = [r for r in caplog.records if "Ignoring duplicate transcript" in r.getMessage()]
    near_drops = [r for r in caplog.records if "Suppressing near-duplicate transcript" in r.getMessage()]
    assert len(exact_drops) == 1
    assert exact_drops[0].levelno == logging.DEBUG
    assert len(near_drops) == 1
    assert near_drops[0].levelno == logging.WARNING


@pytest.mark.asyncio
async def test_completed_callback_pause_controller_handled_drops_transcript() -> None:
    from unittest.mock import MagicMock

    from robot_comic.pause import TranscriptDisposition

    pause = MagicMock()
    pause.handle_transcript.return_value = TranscriptDisposition.HANDLED
    deps = _make_mock_deps(pause_controller=pause)

    llm = _ScriptedLLM([])  # exhausted — would error if dispatch happens
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=llm, tts=_RecordingTTS(), deps=deps)

    await pipeline._on_transcript_completed("pause please")

    pause.handle_transcript.assert_called_once_with("pause please")
    # LLM must not have been invoked since pause handled the transcript.
    assert llm.calls == []


@pytest.mark.asyncio
async def test_completed_callback_pause_controller_dispatch_proceeds() -> None:
    from unittest.mock import MagicMock

    from robot_comic.pause import TranscriptDisposition

    pause = MagicMock()
    pause.handle_transcript.return_value = TranscriptDisposition.DISPATCH
    deps = _make_mock_deps(pause_controller=pause)

    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="ok")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    await pipeline._on_transcript_completed("normal speech")

    pause.handle_transcript.assert_called_once_with("normal speech")
    # Pipeline proceeded to LLM dispatch since pause returned DISPATCH.
    assert pipeline.conversation_history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_completed_callback_welcome_gate_waiting_drops_on_no_match() -> None:
    from robot_comic.welcome_gate import WelcomeGate

    gate = WelcomeGate(["rickles"])
    deps = _make_mock_deps()
    llm = _ScriptedLLM([])  # exhausted — drop must prevent dispatch
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=llm,
        tts=_RecordingTTS(),
        deps=deps,
        welcome_gate=gate,
    )

    await pipeline._on_transcript_completed("hello world")

    # Gate still WAITING since "rickles" not heard.
    from robot_comic.welcome_gate import GateState

    assert gate.state is GateState.WAITING
    assert llm.calls == []


@pytest.mark.asyncio
async def test_completed_callback_welcome_gate_waiting_opens_on_match_and_dispatches() -> None:
    from robot_comic.welcome_gate import GateState, WelcomeGate

    gate = WelcomeGate(["rickles"])
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="hey kid")]),
        tts=_RecordingTTS(),
        deps=deps,
        welcome_gate=gate,
    )

    await pipeline._on_transcript_completed("hey rickles")

    assert gate.state is GateState.GATED
    # LLM did run (gate just opened on the matching transcript).
    assert pipeline.conversation_history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_completed_callback_welcome_gate_gated_dispatches_immediately() -> None:
    from robot_comic.welcome_gate import GateState, WelcomeGate

    gate = WelcomeGate(["rickles"])
    gate.state = GateState.GATED  # pre-opened
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="reply")]),
        tts=_RecordingTTS(),
        deps=deps,
        welcome_gate=gate,
    )

    await pipeline._on_transcript_completed("anything else")

    assert pipeline.conversation_history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_speech_started_no_op_for_legacy_pipeline_without_callbacks() -> None:
    """A pipeline constructed without ``deps`` still wires the callbacks safely.

    Pre-5e.2 callers (host-coupled triples) don't pass ``deps`` to the
    pipeline; the mixin handles speech-start/partial on the host. The
    pipeline must not crash when its own callbacks fire in that mode.
    """
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())
    # All three callbacks must be safe no-ops with no deps / no gate.
    await pipeline._on_speech_started()
    await pipeline._on_partial_transcript("partial")
    # Completed without deps/gate falls through to the LLM loop; provide one
    # response so the loop doesn't exhaust.
    pipeline.llm = _ScriptedLLM([LLMResponse(text="ok")])
    await pipeline._on_transcript_completed("hi")


# ---------------------------------------------------------------------------
# Issue #267 — canned-line fallback for Gemini empty-STOP turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_stop_fallback_no_immediate_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two consecutive empty-STOPs sample two different canned lines.

    The no-immediate-repeat invariant prevents the operator from hearing
    the same filler twice in a row, which would feel canned/broken. The
    sampler only guarantees this when the pool has >=2 entries; the
    bundled default pool has 5, so the invariant always holds for users
    who don't override.
    """
    from robot_comic import composable_pipeline as mod
    from robot_comic import empty_stop_fallbacks as fb_mod

    monkeypatch.setattr(mod, "_resolve_empty_stop_profile_dir", lambda: None)
    # Pin to a tiny two-entry pool so the no-repeat invariant is the only
    # possible outcome (default pool sampling would still satisfy the
    # invariant, but a flake from random.choice would be misleading).
    monkeypatch.setattr(mod, "_load_empty_stop_pool", lambda _d: ("A", "B"))
    # Force pick_fallback to use the bare random.choice path; we already
    # constrained the pool to exactly two entries, so the no-repeat code
    # path is the only branch that fires.

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [LLMResponse(text=""), LLMResponse(text="")],
    )
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hello there")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    # Wait past the dedup window so the second utterance isn't suppressed
    # as a near-duplicate of the first.
    await asyncio.sleep(0)
    pipeline._last_completed_at = 0.0
    await stt.on_completed("totally different second turn")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    spoken = [text for text, _tags in tts.calls]
    assert len(spoken) == 2
    assert spoken[0] != spoken[1]
    assert set(spoken) == {"A", "B"}
    # Guard against the stale module-level cache so other tests aren't
    # affected by what we monkeypatched above.
    fb_mod.clear_cache()

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_empty_stop_fallback_skipped_when_tool_calls_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-call rounds with no text must NOT trigger the canned fallback.

    The orchestrator's ``_run_llm_loop_and_speak`` loop ``continue``-s on
    any non-empty ``tool_calls`` before reaching ``_speak_assistant_text``,
    so the canned line is only ever a substitute for the
    real-empty-STOP case — never for an in-flight tool round-trip.
    """
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "_resolve_empty_stop_profile_dir", lambda: None)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(tool_calls=(ToolCall(id="t1", name="dance", args={}),)),
            LLMResponse(text="Done."),
        ]
    )
    tts = _RecordingTTS()

    async def _dispatch(call: ToolCall) -> str:
        return "ok"

    pipeline = mod.ComposablePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        tool_dispatcher=_dispatch,
        tools_spec=[{"name": "dance"}],
    )

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("dance for me")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(tts.calls) == 1
    assert tts.calls[0][0] == "Done."

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_non_empty_text_takes_no_fallback_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-empty assistant text bypasses the #267 fallback entirely."""
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "_resolve_empty_stop_profile_dir", lambda: None)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="real punchline")])
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi robot")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert tts.calls == [("real punchline", ())]
    # No fallback recorded for the no-repeat memory.
    assert pipeline._last_empty_stop_fallback is None

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_apply_personality_resets_no_repeat_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persona switch must clear the prior persona's no-repeat memory."""
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "_resolve_empty_stop_profile_dir", lambda: None)
    monkeypatch.setattr(mod, "_load_empty_stop_pool", lambda _d: ("A", "B", "C"))

    monkeypatch.setattr(mod, "get_session_instructions", lambda: "sys")
    monkeypatch.setattr(mod, "set_custom_profile", lambda _p: None)

    pipeline = mod.ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([]),
        tts=_ResettableTTS(),
    )
    pipeline._last_empty_stop_fallback = "A"

    await pipeline.apply_personality("other_persona")
    assert pipeline._last_empty_stop_fallback is None
