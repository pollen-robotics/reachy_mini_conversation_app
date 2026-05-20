"""Tests for TTS-down cascade suppression in ComposablePipeline.

Hardware context (2026-05-18): xtts at astralplane.lan:8020 was firewall-blocked
from ricci; 43 interrupted turns in 32 minutes, all hallucinating on ambient mic
input. The cascade is:
  ambient VAD → STT → LLM → TTS (ConnectTimeout) → silence → repeat

The fix: after a TTS service-down exception (ConnectTimeout / ConnectError /
ReadTimeout / OSError), the orchestrator sets ``_tts_down_until`` to suppress
new STT-triggered turns for ``REACHY_MINI_TTS_DOWN_COOLDOWN_S`` seconds. The
cooldown clears automatically on the first successful TTS call.
"""

from __future__ import annotations
import time
import asyncio
from typing import Any, AsyncIterator

import httpx
import pytest

from robot_comic.config import config as _cfg
from robot_comic.backends import AudioFrame, LLMResponse
from robot_comic.composable_pipeline import (
    _TTS_DOWN_COOLDOWN_ENV,
    _TTS_DOWN_COOLDOWN_MAX,
    _TTS_DOWN_COOLDOWN_MIN,
    _TTS_DOWN_COOLDOWN_DEFAULT,
    ComposablePipeline,
)


@pytest.fixture(autouse=True)
def _disable_startup_greeting(monkeypatch: pytest.MonkeyPatch):
    """Suppress the #504 synthetic startup greeting for every test in this
    module so the TTS-down cooldown assertions aren't pre-armed by the
    greeting itself. The greeting is covered in
    ``tests/test_composable_pipeline_startup_greeting.py``.
    """
    monkeypatch.setattr(_cfg, "STARTUP_GREETING_ENABLED", False)
    yield


# ---------------------------------------------------------------------------
# Re-usable mock backends (minimal surface, focused on TTS-down scenario)
# ---------------------------------------------------------------------------


class _ProgrammableSTT:
    """STT mock that exposes bound callbacks for test-driven event injection."""

    def __init__(self) -> None:
        self.on_completed = None
        self.on_partial = None
        self.on_speech_started = None
        self.stopped = False
        self.started = False

    async def start(self, on_completed, on_partial=None, on_speech_started=None) -> None:
        self.started = True
        self.on_completed = on_completed
        self.on_partial = on_partial
        self.on_speech_started = on_speech_started

    async def feed_audio(self, frame: AudioFrame) -> None:  # pragma: no cover
        pass

    async def stop(self) -> None:
        self.stopped = True


class _AlwaysTextLLM:
    """LLM mock that always returns a fixed text response."""

    def __init__(self, text: str = "Hello from the LLM") -> None:
        self._text = text
        self.call_count = 0
        self.prepared = False
        self.shutdown_called = False

    async def prepare(self) -> None:
        self.prepared = True

    async def chat(self, messages: list[dict[str, Any]], tools=None) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(text=self._text)

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _FailingTTS:
    """TTS mock that raises a configurable exception on every ``synthesize`` call."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.call_count = 0
        self.prepared = False
        self.shutdown_called = False

    async def prepare(self) -> None:
        self.prepared = True

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.call_count += 1
        raise self._exc
        yield  # make this an async generator

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def reset_per_session_state(self) -> None:
        pass


class _RecoveringTTS:
    """TTS mock that fails on the first N calls then succeeds."""

    def __init__(self, fail_count: int, exc: BaseException) -> None:
        self._fail_count = fail_count
        self._exc = exc
        self.call_count = 0
        self.prepared = False
        self.shutdown_called = False

    async def prepare(self) -> None:
        self.prepared = True

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise self._exc
        yield AudioFrame(samples=[0, 0], sample_rate=24000)

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def reset_per_session_state(self) -> None:
        pass


class _GoodTTS:
    """TTS mock that always succeeds."""

    def __init__(self) -> None:
        self.call_count = 0
        self.prepared = False
        self.shutdown_called = False

    async def prepare(self) -> None:
        self.prepared = True

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.call_count += 1
        yield AudioFrame(samples=[0, 0], sample_rate=24000)

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def reset_per_session_state(self) -> None:
        pass


async def _wait_for_callback(stt: _ProgrammableSTT, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while stt.on_completed is None:
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError("STT.on_completed was never bound")
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Helper: drain all pending asyncio tasks
# ---------------------------------------------------------------------------


async def _drain(n: int = 20) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Test: ConnectTimeout arms the cooldown, subsequent triggers are suppressed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_timeout_arms_cooldown_and_suppresses_subsequent_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a ConnectTimeout, subsequent STT-triggered turns must be no-ops.

    The cascade scenario: TTS is unreachable → every ambient VAD trigger
    would fire another round-trip. The gate must stop those.
    """
    exc = httpx.ConnectTimeout("simulated timeout")
    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    tts = _FailingTTS(exc)

    # Use a long cooldown so it doesn't expire during the test.
    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "3600")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    # First turn: LLM fires, TTS raises ConnectTimeout → cooldown armed.
    await stt.on_completed("hello robot this is the first turn")
    await _drain()

    assert tts.call_count == 1, "TTS should have been called once (and failed)"
    assert pipeline._tts_down_until > 0.0, "_tts_down_until should be set"

    llm_calls_after_first = llm.call_count

    # Second turn: should be suppressed at the gate — LLM must NOT be called again.
    await stt.on_completed("hello robot this is the second turn")
    await _drain()

    assert llm.call_count == llm_calls_after_first, "LLM must not be called while TTS is in cooldown"
    assert tts.call_count == 1, "TTS must not be called while in cooldown"

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Test: ConnectError also arms the cooldown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_error_arms_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """ConnectError (e.g. connection refused) is treated as service-down."""
    exc = httpx.ConnectError("connection refused")
    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    tts = _FailingTTS(exc)

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "3600")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hello this is a test turn ok")
    await _drain()

    assert pipeline._tts_down_until > 0.0
    llm_count = llm.call_count

    # Subsequent trigger suppressed.
    await stt.on_completed("another ambient trigger suppressed please")
    await _drain()
    assert llm.call_count == llm_count

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Test: ReadTimeout also arms the cooldown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_timeout_arms_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """ReadTimeout (server accepted but never streamed) is treated as service-down."""
    exc = httpx.ReadTimeout("read timeout")
    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    tts = _FailingTTS(exc)

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "3600")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hello this is a read timeout test")
    await _drain()

    assert pipeline._tts_down_until > 0.0

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Test: OSError also arms the cooldown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oserror_arms_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """OSError (e.g. no route to host) is treated as service-down."""
    exc = OSError("no route to host")
    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    tts = _FailingTTS(exc)

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "3600")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hello oserror test turn right here")
    await _drain()

    assert pipeline._tts_down_until > 0.0

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Test: cooldown expiry allows turns to resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_expiry_resumes_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the cooldown window expires, STT-triggered turns must fire again."""
    exc = httpx.ConnectTimeout("simulated timeout")
    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    # After first failure, switch to a good TTS for recovery verification.
    good_tts = _GoodTTS()
    fail_tts = _FailingTTS(exc)

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "3600")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=fail_tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    # Arm the cooldown.
    await stt.on_completed("hello robot this arms the cooldown now")
    await _drain()
    assert pipeline._tts_down_until > 0.0

    # Manually expire the cooldown by back-dating ``_tts_down_until``.
    pipeline._tts_down_until = time.monotonic() - 1.0

    # Swap in a working TTS so the next turn succeeds.
    pipeline.tts = good_tts

    llm_count = llm.call_count
    await stt.on_completed("this turn should fire because cooldown expired")
    await _drain(30)

    # LLM and TTS should both have been called.
    assert llm.call_count > llm_count, "LLM must fire after cooldown expiry"
    assert good_tts.call_count >= 1, "TTS must be called after cooldown expiry"

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Test: successful TTS call clears the cooldown immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_tts_clears_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """When TTS recovers, ``_tts_down_until`` must be reset to 0 and turns resume.

    Scenario: first call fails → cooldown armed → cooldown expires → second
    call succeeds → ``_tts_down_until`` cleared → subsequent triggers fire.
    """
    exc = httpx.ConnectTimeout("timeout")
    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    # First call fails, then succeeds on call 2+.
    tts = _RecoveringTTS(fail_count=1, exc=exc)

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "3600")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    # First turn: TTS fails → cooldown armed.
    await stt.on_completed("hello robot this first turn will fail")
    await _drain()
    assert pipeline._tts_down_until > 0.0, "cooldown should be armed after first failure"

    # Manually expire the cooldown so the second turn gets through.
    pipeline._tts_down_until = time.monotonic() - 1.0

    # Second turn: TTS succeeds → cooldown must be cleared.
    await stt.on_completed("the second turn should recover cleanly now")
    # Drain until TTS call 2 has fired (wait up to 2s).
    deadline = asyncio.get_event_loop().time() + 2.0
    while tts.call_count < 2 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0)

    assert tts.call_count == 2, f"Expected 2 TTS calls, got {tts.call_count}"
    assert pipeline._tts_down_until == 0.0, "_tts_down_until must be cleared after a successful TTS call"

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Test: env-var cooldown duration is used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_var_cooldown_duration_is_respected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``REACHY_MINI_TTS_DOWN_COOLDOWN_S`` sets the cooldown window width."""
    exc = httpx.ConnectTimeout("timeout")
    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    tts = _FailingTTS(exc)

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "120")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    before = time.monotonic()
    await stt.on_completed("hello robot setting cooldown to 120 seconds")
    await _drain()

    # ``_tts_down_until`` should be approximately ``before + 120`` (within 5 s).
    assert pipeline._tts_down_until > 0.0
    expected_min = before + 115.0
    expected_max = before + 125.0
    assert expected_min <= pipeline._tts_down_until <= expected_max, (
        f"_tts_down_until={pipeline._tts_down_until:.1f} not in [{expected_min:.1f}, {expected_max:.1f}]"
    )

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Test: out-of-range env-var clamps and logs a warning
# ---------------------------------------------------------------------------


def test_tts_down_cooldown_s_clamps_below_minimum(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Values below 5.0 s are clamped to 5.0 s with a warning."""
    import logging

    from robot_comic.composable_pipeline import _tts_down_cooldown_s

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="robot_comic.composable_pipeline"):
        result = _tts_down_cooldown_s()

    assert result == _TTS_DOWN_COOLDOWN_MIN
    assert any("clamping" in r.getMessage() for r in caplog.records)


def test_tts_down_cooldown_s_clamps_above_maximum(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Values above 600.0 s are clamped to 600.0 s with a warning."""
    import logging

    from robot_comic.composable_pipeline import _tts_down_cooldown_s

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "99999")
    with caplog.at_level(logging.WARNING, logger="robot_comic.composable_pipeline"):
        result = _tts_down_cooldown_s()

    assert result == _TTS_DOWN_COOLDOWN_MAX
    assert any("clamping" in r.getMessage() for r in caplog.records)


def test_tts_down_cooldown_s_uses_default_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When env var is absent the default (45.0 s) is returned."""
    from robot_comic.composable_pipeline import _tts_down_cooldown_s

    monkeypatch.delenv(_TTS_DOWN_COOLDOWN_ENV, raising=False)
    assert _tts_down_cooldown_s() == _TTS_DOWN_COOLDOWN_DEFAULT


def test_tts_down_cooldown_s_invalid_value_uses_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-numeric env-var value falls back to the default with a warning."""
    import logging

    from robot_comic.composable_pipeline import _tts_down_cooldown_s

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "banana")
    with caplog.at_level(logging.WARNING, logger="robot_comic.composable_pipeline"):
        result = _tts_down_cooldown_s()

    assert result == _TTS_DOWN_COOLDOWN_DEFAULT
    assert any("not a valid float" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: successful TTS path is unchanged — no regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_tts_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: successful TTS still produces frames and no cooldown is set."""
    monkeypatch.delenv(_TTS_DOWN_COOLDOWN_ENV, raising=False)

    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    tts = _GoodTTS()

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hello robot say something please")
    item = await asyncio.wait_for(pipeline.output_queue.get(), timeout=2.0)

    # Frame should be the (sample_rate, samples) tuple from _speak_assistant_text.
    assert isinstance(item, tuple)
    sample_rate, _samples = item
    assert sample_rate == 24000

    # No cooldown armed.
    assert pipeline._tts_down_until == 0.0

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Test: warning logged once on cooldown entry, then suppressed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warning_logged_once_on_tts_down(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The WARNING must fire exactly once when the cooldown is entered."""
    import logging

    exc = httpx.ConnectTimeout("timeout")
    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    tts = _FailingTTS(exc)

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "3600")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    with caplog.at_level(logging.WARNING, logger="robot_comic.composable_pipeline"):
        caplog.clear()
        # First turn arms cooldown.
        await stt.on_completed("hello robot this is turn number one")
        await _drain()

    warning_messages = [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "TTS unreachable" in r.getMessage()
    ]
    assert len(warning_messages) == 1, f"Expected exactly 1 TTS-unreachable WARNING; got {len(warning_messages)}"

    # Subsequent suppressed turns should NOT emit additional TTS-unreachable warnings.
    with caplog.at_level(logging.WARNING, logger="robot_comic.composable_pipeline"):
        caplog.clear()
        await stt.on_completed("this second trigger should be silently suppressed")
        await _drain()

    suppression_warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and "TTS unreachable" in r.getMessage()
    ]
    assert len(suppression_warnings) == 0, "No second TTS-unreachable WARNING should fire during active cooldown"

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Test: recovery INFO log fires once when TTS recovers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_info_logged_once(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """An INFO log fires exactly once when TTS recovers after a cooldown."""
    import logging

    exc = httpx.ConnectTimeout("timeout")
    stt = _ProgrammableSTT()
    llm = _AlwaysTextLLM()
    tts = _RecoveringTTS(fail_count=1, exc=exc)

    monkeypatch.setenv(_TTS_DOWN_COOLDOWN_ENV, "3600")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)
    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    # First turn fails and arms cooldown.
    await stt.on_completed("hello robot first turn failure test")
    await _drain()
    assert pipeline._tts_down_until > 0.0

    # Expire the cooldown.
    pipeline._tts_down_until = time.monotonic() - 1.0

    # Second turn succeeds; expect the recovery INFO.
    with caplog.at_level(logging.INFO, logger="robot_comic.composable_pipeline"):
        caplog.clear()
        await stt.on_completed("second turn should recover and log info now")
        deadline = asyncio.get_event_loop().time() + 2.0
        while tts.call_count < 2 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0)

    recovery_logs = [r for r in caplog.records if r.levelno == logging.INFO and "TTS recovered" in r.getMessage()]
    assert len(recovery_logs) == 1, f"Expected exactly 1 'TTS recovered' INFO log; got {len(recovery_logs)}"

    await pipeline.shutdown()
    await task
