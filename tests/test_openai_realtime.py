import asyncio
import base64
import logging
import time
from typing import Any
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import numpy as np

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.openai_realtime import (
    OpenaiRealtimeHandler,
    OPEN_AI_INPUT_SAMPLE_RATE,
    OPEN_AI_OUTPUT_SAMPLE_RATE,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _build_handler(loop: asyncio.AbstractEventLoop) -> OpenaiRealtimeHandler:
    asyncio.set_event_loop(loop)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return OpenaiRealtimeHandler(deps)


def _build_handler_simple() -> OpenaiRealtimeHandler:
    """Build handler without needing explicit event loop."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return OpenaiRealtimeHandler(deps)


def test_format_timestamp_uses_wall_clock() -> None:
    """Test that format_timestamp uses wall clock time."""
    loop = asyncio.new_event_loop()
    try:
        print("Testing format_timestamp...")
        handler = _build_handler(loop)
        formatted = handler.format_timestamp()
        print(f"Formatted timestamp: {formatted}")
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    # Extract year from "[YYYY-MM-DD ...]"
    year = int(formatted[1:5])
    assert year == datetime.now(timezone.utc).year

@pytest.mark.asyncio
async def test_start_up_retries_on_abrupt_close(monkeypatch: Any, caplog: Any) -> None:
    """First connection dies with ConnectionClosedError during iteration -> retried.

    Second connection iterates cleanly (no events) -> start_up returns without raising.
    Ensures handler clears self.connection at the end.
    """
    caplog.set_level(logging.WARNING)

    # Use a local Exception as the module's ConnectionClosedError to avoid ws dependency
    FakeCCE = type("FakeCCE", (Exception,), {})
    monkeypatch.setattr(rt_mod, "ConnectionClosedError", FakeCCE)

    # Make asyncio.sleep return immediately (for backoff)
    async def _fast_sleep(*_a: Any, **_kw: Any) -> None: return None
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep, raising=False)

    attempt_counter = {"n": 0}

    class FakeConn:
        """Minimal realtime connection stub."""

        def __init__(self, mode: str):
            self._mode = mode

            class _Session:
                async def update(self, **_kw: Any) -> None: return None
            self.session = _Session()

            class _InputAudioBuffer:
                async def append(self, **_kw: Any) -> None: return None
            self.input_audio_buffer = _InputAudioBuffer()

            class _Item:
                async def create(self, **_kw: Any) -> None: return None

            class _Conversation:
                item = _Item()
            self.conversation = _Conversation()

            class _Response:
                async def create(self, **_kw: Any) -> None: return None
                async def cancel(self, **_kw: Any) -> None: return None
            self.response = _Response()

        async def __aenter__(self) -> "FakeConn": return self
        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: return False
        async def close(self) -> None: return None

        # Async iterator protocol
        def __aiter__(self) -> "FakeConn": return self
        async def __anext__(self) -> None:
            if self._mode == "raise_on_iter":
                raise FakeCCE("abrupt close (simulated)")
            raise StopAsyncIteration  # clean exit (no events)

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            attempt_counter["n"] += 1
            mode = "raise_on_iter" if attempt_counter["n"] == 1 else "clean"
            return FakeConn(mode)

    class FakeClient:
        def __init__(self, **_kw: Any) -> None: self.realtime = FakeRealtime()

    # Patch the OpenAI client used by the handler
    monkeypatch.setattr(rt_mod, "AsyncOpenAI", FakeClient)

    # Build handler with minimal deps
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)

    # Run: should retry once and exit cleanly
    await handler.start_up()

    # Validate: two attempts total (fail -> retry -> succeed), and connection cleared
    assert attempt_counter["n"] == 2
    assert handler.connection is None

    # Optional: confirm we logged the unexpected close once
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "closed unexpectedly" in r.msg]
    assert len(warnings) == 1


class TestConstants:
    """Tests for module constants."""

    def test_input_sample_rate(self) -> None:
        """Test OPEN_AI_INPUT_SAMPLE_RATE constant."""
        assert OPEN_AI_INPUT_SAMPLE_RATE == 24000

    def test_output_sample_rate(self) -> None:
        """Test OPEN_AI_OUTPUT_SAMPLE_RATE constant."""
        assert OPEN_AI_OUTPUT_SAMPLE_RATE == 24000


class TestOpenaiRealtimeHandlerInit:
    """Tests for OpenaiRealtimeHandler initialization."""

    def test_init_with_deps(self) -> None:
        """Test handler initializes with dependencies."""
        handler = _build_handler_simple()

        assert handler.deps is not None
        assert handler.connection is None
        assert handler.gradio_mode is False
        assert handler.instance_path is None

    def test_init_gradio_mode(self) -> None:
        """Test handler initializes with gradio_mode."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps, gradio_mode=True)

        assert handler.gradio_mode is True

    def test_init_instance_path(self) -> None:
        """Test handler initializes with instance_path."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps, instance_path="/tmp/test")

        assert handler.instance_path == "/tmp/test"

    def test_init_sample_rates(self) -> None:
        """Test handler has correct sample rates."""
        handler = _build_handler_simple()

        assert handler.input_sample_rate == 24000
        assert handler.output_sample_rate == 24000

    def test_init_creates_output_queue(self) -> None:
        """Test handler creates output queue."""
        handler = _build_handler_simple()

        assert handler.output_queue is not None
        assert isinstance(handler.output_queue, asyncio.Queue)

    def test_init_default_flags(self) -> None:
        """Test handler has correct default flags."""
        handler = _build_handler_simple()

        assert handler.is_idle_tool_call is False
        assert handler._shutdown_requested is False
        assert handler._key_source == "env"
        assert handler._provided_api_key is None

    def test_init_debounce_settings(self) -> None:
        """Test handler has debounce settings."""
        handler = _build_handler_simple()

        assert handler.partial_transcript_task is None
        assert handler.partial_transcript_sequence == 0
        assert handler.partial_debounce_delay == 0.5

    def test_init_session_renewal_settings(self) -> None:
        """Test handler has session renewal settings."""
        handler = _build_handler_simple()

        assert handler._session_start_time == 0.0
        assert handler._session_max_duration == 55 * 60
        assert handler._session_renewal_task is None


class TestOpenaiRealtimeHandlerCopy:
    """Tests for OpenaiRealtimeHandler copy method."""

    def test_copy_creates_new_instance(self) -> None:
        """Test copy creates a new handler instance."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps, gradio_mode=True, instance_path="/test")

        copied = handler.copy()

        assert copied is not handler
        assert copied.deps is handler.deps
        assert copied.gradio_mode == handler.gradio_mode
        assert copied.instance_path == handler.instance_path


class TestOpenaiRealtimeHandlerFormatTimestamp:
    """Tests for OpenaiRealtimeHandler format_timestamp method."""

    def test_format_timestamp_contains_date(self) -> None:
        """Test format_timestamp contains date."""
        handler = _build_handler_simple()
        formatted = handler.format_timestamp()

        # Should contain date in YYYY-MM-DD format
        assert "-" in formatted
        year = int(formatted[1:5])
        assert 2020 <= year <= 2100

    def test_format_timestamp_contains_elapsed(self) -> None:
        """Test format_timestamp contains elapsed time."""
        handler = _build_handler_simple()
        time.sleep(0.1)
        formatted = handler.format_timestamp()

        # Should contain elapsed seconds
        assert "+" in formatted
        assert "s]" in formatted


class TestOpenaiRealtimeHandlerReceive:
    """Tests for OpenaiRealtimeHandler receive method."""

    @pytest.mark.asyncio
    async def test_receive_no_connection(self) -> None:
        """Test receive does nothing without connection."""
        handler = _build_handler_simple()
        handler.connection = None

        # Create audio frame
        audio = np.zeros((1, 1000), dtype=np.int16)
        frame = (24000, audio)

        # Should not raise
        await handler.receive(frame)

    @pytest.mark.asyncio
    async def test_receive_with_connection(self) -> None:
        """Test receive sends audio to connection."""
        handler = _build_handler_simple()

        # Mock connection
        mock_conn = MagicMock()
        mock_conn.input_audio_buffer = MagicMock()
        mock_conn.input_audio_buffer.append = AsyncMock()
        handler.connection = mock_conn

        # Create audio frame
        audio = np.zeros(1000, dtype=np.int16)
        frame = (24000, audio)

        await handler.receive(frame)

        mock_conn.input_audio_buffer.append.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_resamples_if_needed(self) -> None:
        """Test receive resamples audio if sample rate differs."""
        handler = _build_handler_simple()

        mock_conn = MagicMock()
        mock_conn.input_audio_buffer = MagicMock()
        mock_conn.input_audio_buffer.append = AsyncMock()
        handler.connection = mock_conn

        # Create audio at different sample rate (float for resampling)
        audio = np.zeros(2000, dtype=np.float32)
        frame = (48000, audio)  # Different from 24000

        await handler.receive(frame)

        mock_conn.input_audio_buffer.append.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_handles_stereo(self) -> None:
        """Test receive handles stereo audio."""
        handler = _build_handler_simple()

        mock_conn = MagicMock()
        mock_conn.input_audio_buffer = MagicMock()
        mock_conn.input_audio_buffer.append = AsyncMock()
        handler.connection = mock_conn

        # Create stereo audio (2 channels)
        audio = np.zeros((1000, 2), dtype=np.int16)
        frame = (24000, audio)

        await handler.receive(frame)

        mock_conn.input_audio_buffer.append.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_handles_exception(self) -> None:
        """Test receive handles connection exception gracefully."""
        handler = _build_handler_simple()

        mock_conn = MagicMock()
        mock_conn.input_audio_buffer = MagicMock()
        mock_conn.input_audio_buffer.append = AsyncMock(side_effect=Exception("Test error"))
        handler.connection = mock_conn

        audio = np.zeros(1000, dtype=np.int16)
        frame = (24000, audio)

        # Should not raise
        await handler.receive(frame)


class TestOpenaiRealtimeHandlerShutdown:
    """Tests for OpenaiRealtimeHandler shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_flag(self) -> None:
        """Test shutdown sets shutdown flag."""
        handler = _build_handler_simple()

        await handler.shutdown()

        assert handler._shutdown_requested is True

    @pytest.mark.asyncio
    async def test_shutdown_closes_connection(self) -> None:
        """Test shutdown closes connection."""
        handler = _build_handler_simple()

        mock_conn = MagicMock()
        mock_conn.close = AsyncMock()
        handler.connection = mock_conn

        await handler.shutdown()

        mock_conn.close.assert_called_once()
        assert handler.connection is None

    @pytest.mark.asyncio
    async def test_shutdown_cancels_partial_task(self) -> None:
        """Test shutdown cancels partial transcript task."""
        handler = _build_handler_simple()

        # Create a fake task
        async def fake_task() -> None:
            await asyncio.sleep(10)

        handler.partial_transcript_task = asyncio.create_task(fake_task())

        await handler.shutdown()

        assert handler.partial_transcript_task.cancelled() or handler.partial_transcript_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_renewal_task(self) -> None:
        """Test shutdown cancels session renewal task."""
        handler = _build_handler_simple()

        async def fake_task() -> None:
            await asyncio.sleep(10)

        handler._session_renewal_task = asyncio.create_task(fake_task())

        await handler.shutdown()

        assert handler._session_renewal_task.cancelled() or handler._session_renewal_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_clears_output_queue(self) -> None:
        """Test shutdown clears output queue."""
        handler = _build_handler_simple()

        # Add items to queue
        await handler.output_queue.put("item1")
        await handler.output_queue.put("item2")
        assert not handler.output_queue.empty()

        await handler.shutdown()

        assert handler.output_queue.empty()


class TestOpenaiRealtimeHandlerSendIdleSignal:
    """Tests for OpenaiRealtimeHandler send_idle_signal method."""

    @pytest.mark.asyncio
    async def test_send_idle_signal_no_connection(self) -> None:
        """Test send_idle_signal does nothing without connection."""
        handler = _build_handler_simple()
        handler.connection = None

        # Should not raise
        await handler.send_idle_signal(20.0)

    @pytest.mark.asyncio
    async def test_send_idle_signal_sets_flag(self) -> None:
        """Test send_idle_signal sets idle tool call flag."""
        handler = _build_handler_simple()

        mock_conn = MagicMock()
        mock_conn.conversation = MagicMock()
        mock_conn.conversation.item = MagicMock()
        mock_conn.conversation.item.create = AsyncMock()
        mock_conn.response = MagicMock()
        mock_conn.response.create = AsyncMock()
        handler.connection = mock_conn

        await handler.send_idle_signal(20.0)

        assert handler.is_idle_tool_call is True

    @pytest.mark.asyncio
    async def test_send_idle_signal_sends_message(self) -> None:
        """Test send_idle_signal sends message to connection."""
        handler = _build_handler_simple()

        mock_conn = MagicMock()
        mock_conn.conversation = MagicMock()
        mock_conn.conversation.item = MagicMock()
        mock_conn.conversation.item.create = AsyncMock()
        mock_conn.response = MagicMock()
        mock_conn.response.create = AsyncMock()
        handler.connection = mock_conn

        await handler.send_idle_signal(20.0)

        mock_conn.conversation.item.create.assert_called_once()
        mock_conn.response.create.assert_called_once()


class TestOpenaiRealtimeHandlerEmitDebouncedPartial:
    """Tests for OpenaiRealtimeHandler _emit_debounced_partial method."""

    @pytest.mark.asyncio
    async def test_emit_debounced_partial_waits(self) -> None:
        """Test _emit_debounced_partial waits for debounce delay."""
        handler = _build_handler_simple()
        handler.partial_debounce_delay = 0.05  # Short delay for testing

        start = time.monotonic()
        await handler._emit_debounced_partial("test", 1)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_emit_debounced_partial_checks_sequence(self) -> None:
        """Test _emit_debounced_partial checks sequence number."""
        handler = _build_handler_simple()
        handler.partial_debounce_delay = 0.01
        handler.partial_transcript_sequence = 5

        # Old sequence should not emit
        await handler._emit_debounced_partial("test", 1)

        # Queue should be empty (sequence mismatch)
        assert handler.output_queue.empty()

    @pytest.mark.asyncio
    async def test_emit_debounced_partial_emits_on_match(self) -> None:
        """Test _emit_debounced_partial emits on sequence match."""
        handler = _build_handler_simple()
        handler.partial_debounce_delay = 0.01
        handler.partial_transcript_sequence = 5

        await handler._emit_debounced_partial("test", 5)

        assert not handler.output_queue.empty()


class TestOpenaiRealtimeHandlerApplyPersonality:
    """Tests for OpenaiRealtimeHandler apply_personality method."""

    @pytest.mark.asyncio
    async def test_apply_personality_no_connection(self) -> None:
        """Test apply_personality without connection."""
        handler = _build_handler_simple()
        handler.connection = None

        with patch("reachy_mini_conversation_app.config.set_custom_profile"):
            with patch("reachy_mini_conversation_app.prompts.get_session_instructions", return_value="test"):
                with patch("reachy_mini_conversation_app.prompts.get_session_voice", return_value="alloy"):
                    result = await handler.apply_personality("test_profile")

        assert "Will take effect on next connection" in result

    @pytest.mark.asyncio
    async def test_apply_personality_with_connection(self) -> None:
        """Test apply_personality with active connection."""
        handler = _build_handler_simple()

        mock_conn = MagicMock()
        mock_conn.session = MagicMock()
        mock_conn.session.update = AsyncMock()
        mock_conn.close = AsyncMock()
        handler.connection = mock_conn

        # Mock client for restart
        handler.client = MagicMock()

        with patch("reachy_mini_conversation_app.config.set_custom_profile"):
            with patch("reachy_mini_conversation_app.prompts.get_session_instructions", return_value="test"):
                with patch("reachy_mini_conversation_app.prompts.get_session_voice", return_value="alloy"):
                    with patch.object(handler, "_restart_session", new_callable=AsyncMock):
                        result = await handler.apply_personality("test_profile")

        assert "Applied personality" in result


class TestOpenaiRealtimeHandlerGetAvailableVoices:
    """Tests for OpenaiRealtimeHandler get_available_voices method."""

    @pytest.mark.asyncio
    async def test_get_available_voices_fallback(self) -> None:
        """Test get_available_voices returns fallback on error."""
        handler = _build_handler_simple()

        mock_client = MagicMock()
        mock_client.models = MagicMock()
        mock_client.models.retrieve = AsyncMock(side_effect=Exception("API error"))
        handler.client = mock_client

        voices = await handler.get_available_voices()

        assert "cedar" in voices
        assert "alloy" in voices

    @pytest.mark.asyncio
    async def test_get_available_voices_default_first(self) -> None:
        """Test get_available_voices has cedar first."""
        handler = _build_handler_simple()

        mock_client = MagicMock()
        mock_client.models = MagicMock()
        mock_client.models.retrieve = AsyncMock(side_effect=Exception("API error"))
        handler.client = mock_client

        voices = await handler.get_available_voices()

        assert voices[0] == "cedar"


class TestOpenaiRealtimeHandlerPersistApiKey:
    """Tests for OpenaiRealtimeHandler _persist_api_key_if_needed method."""

    def test_persist_api_key_not_gradio_mode(self) -> None:
        """Test _persist_api_key_if_needed does nothing outside gradio mode."""
        handler = _build_handler_simple()
        handler.gradio_mode = False

        # Should not raise
        handler._persist_api_key_if_needed()

    def test_persist_api_key_not_textbox_source(self) -> None:
        """Test _persist_api_key_if_needed does nothing when key from env."""
        handler = _build_handler_simple()
        handler.gradio_mode = True
        handler._key_source = "env"

        # Should not raise
        handler._persist_api_key_if_needed()

    def test_persist_api_key_no_key(self) -> None:
        """Test _persist_api_key_if_needed does nothing without key."""
        handler = _build_handler_simple()
        handler.gradio_mode = True
        handler._key_source = "textbox"
        handler._provided_api_key = ""

        # Should not raise
        handler._persist_api_key_if_needed()

    def test_persist_api_key_no_instance_path(self) -> None:
        """Test _persist_api_key_if_needed does nothing without instance path."""
        handler = _build_handler_simple()
        handler.gradio_mode = True
        handler._key_source = "textbox"
        handler._provided_api_key = "test_key"
        handler.instance_path = None

        # Should not raise
        handler._persist_api_key_if_needed()


class TestOpenaiRealtimeHandlerCheckBackgroundCompletions:
    """Tests for OpenaiRealtimeHandler _check_background_completions method."""

    @pytest.mark.asyncio
    async def test_check_background_completions_no_connection(self) -> None:
        """Test _check_background_completions does nothing without connection."""
        handler = _build_handler_simple()
        handler.connection = None

        # Should not raise
        await handler._check_background_completions()

    @pytest.mark.asyncio
    async def test_check_background_completions_no_notifications(self) -> None:
        """Test _check_background_completions with no notifications."""
        handler = _build_handler_simple()

        mock_conn = MagicMock()
        handler.connection = mock_conn

        with patch("reachy_mini_conversation_app.openai_realtime.BackgroundTaskManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.get_pending_notification.return_value = None
            mock_manager.get_instance.return_value = mock_instance

            # Should not raise
            await handler._check_background_completions()
