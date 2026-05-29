import random
import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastrtc import AdditionalOutputs

import reachy_mini_conversation_app.base_realtime as base_rt_mod
import reachy_mini_conversation_app.openai_realtime as rt_mod
import reachy_mini_conversation_app.tools.core_tools as ct_mod
import reachy_mini_conversation_app.tools.background_tool_manager as btm_mod
from reachy_mini_conversation_app.config import OPENAI_BACKEND, config, get_default_voice_for_backend
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.background_tool_manager import ToolCallRoutine


OPENAI_DEFAULT_VOICE = get_default_voice_for_backend(OPENAI_BACKEND)


def _build_handler(loop: asyncio.AbstractEventLoop) -> OpenaiRealtimeHandler:
    asyncio.set_event_loop(loop)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return OpenaiRealtimeHandler(deps)


async def _run_openai_handler_with_events(
    monkeypatch: Any,
    events: list[Any],
    *,
    movement_manager: MagicMock | None = None,
) -> OpenaiRealtimeHandler:
    """Run an OpenAI realtime handler against a fixed event sequence."""
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda default=OPENAI_DEFAULT_VOICE: "alloy")
    monkeypatch.setattr(rt_mod, "get_active_tool_specs", lambda _: [])

    class FakeSession:
        async def update(self, **_kw: Any) -> None:
            pass

    class FakeInputAudioBuffer:
        async def append(self, **_kw: Any) -> None:
            pass

    class FakeItem:
        async def create(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeItem()

    class FakeResponse:
        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            pass

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeInputAudioBuffer()
        conversation = FakeConversation()
        response = FakeResponse()

        def __init__(self) -> None:
            self._events = iter(events)

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> Any:
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            return FakeConn()

    class FakeClient:
        def __init__(self) -> None:
            self.realtime = FakeRealtime()

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager or MagicMock())
    handler = OpenaiRealtimeHandler(deps)
    handler.client = FakeClient()

    start_up = MagicMock()
    shutdown = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_up", start_up)
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", shutdown)

    await handler._run_realtime_session()
    return handler


@pytest.mark.asyncio
async def test_tool_completion_does_not_reset_head_wobbler(monkeypatch: Any) -> None:
    """Tool completion should not interrupt ongoing speech wobble."""
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda default=OPENAI_DEFAULT_VOICE: "alloy")
    monkeypatch.setattr(rt_mod, "get_active_tool_specs", lambda _: [])

    async def _fake_dispatch(tool_name: str, args_json: str, deps: Any, **_kw: Any) -> dict[str, Any]:
        return {"image_description": "A person in front of a door.", "tool": tool_name}

    monkeypatch.setattr(btm_mod, "dispatch_tool_call", _fake_dispatch)

    class FakeSession:
        async def update(self, **_kw: Any) -> None:
            pass

    class FakeInputAudioBuffer:
        async def append(self, **_kw: Any) -> None:
            pass

    class FakeItem:
        def __init__(self) -> None:
            self.created = asyncio.Event()

        async def create(self, **_kw: Any) -> None:
            self.created.set()

    class FakeConversation:
        def __init__(self) -> None:
            self.item = FakeItem()

    class FakeResponse:
        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            pass

    class FakeConn:
        def __init__(self) -> None:
            self._events: asyncio.Queue[Any] = asyncio.Queue()
            self.session = FakeSession()
            self.input_audio_buffer = FakeInputAudioBuffer()
            self.conversation = FakeConversation()
            self.response = FakeResponse()

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            await self._events.put(None)

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> Any:
            event = await self._events.get()
            if event is None:
                raise StopAsyncIteration
            return event

    class FakeRealtime:
        def __init__(self) -> None:
            self.conn = FakeConn()

        def connect(self, **_kw: Any) -> FakeConn:
            return self.conn

    class FakeClient:
        def __init__(self) -> None:
            self.realtime = FakeRealtime()

    head_wobbler = MagicMock()
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        head_wobbler=head_wobbler,
    )
    handler = OpenaiRealtimeHandler(deps)
    fake_client = FakeClient()
    handler.client = fake_client

    session_task = asyncio.create_task(handler._run_realtime_session())

    await asyncio.sleep(0)
    await handler.tool_manager.start_tool(
        call_id="call_1",
        tool_call_routine=ToolCallRoutine(
            tool_name="camera",
            args_json_str='{"question":"What do I see?"}',
            deps=deps,
        ),
        is_idle_tool_call=False,
    )

    fake_conn = fake_client.realtime.conn
    await asyncio.wait_for(fake_conn.conversation.item.created.wait(), timeout=2.0)
    await fake_conn.close()
    await asyncio.wait_for(session_task, timeout=2.0)

    head_wobbler.reset.assert_not_called()


@pytest.mark.asyncio
async def test_non_idle_tool_call_does_not_queue_progress_response(monkeypatch: Any) -> None:
    """Tool-call startup should not enqueue a second speech response."""
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda default=OPENAI_DEFAULT_VOICE: "alloy")
    monkeypatch.setattr(rt_mod, "get_active_tool_specs", lambda _: [])

    class FakeEvent:
        def __init__(self, etype: str, **kwargs: Any) -> None:
            self.type = etype
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeSession:
        async def update(self, **_kw: Any) -> None:
            pass

    class FakeInputAudioBuffer:
        async def append(self, **_kw: Any) -> None:
            pass

    class FakeItem:
        async def create(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeItem()

    class FakeResponse:
        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            pass

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeInputAudioBuffer()
        conversation = FakeConversation()
        response = FakeResponse()

        def __init__(self) -> None:
            self._events = iter(
                [
                    FakeEvent(
                        "response.function_call_arguments.done",
                        name="camera",
                        arguments='{"question":"What do I see?"}',
                        call_id="call_camera_1",
                    )
                ]
            )

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> FakeEvent:
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            return FakeConn()

    class FakeClient:
        def __init__(self) -> None:
            self.realtime = FakeRealtime()

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = OpenaiRealtimeHandler(deps)
    fake_client = FakeClient()
    handler.client = fake_client
    safe_response_create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", safe_response_create)
    start_up = MagicMock()
    shutdown = AsyncMock()
    start_tool = AsyncMock(return_value=MagicMock(tool_id="camera-call_camera_1-0"))
    monkeypatch.setattr(type(handler.tool_manager), "start_up", start_up)
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", shutdown)
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    await handler._run_realtime_session()

    start_tool.assert_awaited_once()
    safe_response_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_speech_events_reset_idle_timer(monkeypatch: Any) -> None:
    """User speech/transcription events should postpone idle behavior."""
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda default=OPENAI_DEFAULT_VOICE: "alloy")
    monkeypatch.setattr(rt_mod, "get_active_tool_specs", lambda _: [])

    class FakeEvent:
        def __init__(self, etype: str, **kwargs: Any) -> None:
            self.type = etype
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeSession:
        async def update(self, **_kw: Any) -> None:
            pass

    class FakeInputAudioBuffer:
        async def append(self, **_kw: Any) -> None:
            pass

    class FakeItem:
        async def create(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeItem()

    class FakeResponse:
        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            pass

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeInputAudioBuffer()
        conversation = FakeConversation()
        response = FakeResponse()

        def __init__(self) -> None:
            self._events = iter(
                [
                    FakeEvent("input_audio_buffer.speech_started"),
                    FakeEvent("conversation.item.input_audio_transcription.completed", transcript="hello there"),
                ]
            )

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> FakeEvent:
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            return FakeConn()

    class FakeClient:
        def __init__(self) -> None:
            self.realtime = FakeRealtime()

    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    handler = OpenaiRealtimeHandler(deps)
    fake_client = FakeClient()
    handler.client = fake_client
    handler.last_activity_time = asyncio.get_running_loop().time() - 60.0

    start_up = MagicMock()
    shutdown = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_up", start_up)
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", shutdown)

    previous_activity_time = handler.last_activity_time
    await handler._run_realtime_session()

    assert handler.last_activity_time > previous_activity_time
    movement_manager.set_listening.assert_any_call(True)


@pytest.mark.asyncio
async def test_empty_user_transcript_exits_listening_without_chat_message(monkeypatch: Any) -> None:
    """Blank VAD commits should not leave listening motion frozen."""
    movement_manager = MagicMock()

    handler = await _run_openai_handler_with_events(
        monkeypatch,
        [
            SimpleNamespace(type="input_audio_buffer.speech_started"),
            SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="   "),
        ],
        movement_manager=movement_manager,
    )

    assert [call.args[0] for call in movement_manager.set_listening.call_args_list] == [True, False]
    assert handler.output_queue.empty()
    assert handler._turn_user_done_at is None


@pytest.mark.asyncio
async def test_empty_audio_buffer_error_exits_listening_without_chat_error(monkeypatch: Any) -> None:
    """Empty audio-buffer commits are internal and should restore listening state."""
    movement_manager = MagicMock()

    handler = await _run_openai_handler_with_events(
        monkeypatch,
        [
            SimpleNamespace(type="input_audio_buffer.speech_started"),
            SimpleNamespace(
                type="error",
                error=SimpleNamespace(code="input_audio_buffer_commit_empty", message="empty audio buffer"),
            ),
        ],
        movement_manager=movement_manager,
    )

    assert [call.args[0] for call in movement_manager.set_listening.call_args_list] == [True, False]
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_apply_personality_preserves_manual_voice_override(monkeypatch: Any) -> None:
    """Applying a profile should not discard a voice manually selected in the current session."""
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda: "cedar")
    monkeypatch.setattr("reachy_mini_conversation_app.config.set_custom_profile", lambda _profile: None)

    handler = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    update = AsyncMock()
    handler.connection = SimpleNamespace(session=SimpleNamespace(update=update))
    handler._voice_override = "marin"
    restart = AsyncMock()
    monkeypatch.setattr(handler, "_restart_session", restart)

    status = await handler.apply_personality("example")

    assert status == "Applied personality and restarted realtime session."
    assert handler.get_current_voice() == "marin"
    restart.assert_awaited_once()
    session = update.await_args.kwargs["session"]
    assert session["audio"]["output"]["voice"] == "marin"


def test_handler_uses_startup_voice_at_startup(monkeypatch: Any) -> None:
    """OpenAI handler startup should restore a persisted startup voice."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")

    handler = OpenaiRealtimeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="shimmer",
    )

    assert handler.get_current_voice() == "shimmer"


def test_copy_preserves_current_voice_override(monkeypatch: Any) -> None:
    """Copied OpenAI handlers should keep the current voice override."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")

    handler = OpenaiRealtimeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="shimmer",
    )
    handler._voice_override = "marin"

    copied_handler = handler.copy()

    assert copied_handler.get_current_voice() == "marin"


def test_format_timestamp_uses_wall_clock() -> None:
    """Test that format_timestamp uses wall clock time."""
    try:
        previous_loop = asyncio.get_event_loop()
    except RuntimeError:
        previous_loop = asyncio.new_event_loop()
    loop = asyncio.new_event_loop()
    try:
        print("Testing format_timestamp...")
        handler = _build_handler(loop)
        formatted = handler.format_timestamp()
        print(f"Formatted timestamp: {formatted}")
    finally:
        loop.close()
        asyncio.set_event_loop(previous_loop)

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

    # Use a local Exception as the base module's ConnectionClosedError to avoid ws dependency.
    FakeCCE = type("FakeCCE", (Exception,), {})
    monkeypatch.setattr(base_rt_mod, "ConnectionClosedError", FakeCCE)

    # Make asyncio.sleep return immediately (for backoff)
    _real_sleep = asyncio.sleep

    async def _mock_sleep(*_a: Any, **_kw: Any) -> None:
        await _real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _mock_sleep, raising=False)

    attempt_counter = {"n": 0}

    class FakeConn:
        """Minimal realtime connection stub."""

        def __init__(self, mode: str):
            self._mode = mode

            class _Session:
                async def update(self, **_kw: Any) -> None:
                    return None

            self.session = _Session()

            class _InputAudioBuffer:
                async def append(self, **_kw: Any) -> None:
                    return None

            self.input_audio_buffer = _InputAudioBuffer()

            class _Item:
                async def create(self, **_kw: Any) -> None:
                    return None

            class _Conversation:
                item = _Item()

            self.conversation = _Conversation()

            class _Response:
                async def create(self, **_kw: Any) -> None:
                    return None

                async def cancel(self, **_kw: Any) -> None:
                    return None

            self.response = _Response()

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def close(self) -> None:
            return None

        # Async iterator protocol
        def __aiter__(self) -> "FakeConn":
            return self

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
        def __init__(self, **_kw: Any) -> None:
            self.realtime = FakeRealtime()

    # Patch the OpenAI client used by the handler
    monkeypatch.setattr(rt_mod, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")

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


@pytest.mark.asyncio
async def test_start_up_openai_gradio_collects_textbox_api_key(monkeypatch: Any) -> None:
    """OpenAI should own Gradio textbox credential collection."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps, gradio_mode=True)
    handler.latest_args = ["profile", "voice", "unused", "sk-textbox-secret"]

    build_client = AsyncMock(return_value=MagicMock())
    run_realtime_session = AsyncMock(return_value=None)
    wait_for_args = AsyncMock(return_value=None)

    monkeypatch.setattr(handler, "_build_realtime_client", build_client)
    monkeypatch.setattr(handler, "_run_realtime_session", run_realtime_session)
    monkeypatch.setattr(handler, "wait_for_args", wait_for_args)

    await handler.start_up()

    wait_for_args.assert_awaited_once()
    build_client.assert_awaited_once_with()
    run_realtime_session.assert_awaited_once()
    assert handler._provided_api_key == "sk-textbox-secret"


@pytest.mark.asyncio
async def test_run_realtime_session_propagates_session_update_failure(monkeypatch: Any) -> None:
    """A failed session.update must abort startup instead of looking like a clean session exit."""
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda default=OPENAI_DEFAULT_VOICE: "alloy")
    monkeypatch.setattr(rt_mod, "get_active_tool_specs", lambda _: [])

    class FakeSession:
        async def update(self, **_kw: Any) -> None:
            raise RuntimeError("invalid session config")

    class FakeConn:
        session = FakeSession()

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            return FakeConn()

    class FakeClient:
        def __init__(self) -> None:
            self.realtime = FakeRealtime()

    handler = rt_mod.OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = FakeClient()

    with pytest.raises(RuntimeError, match="invalid session config"):
        await handler._run_realtime_session()


@pytest.mark.asyncio
async def test_handler_uses_openai_sample_rate_for_openai_backend(monkeypatch: Any) -> None:
    """OpenAI backend should keep the 24 kHz realtime audio configuration."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")

    handler = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.input_sample_rate == 24000
    assert handler.output_sample_rate == 24000


# ---- Cost calculation tests ----


def _make_usage(
    audio_in: int | None = 0,
    text_in: int | None = 0,
    image_in: int | None = 0,
    audio_out: int | None = 0,
    text_out: int | None = 0,
    has_input: bool = True,
    has_output: bool = True,
) -> MagicMock:
    """Build a fake usage object matching the OpenAI response.usage shape."""
    usage = MagicMock()
    if has_input:
        inp = MagicMock()
        inp.audio_tokens = audio_in
        inp.text_tokens = text_in
        inp.image_tokens = image_in
        usage.input_token_details = inp
    else:
        usage.input_token_details = None
    if has_output:
        out = MagicMock()
        out.audio_tokens = audio_out
        out.text_tokens = text_out
        usage.output_token_details = out
    else:
        usage.output_token_details = None
    return usage


@pytest.mark.parametrize(
    "usage_kwargs, expect_positive",
    [
        # All token types present → positive cost
        ({"audio_in": 1000, "text_in": 2000, "image_in": 500, "audio_out": 800, "text_out": 300}, True),
        # All None tokens → must not crash
        ({"audio_in": None, "text_in": None, "image_in": None, "audio_out": None, "text_out": None}, False),
        # Mix of None and valid ints
        ({"audio_in": None, "text_in": 500, "image_in": None, "audio_out": 1000, "text_out": None}, True),
        # Missing input/output details entirely
        ({"has_input": False, "has_output": False}, False),
    ],
    ids=["normal", "all_none", "mixed", "missing_details"],
)
def test_compute_response_cost(usage_kwargs: dict[str, Any], expect_positive: bool) -> None:
    """Verify handler cost computation handles various token combinations without crashing."""
    usage = _make_usage(**usage_kwargs)
    handler = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    cost = handler._compute_response_cost(usage)
    if expect_positive:
        assert cost > 0
    else:
        assert cost == 0.0


# ---- Stress test: response.create rejection + retry ----


@pytest.mark.asyncio
async def test_response_sender_retries_when_active_response_error_uses_type_only(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """Retry active-response rejections even when the server omits ``error.code``.

    Some backends only populate ``error.type=conversation_already_has_active_response``.
    That should still take the retry path and must not be surfaced as a user-facing error.
    """
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(base_rt_mod, "_RESPONSE_REJECTION_RETRY_DELAY", 0.01)
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda default=OPENAI_DEFAULT_VOICE: "alloy")
    monkeypatch.setattr(rt_mod, "get_active_tool_specs", lambda _: [])

    class FakeError:
        def __init__(self, message: str) -> None:
            self.message = message
            self.code = None
            self.type = "conversation_already_has_active_response"
            self.event_id = None
            self.param = None

        def __repr__(self) -> str:
            return f"RealtimeError(message='{self.message}', type='{self.type}', code=None, event_id=None, param=None)"

    class FakeEvent:
        def __init__(self, etype: str, **kwargs: Any) -> None:
            self.type = etype
            for key, value in kwargs.items():
                setattr(self, key, value)

    event_queue: asyncio.Queue[FakeEvent | None] = asyncio.Queue()

    class FakeSession:
        async def update(self, **_kw: Any) -> None:
            pass

    class FakeInputAudioBuffer:
        async def append(self, **_kw: Any) -> None:
            pass

    class FakeItem:
        async def create(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeItem()

    class FakeResponseAPI:
        def __init__(self) -> None:
            self.call_count = 0

        async def create(self, **_kw: Any) -> None:
            self.call_count += 1
            if self.call_count == 1:
                event_queue.put_nowait(
                    FakeEvent(
                        "error",
                        error=FakeError("Cannot create response while another response is in progress."),
                    )
                )
            else:
                event_queue.put_nowait(FakeEvent("response.created"))
                event_queue.put_nowait(FakeEvent("response.done", response=MagicMock()))

        async def cancel(self, **_kw: Any) -> None:
            pass

    fake_response_api = FakeResponseAPI()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeInputAudioBuffer()
        conversation = FakeConversation()
        response = fake_response_api

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> FakeEvent:
            event = await event_queue.get()
            if event is None:
                raise StopAsyncIteration
            return event

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            return FakeConn()

    class FakeClient:
        def __init__(self) -> None:
            self.realtime = FakeRealtime()

    handler = rt_mod.OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = FakeClient()

    session_task = asyncio.create_task(handler._run_realtime_session())
    await asyncio.sleep(0)
    await handler._safe_response_create(instructions="req")
    await asyncio.sleep(0.1)
    await event_queue.put(None)
    await asyncio.wait_for(session_task, timeout=2.0)

    assert fake_response_api.call_count == 2
    assert not any(
        record.levelname == "ERROR" and "Realtime error" in record.getMessage() for record in caplog.records
    )
    assert any("worker will retry after active response finishes" in record.getMessage() for record in caplog.records)
    queued_outputs = []
    while not handler.output_queue.empty():
        queued_outputs.append(handler.output_queue.get_nowait())
    queued_messages = [
        message
        for output in queued_outputs
        if isinstance(output, AdditionalOutputs)
        for message in output.args
        if isinstance(message, dict)
    ]
    assert not any(str(message.get("content", "")).startswith("[error]") for message in queued_messages)


@pytest.mark.asyncio
async def test_response_sender_retries_on_active_response_rejection(monkeypatch: Any, caplog: Any) -> None:
    """Stress test: response.create rejection + retry via real event processing.

    Tool results (is_idle_tool_call=False) queue response.create calls via
    _safe_response_create.  When the server rejects some with
    ``conversation_already_has_active_response``, the error event flows through
    the event handler and _response_sender_loop retries the rejected request.

    The full _run_realtime_session event loop runs so that the error-handling
    code path (setting _last_response_rejected) is exercised by real event
    processing, not mocked out.
    """
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(base_rt_mod, "_RESPONSE_REJECTION_RETRY_DELAY", 0.01)

    FakeCCE = type("FakeCCE", (Exception,), {})
    monkeypatch.setattr(base_rt_mod, "ConnectionClosedError", FakeCCE)
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda default=OPENAI_DEFAULT_VOICE: "alloy")
    monkeypatch.setattr(rt_mod, "get_active_tool_specs", lambda _: [])

    N_TOOL_RESULTS = 400
    REJECT_CALL_NUMBERS = {1, 3, 5, 10, 25, 50, 75, 100, 150, 200, 300, 399}
    EXPECTED_TOTAL_CALLS = N_TOOL_RESULTS + len(REJECT_CALL_NUMBERS)

    response_create_log: list[tuple[int, dict[str, Any]]] = []
    handler_ref: list[rt_mod.OpenaiRealtimeHandler] = []

    # ---- Fake event / error objects mirroring the OpenAI SDK shapes ----

    class FakeError:
        def __init__(self, message: str, code: str) -> None:
            self.message = message
            self.code = code
            self.type = "invalid_request_error"
            self.event_id = None
            self.param = None

        def __repr__(self) -> str:
            return (
                f"RealtimeError(message='{self.message}', type='{self.type}', "
                f"code='{self.code}', event_id=None, param=None)"
            )

    class FakeEvent:
        def __init__(self, etype: str, **kwargs: Any) -> None:
            self.type = etype
            for k, v in kwargs.items():
                setattr(self, k, v)

    event_queue: asyncio.Queue[FakeEvent | None] = asyncio.Queue()

    # ---- Fake connection components ----

    class FakeResponseAPI:
        """Mimics connection.response.

        Pushes server events into the shared event_queue so they flow
        through the real event-handling code.  Also guards the serialization
        invariant: every create() must arrive when no response is active.
        """

        def __init__(self) -> None:
            self._call_count = 0
            self._serialization_violations: list[int] = []

        async def create(self, **kwargs: Any) -> None:
            self._call_count += 1
            n = self._call_count
            response_create_log.append((n, kwargs))

            h = handler_ref[0]

            # Real backend rejects when a response is already active.
            if not h._response_done_event.is_set():
                self._serialization_violations.append(n)
                await event_queue.put(
                    FakeEvent(
                        "error",
                        error=FakeError(
                            message=(
                                f"Conversation already has an active response in "
                                f"progress: resp_fake{n}. Wait until the response "
                                f"is finished before creating a new one."
                            ),
                            code="conversation_already_has_active_response",
                        ),
                    )
                )
                await asyncio.sleep(0)
                await event_queue.put(FakeEvent("response.done", response=MagicMock()))
                return

            # Intentional rejections (simulating a race where another
            # response sneaks in right after our check).
            if n in REJECT_CALL_NUMBERS:
                await event_queue.put(
                    FakeEvent(
                        "error",
                        error=FakeError(
                            message=(
                                f"Conversation already has an active response in "
                                f"progress: resp_fake{n}. Wait until the response "
                                f"is finished before creating a new one."
                            ),
                            code="conversation_already_has_active_response",
                        ),
                    )
                )
                await asyncio.sleep(0)
            else:
                await event_queue.put(FakeEvent("response.created"))

            await event_queue.put(FakeEvent("response.done", response=MagicMock()))

        async def cancel(self, **_kw: Any) -> None:
            pass

    fake_response_api = FakeResponseAPI()

    class FakeSession:
        async def update(self, **_kw: Any) -> None:
            pass

    class FakeInputAudioBuffer:
        async def append(self, **_kw: Any) -> None:
            pass

    class FakeItem:
        async def create(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeItem()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeInputAudioBuffer()
        conversation = FakeConversation()
        response = fake_response_api

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> FakeEvent:
            event = await event_queue.get()
            if event is None:  # sentinel → end iteration
                raise StopAsyncIteration
            return event

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            return FakeConn()

    class FakeClient:
        def __init__(self, **_kw: Any) -> None:
            self.realtime = FakeRealtime()

    monkeypatch.setattr(rt_mod, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")

    # Patch dispatch_tool_call so tools complete with a result.
    async def _fake_dispatch(tool_name: str, args_json: str, deps: Any, **_kw: Any) -> dict[str, Any]:
        await asyncio.sleep(random.uniform(0.3, 0.5))
        return {"ok": True, "tool": tool_name}

    monkeypatch.setattr(btm_mod, "dispatch_tool_call", _fake_dispatch)

    # ---- Build handler and start the full realtime session ----

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler_ref.append(handler)

    asyncio.create_task(handler.start_up())

    # ---- Start tools via the real BackgroundToolManager pipeline ----
    # start_tool → _run_tool → notification queue → listener → _handle_tool_result

    for i in range(N_TOOL_RESULTS):
        await handler.tool_manager.start_tool(
            call_id=f"call_{i}",
            tool_call_routine=ToolCallRoutine(
                tool_name="test_tool",
                args_json_str=f'{{"index": {i}}}',
                deps=deps,
            ),
            is_idle_tool_call=False,
        )

    # Wait until spawned tool tasks, the listener, and the sender have drained.
    # This stress test queues hundreds of serialized response.create calls; a
    # condition-based wait avoids racing slower CI runners while still failing
    # promptly if the sender stops making progress.
    deadline = asyncio.get_event_loop().time() + 25.0
    while fake_response_api._call_count < EXPECTED_TOTAL_CALLS and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)

    # ---- Tear down ----

    await event_queue.put(None)  # sentinel stops event iteration

    await handler.shutdown()

    # ---- Assertions ----

    # Serialization: every response.create() must have been called only when
    # no response was in-flight (_response_done_event was set).  Any violation
    # means the sender fired a new request before the previous one finished.
    assert fake_response_api._serialization_violations == [], (
        f"response.create() was called while a response was still active on "
        f"call(s) {fake_response_api._serialization_violations}"
    )

    # Total response.create() calls = tool results + retries for rejected ones
    assert fake_response_api._call_count == EXPECTED_TOTAL_CALLS, (
        f"Expected {EXPECTED_TOTAL_CALLS} response.create calls "
        f"({N_TOOL_RESULTS} results + {len(REJECT_CALL_NUMBERS)} retries), "
        f"got {fake_response_api._call_count}"
    )

    # The error event handler must have set _last_response_rejected for each
    # rejection (the log message comes from the event handler code path).
    rejection_logs = [r for r in caplog.records if "worker will retry" in getattr(r, "msg", "")]
    assert len(rejection_logs) == len(REJECT_CALL_NUMBERS), (
        f"Expected {len(REJECT_CALL_NUMBERS)} rejection entries from error handler, got {len(rejection_logs)}"
    )

    # The sender loop must have retried after each rejection.
    retry_logs = [r for r in caplog.records if "response.create was rejected; retrying" in getattr(r, "msg", "")]
    assert len(retry_logs) == len(REJECT_CALL_NUMBERS), (
        f"Expected {len(REJECT_CALL_NUMBERS)} retry entries from sender loop, got {len(retry_logs)}"
    )


# ---- Response creation timeout guard tests ----


@pytest.mark.asyncio
async def test_response_sender_loop_times_out_waiting_for_response_done(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """If response.done is never received the sender loop should time out.

    Rather than hang forever, it force-sets the event and moves on.
    """
    caplog.set_level(logging.DEBUG)

    monkeypatch.setattr(base_rt_mod, "_RESPONSE_DONE_TIMEOUT", 0.3)

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)

    create_count = 0

    class FakeResponse:
        async def create(self, **_kw: Any) -> None:
            nonlocal create_count
            create_count += 1
            # Simulate response.created clearing the event, but never
            # send response.done (so the event stays cleared forever).
            handler._response_done_event.clear()

        async def cancel(self, **_kw: Any) -> None:
            pass

    fake_conn = MagicMock()
    fake_conn.response = FakeResponse()
    handler.connection = fake_conn

    # Queue two requests
    await handler._safe_response_create(instructions="req1")
    await handler._safe_response_create(instructions="req2")

    sender_task = asyncio.create_task(handler._response_sender_loop())

    # Give enough time for both requests to time out (0.3s each + margin)
    await asyncio.sleep(1.5)

    handler.connection = None  # signal the loop to exit
    handler._response_done_event.set()
    await asyncio.wait_for(sender_task, timeout=2.0)

    assert create_count == 2, f"Expected 2 response.create calls, got {create_count}"

    timeout_logs = [r for r in caplog.records if "Timed out waiting for response.done" in r.getMessage()]
    assert len(timeout_logs) == 2, f"Expected 2 timeout warnings, got {len(timeout_logs)}"


@pytest.mark.asyncio
async def test_response_sender_loop_times_out_waiting_for_previous_response(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """If a previous response never completes, the pre-condition wait times out.

    It should force-set the event and proceed to send.
    """
    caplog.set_level(logging.DEBUG)

    monkeypatch.setattr(base_rt_mod, "_RESPONSE_DONE_TIMEOUT", 0.3)

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)

    # Pretend a response is already in-flight (event cleared)
    handler._response_done_event.clear()

    created = asyncio.Event()

    class FakeResponse:
        async def create(self, **_kw: Any) -> None:
            # Immediately complete the response cycle so the loop can finish
            handler._response_done_event.set()
            created.set()

        async def cancel(self, **_kw: Any) -> None:
            pass

    fake_conn = MagicMock()
    fake_conn.response = FakeResponse()
    handler.connection = fake_conn

    await handler._safe_response_create(instructions="waiting_req")

    sender_task = asyncio.create_task(handler._response_sender_loop())

    # Wait for the request to be sent (after timing out on the pre-condition)
    await asyncio.wait_for(created.wait(), timeout=2.0)

    handler.connection = None
    handler._response_done_event.set()
    await asyncio.wait_for(sender_task, timeout=2.0)

    timeout_logs = [r for r in caplog.records if "Timed out waiting for previous response" in r.getMessage()]
    assert len(timeout_logs) == 1, f"Expected 1 pre-condition timeout warning, got {len(timeout_logs)}"


@pytest.mark.asyncio
async def test_openai_excludes_head_tracking_when_no_head_tracker(monkeypatch: Any) -> None:
    """head_tracking tool must not appear in OpenAI session config when head_tracker is not active."""
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda default=None: "alloy")

    # mock ALL_TOOL_SPECS to include at least head_tracking and one other tool, to verify that only head_tracking is excluded, not all tools
    monkeypatch.setattr(
        ct_mod,
        "ALL_TOOL_SPECS",
        [
            {"type": "function", "name": "head_tracking", "description": "head_tracking", "parameters": {}},
            {"type": "function", "name": "fake_tool", "description": "fake_tool", "parameters": {}},
        ],
    )

    session_kwargs: dict = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            session_kwargs["session"] = kwargs.get("session")

    class FakeInputAudioBuffer:
        async def append(self, **_kw: Any) -> None:
            pass

    class FakeItem:
        async def create(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeItem()

    class FakeResponse:
        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            pass

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeInputAudioBuffer()
        conversation = FakeConversation()
        response = FakeResponse()

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            return FakeConn()

    class FakeClient:
        def __init__(self) -> None:
            self.realtime = FakeRealtime()

    # case 1: no camera at all, --no-camera flag passed
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_worker=None)
    handler = OpenaiRealtimeHandler(deps)
    handler.client = FakeClient()
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    session_tools = session_kwargs.get("session", {}).get("tools", [])
    tool_names = [t["name"] for t in session_tools]
    assert "head_tracking" not in tool_names, "case 1 failed: camera_worker=None"
    assert "fake_tool" in tool_names, "case 1 failed: a non-head-tracking tool was unexpectedly excluded"

    # case 2: camera is running but --head-tracker flag was not passed
    session_kwargs.clear()
    camera_worker = MagicMock()
    camera_worker.head_tracker = None
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_worker=camera_worker)
    handler = OpenaiRealtimeHandler(deps)
    handler.client = FakeClient()
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    session_tools = session_kwargs.get("session", {}).get("tools", [])
    tool_names = [t["name"] for t in session_tools]
    assert "head_tracking" not in tool_names, "case 2 failed: camera_worker.head_tracker=None"
    assert "fake_tool" in tool_names, "case 2 failed: a non-head-tracking tool was unexpectedly excluded"
