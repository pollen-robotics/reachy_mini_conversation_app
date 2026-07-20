import json
import time
import base64
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.conversation_handler as conv_mod
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, ToolNotification


HF_DEFAULT_VOICE = get_default_voice()


class _FakeEvent:
    """A minimal realtime event: a `type` plus arbitrary attributes."""

    def __init__(self, event_type: str, **fields: Any) -> None:
        """Store the event type and any extra attributes."""
        self.type = event_type
        self.__dict__.update(fields)


def _make_fake_realtime_client(
    *,
    events: tuple[_FakeEvent, ...] = (),
    captured_update: dict[str, Any] | None = None,
    captured_connect: dict[str, Any] | None = None,
) -> Any:
    """Build a fake AsyncOpenAI-shaped client whose realtime session yields `events`.

    When given, `captured_update`/`captured_connect` record the kwargs passed to
    `session.update(...)` / `realtime.connect(...)`.
    """

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            if captured_update is not None:
                captured_update.update(kwargs)

    class FakeNoop:
        async def append(self, **_kw: Any) -> None:
            pass

        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeNoop()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeNoop()
        conversation = FakeConversation()
        response = FakeNoop()

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

        async def __anext__(self) -> _FakeEvent:
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **kwargs: Any) -> FakeConn:
            if captured_connect is not None:
                captured_connect.update(kwargs)
            return FakeConn()

    class FakeClient:
        realtime = FakeRealtime()

    return FakeClient()


def _fake_openai_client(captured_kwargs: dict[str, Any]) -> type:
    """Return a fake AsyncOpenAI class that records its constructor kwargs."""

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    return FakeClient


def _fake_allocator(connect_url: str, posts: list[tuple[str, dict[str, str] | None]]) -> type:
    """Return a fake httpx.AsyncClient whose POST records (url, headers) and returns `connect_url`."""

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"session_id": "session-123", "connect_url": connect_url}

    class FakeAsyncClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def post(self, url: str, headers: dict[str, str] | None = None) -> FakeResponse:
            posts.append((url, headers))
            return FakeResponse()

    return FakeAsyncClient


def _plain_handler() -> HuggingFaceRealtimeHandler:
    return HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


def _session_handler(monkeypatch: Any, events: tuple[_FakeEvent, ...]) -> HuggingFaceRealtimeHandler:
    """Build a handler wired to a fake realtime client yielding `events`, ready to run a session."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    handler = _plain_handler()
    handler.client = _make_fake_realtime_client(events=events)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    return handler


def _drain(handler: HuggingFaceRealtimeHandler) -> list[Any]:
    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    return items


def _messages(items: list[Any]) -> list[dict[str, Any]]:
    return [item.args[0] for item in items if isinstance(item, AdditionalOutputs)]


@pytest.mark.asyncio
async def test_partial_transcription_uses_latest_snapshot(monkeypatch: Any) -> None:
    """Partial transcription snapshots should replace older snapshots for the same item."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey"),
            _FakeEvent(
                "conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey, how are you?"
            ),
        )
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    assert handler.input_transcript_chunks_by_item.item_id == "item-1"
    assert handler.input_transcript_chunks_by_item.deltas == ["Hey, how are you?"]


@pytest.mark.asyncio
async def test_emit_skips_idle_signal_while_response_active(monkeypatch: Any) -> None:
    """Idle tools should not trigger while a response is still active."""
    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    handler = HuggingFaceRealtimeHandler(deps)
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler._response_done_event.clear()

    send_idle_signal = AsyncMock()
    monkeypatch.setattr(handler, "send_idle_signal", send_idle_signal)
    monkeypatch.setattr(conv_mod, "wait_for_item", AsyncMock(return_value=None))

    result = await handler.emit()

    assert result is None
    send_idle_signal.assert_not_awaited()


@pytest.mark.asyncio
async def test_parallel_tool_calls_trigger_single_response(monkeypatch: Any) -> None:
    """Parallel tool calls in one turn should yield one response, not one per completed tool."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    handler._in_flight_tool_calls = {"call_a", "call_b"}

    def _completed(call_id: str) -> ToolNotification:
        return ToolNotification(
            id=call_id,
            tool_name="test__parallel_probe",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"ok": True},
        )

    await handler._handle_tool_result(_completed("call_a"))
    assert create.await_count == 0

    await handler._handle_tool_result(_completed("call_b"))
    assert create.await_count == 1


def test_handler_uses_hf_startup_voice_at_startup(monkeypatch: Any) -> None:
    """Hugging Face startup should restore persisted HF voices."""
    handler = HuggingFaceRealtimeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="Aiden",
    )

    assert handler.get_current_voice() == "Aiden"


def test_handler_ignores_unsupported_hf_profile_voice(monkeypatch: Any) -> None:
    """Unsupported profile voices should not be sent to the Hugging Face backend."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "cedar")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == HF_DEFAULT_VOICE
    session = handler._get_session_config([])
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_handler_normalizes_hf_voice_case(monkeypatch: Any) -> None:
    """Lowercase Hugging Face speaker names should resolve to the curated UI value."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "serena")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == "Serena"


@pytest.mark.asyncio
async def test_run_realtime_session_uses_default_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Use the backend default speaker when no profile voice is selected for the hf LB."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_update=captured_update)

    await handler._run_realtime_session()

    session = captured_update["session"]
    # HF at 16 kHz passes None so the backend uses its optimal default (16 kHz).
    assert session["audio"]["input"]["format"]["rate"] is None
    assert session["audio"]["output"]["format"]["rate"] is None
    assert session["audio"]["input"]["transcription"]["language"] == "en"
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_huggingface_session_uses_configured_transcription_language(monkeypatch: Any) -> None:
    """Hugging Face realtime sessions should forward the configured transcription language."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "zh")
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    session = handler._get_session_config([])

    assert session["audio"]["input"]["transcription"]["language"] == "zh"


@pytest.mark.asyncio
async def test_run_realtime_session_passes_allocated_session_query(monkeypatch: Any) -> None:
    """Hugging Face sessions must forward the allocated session token to the websocket connect call."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    captured_connect: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_connect=captured_connect)
    handler._realtime_connect_query = {"session_token": "abc123"}

    await handler._run_realtime_session()

    assert "model" not in captured_connect
    assert captured_connect["extra_query"] == {"session_token": "abc123"}


@pytest.mark.asyncio
async def test_build_realtime_client_uses_direct_hf_ws_url(monkeypatch: Any) -> None:
    """Hugging Face direct websocket mode should bypass the session allocator."""
    client_kwargs: dict[str, Any] = {}

    def _no_allocator(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("session allocator should not be called in direct websocket mode")

    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _no_allocator)
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_TOKEN", None)
    monkeypatch.setattr(
        config,
        "HF_REALTIME_WS_URL",
        "ws://127.0.0.1:8765/v1/realtime?session_token=abc123&model=ignored-by-sdk",
    )

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert client_kwargs["api_key"] == "DUMMY"
    assert client_kwargs["base_url"] == "http://127.0.0.1:8765/v1"
    assert client_kwargs["websocket_base_url"] == "ws://127.0.0.1:8765/v1"
    assert handler._realtime_connect_query == {"session_token": "abc123"}


@pytest.mark.parametrize(
    ("hf_token", "expected_header", "expected_api_key"),
    [
        ("hf-secret", {"Authorization": "Bearer hf-secret"}, "hf-secret"),
        (None, None, "DUMMY"),
    ],
)
@pytest.mark.asyncio
async def test_build_realtime_client_deployed_allocates_with_hf_token_only(
    monkeypatch: Any,
    hf_token: str | None,
    expected_header: dict[str, str] | None,
    expected_api_key: str,
) -> None:
    """Deployed mode allocates via the session URL, authenticating with HF_TOKEN only (never an OpenAI key)."""
    client_kwargs: dict[str, Any] = {}
    posts: list[tuple[str, dict[str, str] | None]] = []
    connect_url = "wss://hf.example.test/v1/realtime?session_token=allocated"
    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _fake_allocator(connect_url, posts))
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    # A stale local URL must be ignored in deployed mode.
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")
    monkeypatch.setattr(config, "HF_TOKEN", hf_token)

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert posts == [("https://lb.example.test/session", expected_header)]
    assert client_kwargs["api_key"] == expected_api_key
    assert client_kwargs["base_url"] == "https://hf.example.test/v1"
    assert client_kwargs["websocket_base_url"] == "wss://hf.example.test/v1"
    assert handler._realtime_connect_query == {"session_token": "allocated"}


@pytest.mark.asyncio
async def test_apply_personality_uses_selected_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Live personality updates should honor the selected Qwen CustomVoice speaker."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "new instructions")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Serena")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    monkeypatch.setattr(handler, "_restart_session", AsyncMock(return_value=None))

    result = await handler.apply_personality("mars_rover")

    assert "restarted realtime session" in result.lower()
    session = captured_update["session"]
    assert session["instructions"] == "new instructions"
    assert session["audio"]["output"]["voice"] == "Serena"


@pytest.mark.asyncio
async def test_change_voice_updates_live_hf_session_without_restart(monkeypatch: Any) -> None:
    """Changing Hugging Face voice should update the active session in place."""
    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    restart = AsyncMock(return_value=None)
    monkeypatch.setattr(handler, "_restart_session", restart)

    result = await handler.change_voice("Serena")

    assert result == "Voice changed to Serena."
    assert handler.get_current_voice() == "Serena"
    restart.assert_not_awaited()
    session = captured_update["session"]
    assert session["audio"]["output"]["voice"] == "Serena"


@pytest.mark.asyncio
async def test_run_session_decodes_audio_delta(monkeypatch: Any) -> None:
    """An audio delta is base64-decoded and enqueued as an (rate, int16) frame."""
    pcm = np.array([1, 2, 3, 4], dtype=np.int16)
    delta = base64.b64encode(pcm.tobytes()).decode("utf-8")
    handler = _session_handler(monkeypatch, (_FakeEvent("response.output_audio.delta", delta=delta),))

    await handler._run_realtime_session()

    frames = [item for item in _drain(handler) if isinstance(item, tuple)]
    assert len(frames) == 1
    rate, array = frames[0]
    assert rate == handler.SAMPLE_RATE
    np.testing.assert_array_equal(array.reshape(-1), pcm)


@pytest.mark.asyncio
async def test_run_session_emits_completed_transcript(monkeypatch: Any) -> None:
    """A non-empty completed transcript is enqueued as a user message and stops listening."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("conversation.item.input_audio_transcription.completed", transcript="Hello there"),),
    )

    await handler._run_realtime_session()

    assert {"role": "user", "content": "Hello there"} in _messages(_drain(handler))
    handler.deps.movement_manager.set_listening.assert_any_call(False)


@pytest.mark.asyncio
async def test_run_session_skips_empty_completed_transcript(monkeypatch: Any) -> None:
    """An empty completed transcript is ignored but still stops listening."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("conversation.item.input_audio_transcription.completed", transcript="   "),),
    )

    await handler._run_realtime_session()

    assert [msg for msg in _messages(_drain(handler)) if msg.get("role") == "user"] == []
    handler.deps.movement_manager.set_listening.assert_any_call(False)


@pytest.mark.asyncio
async def test_run_session_generic_error_is_surfaced(monkeypatch: Any) -> None:
    """A generic realtime error is forwarded to the UI as an assistant message."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("error", error=SimpleNamespace(message="bad", code="server_error")),),
    )

    await handler._run_realtime_session()

    assert {"role": "assistant", "content": "[error] bad"} in _messages(_drain(handler))


@pytest.mark.asyncio
async def test_run_session_commit_empty_error_is_internal(monkeypatch: Any) -> None:
    """A commit-empty error stops listening and is not surfaced to the UI."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("error", error=SimpleNamespace(message="empty", code="input_audio_buffer_commit_empty")),),
    )

    await handler._run_realtime_session()

    assert _messages(_drain(handler)) == []
    handler.deps.movement_manager.set_listening.assert_any_call(False)


@pytest.mark.asyncio
async def test_run_session_active_response_error_flags_rejection(monkeypatch: Any) -> None:
    """A duplicate-response error is recorded for the sender worker, not surfaced."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("error", error=SimpleNamespace(message="x", code="conversation_already_has_active_response")),),
    )

    await handler._run_realtime_session()

    assert handler._last_response_rejected is True
    assert _messages(_drain(handler)) == []


@pytest.mark.asyncio
async def test_run_session_dispatches_valid_tool_call(monkeypatch: Any) -> None:
    """A valid function call starts a background tool and announces it."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("response.function_call_arguments.done", name="dance", arguments="{}", call_id="c1"),),
    )
    start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="dance-1"))
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    await handler._run_realtime_session()

    start_tool.assert_awaited_once()
    assert any("Used tool dance" in msg["content"] for msg in _messages(_drain(handler)))


@pytest.mark.asyncio
async def test_run_session_ignores_invalid_tool_call(monkeypatch: Any) -> None:
    """A function call with a non-string name is ignored."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("response.function_call_arguments.done", name=None, arguments="{}", call_id="c1"),),
    )
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    await handler._run_realtime_session()

    start_tool.assert_not_awaited()
    assert not any("Used tool" in msg["content"] for msg in _messages(_drain(handler)))


def test_sanitize_tool_result_strips_camera_image() -> None:
    """Camera results drop the raw image bytes and flag that an image was attached."""
    sanitized = HuggingFaceRealtimeHandler._sanitize_tool_result_for_model("camera", {"b64_im": "x", "seen": "cat"})
    assert sanitized == {"seen": "cat", "image_attached": True}


def test_sanitize_tool_result_passthrough_for_non_camera() -> None:
    """Non-camera results are returned unchanged."""
    result = {"status": "ok"}
    assert HuggingFaceRealtimeHandler._sanitize_tool_result_for_model("dance", result) == result


class _FakeConnection:
    def __init__(self) -> None:
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=AsyncMock()))


@pytest.mark.asyncio
async def test_handle_tool_result_error_is_reported_and_prompts_response() -> None:
    """A failed tool reports its error to the UI and requests a spoken follow-up."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    note = ToolNotification(id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.FAILED, error="boom")

    await handler._handle_tool_result(note)

    assert {"role": "assistant", "content": json.dumps({"error": "boom"})} in _messages(_drain(handler))
    assert not handler._pending_responses.empty()
    handler.connection.conversation.item.create.assert_awaited()


@pytest.mark.asyncio
async def test_handle_tool_result_no_connection_is_dropped() -> None:
    """With no connection the tool result is dropped without enqueuing anything."""
    handler = _plain_handler()
    handler.connection = None
    note = ToolNotification(
        id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.COMPLETED, result={"ok": 1}
    )

    await handler._handle_tool_result(note)

    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_handle_tool_result_missing_result_reports_error() -> None:
    """A tool that returns neither result nor error is reported as a failure."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    note = ToolNotification(id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.COMPLETED)

    await handler._handle_tool_result(note)

    expected = json.dumps({"error": "No result returned from tool execution"})
    assert {"role": "assistant", "content": expected} in _messages(_drain(handler))


@pytest.mark.asyncio
async def test_wait_for_response_done_returns_true_when_set() -> None:
    """The tool-result gate passes immediately when no response is active."""
    handler = _plain_handler()
    handler._response_done_event.set()
    assert await handler._wait_for_response_done_before_tool_result() is True


@pytest.mark.asyncio
async def test_wait_for_response_done_times_out(monkeypatch: Any) -> None:
    """The tool-result gate fails when the active response never finishes."""
    monkeypatch.setattr(hf_mod, "_RESPONSE_DONE_TIMEOUT", 0.01)
    handler = _plain_handler()
    handler._response_done_event.clear()
    assert await handler._wait_for_response_done_before_tool_result() is False


@pytest.mark.asyncio
async def test_emit_debounced_partial_emits_current_snapshot() -> None:
    """A partial transcript is emitted when it is still the latest snapshot."""
    handler = _plain_handler()
    handler.partial_debounce_delay = 0
    handler.input_transcript_chunks_by_item.item_id = "item-1"
    handler.input_transcript_chunks_by_item.deltas = ["hi"]

    await handler._emit_debounced_partial("hi", "item-1", 0)

    assert {"role": "user_partial", "content": "hi"} in _messages(_drain(handler))


@pytest.mark.asyncio
async def test_emit_debounced_partial_skips_stale_snapshot() -> None:
    """A partial transcript is dropped once a newer delta has arrived."""
    handler = _plain_handler()
    handler.partial_debounce_delay = 0
    handler.input_transcript_chunks_by_item.item_id = "item-1"
    handler.input_transcript_chunks_by_item.deltas = ["hi", "hi there"]

    await handler._emit_debounced_partial("hi", "item-1", 0)

    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_emit_runs_idle_behavior_when_due(monkeypatch: Any) -> None:
    """Emit runs the idle behavior once the idle thresholds are exceeded and the robot is idle."""
    handler = _plain_handler()
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler.last_idle_behavior_time = handler.last_activity_time
    handler.deps.movement_manager.is_idle.return_value = True
    handler._response_done_event.set()
    send_idle_signal = AsyncMock()
    monkeypatch.setattr(handler, "send_idle_signal", send_idle_signal)
    monkeypatch.setattr(conv_mod, "wait_for_item", AsyncMock(return_value=None))

    before = time.monotonic()
    await handler.emit()

    send_idle_signal.assert_awaited_once()
    assert handler.last_idle_behavior_time >= before


@pytest.mark.asyncio
async def test_emit_keeps_idle_timer_when_idle_behavior_fails(monkeypatch: Any) -> None:
    """A failing idle behavior returns None and does not advance the idle-behavior timer."""
    handler = _plain_handler()
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler.last_idle_behavior_time = handler.last_activity_time
    handler.deps.movement_manager.is_idle.return_value = True
    handler._response_done_event.set()
    monkeypatch.setattr(handler, "send_idle_signal", AsyncMock(side_effect=RuntimeError("closed")))

    previous = handler.last_idle_behavior_time
    result = await handler.emit()

    assert result is None
    assert handler.last_idle_behavior_time == previous


def test_mark_activity_swallows_observer_errors() -> None:
    """The activity observer is notified, and a raising observer is ignored."""
    handler = _plain_handler()
    seen: list[str] = []
    handler.set_activity_observer(seen.append)
    handler._mark_activity("user_speech_started")
    assert seen == ["user_speech_started"]

    def _raise(_reason: str) -> None:
        raise RuntimeError("observer boom")

    handler.set_activity_observer(_raise)
    handler._mark_activity("later")  # must not raise


@pytest.mark.asyncio
async def test_send_idle_signal_noop_without_connection(monkeypatch: Any) -> None:
    """No idle tool runs when there is no active connection."""
    handler = _plain_handler()
    handler.connection = None
    start_idle = AsyncMock()
    monkeypatch.setattr(conv_mod, "start_idle_tool_call", start_idle)

    await handler.send_idle_signal(200.0)

    start_idle.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_idle_signal_dispatches_idle_tool(monkeypatch: Any) -> None:
    """With an active connection the locally selected idle tool is dispatched."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    monkeypatch.setattr(conv_mod, "get_tool_specs", lambda: [{"name": "dance"}])
    start_idle = AsyncMock()
    monkeypatch.setattr(conv_mod, "start_idle_tool_call", start_idle)

    await handler.send_idle_signal(200.0)

    start_idle.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_without_connection_is_dropped() -> None:
    """A frame received before the connection opens is dropped."""
    handler = _plain_handler()
    handler.connection = None
    await handler.receive((16000, np.zeros(4, dtype=np.int16)))  # must not raise


@pytest.mark.asyncio
async def test_receive_ignores_empty_frame() -> None:
    """An empty audio frame is not forwarded to the realtime buffer."""
    handler = _plain_handler()
    append = AsyncMock()
    handler.connection = SimpleNamespace(input_audio_buffer=SimpleNamespace(append=append))
    await handler.receive((16000, np.array([], dtype=np.int16)))
    append.assert_not_awaited()


@pytest.mark.asyncio
async def test_receive_downmixes_stereo_and_forwards() -> None:
    """A stereo frame is reduced to mono and appended to the realtime buffer."""
    handler = _plain_handler()
    append = AsyncMock()
    handler.connection = SimpleNamespace(input_audio_buffer=SimpleNamespace(append=append))
    stereo = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.int16)
    await handler.receive((16000, stereo))
    append.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_session_speech_started_clears_queue_and_listens(monkeypatch: Any) -> None:
    """User speech start clears the playback queue and enters listening mode."""
    handler = _session_handler(monkeypatch, (_FakeEvent("input_audio_buffer.speech_started"),))
    clear_queue = MagicMock()
    handler._clear_queue = clear_queue

    await handler._run_realtime_session()

    clear_queue.assert_called_once()
    handler.deps.movement_manager.set_listening.assert_any_call(True)


@pytest.mark.asyncio
async def test_run_session_response_lifecycle_toggles_done_event(monkeypatch: Any) -> None:
    """response.created clears the done gate and response.done sets it again."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("response.created"), _FakeEvent("response.done")),
    )

    await handler._run_realtime_session()

    assert handler._response_done_event.is_set()
    handler.deps.movement_manager.set_speaking.assert_any_call(True)
    handler.deps.movement_manager.set_speaking.assert_any_call(False)
