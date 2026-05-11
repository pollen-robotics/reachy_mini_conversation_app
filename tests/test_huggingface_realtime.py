import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.base_realtime as base_rt_mod
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.config import HF_BACKEND, config, get_default_voice_for_backend
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


HF_DEFAULT_VOICE = get_default_voice_for_backend(HF_BACKEND)


def _make_usage(
    audio_in: int | None = 100,
    text_in: int | None = 200,
    image_in: int | None = 300,
    audio_out: int | None = 400,
    text_out: int | None = 500,
    has_input: bool = True,
    has_output: bool = True,
) -> MagicMock:
    """Build a fake usage object matching the OpenAI-compatible response.usage shape."""
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


@pytest.mark.asyncio
async def test_partial_transcription_uses_latest_snapshot(monkeypatch: Any) -> None:
    """Partial transcription snapshots should replace older snapshots for the same item."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_active_tool_specs", lambda _: [])

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
                    FakeEvent("conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey"),
                    FakeEvent(
                        "conversation.item.input_audio_transcription.delta",
                        item_id="item-1",
                        delta="Hey, how are you?",
                    ),
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
    handler = HuggingFaceRealtimeHandler(deps)
    fake_client = FakeClient()
    handler.client = fake_client

    start_up = MagicMock()
    shutdown = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_up", start_up)
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", shutdown)

    await handler._run_realtime_session()

    assert handler.input_transcript_chunks_by_item.item_id == "item-1"
    assert handler.input_transcript_chunks_by_item.deltas == ["Hey, how are you?"]


@pytest.mark.asyncio
async def test_output_audio_delta_passes_output_sample_rate_to_head_wobbler(monkeypatch: Any) -> None:
    """Assistant audio deltas should propagate the realtime output sample rate to the head wobbler."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_active_tool_specs", lambda _: [])
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")

    audio_delta = "AAABAAIAAwA="

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
                    FakeEvent("response.created"),
                    FakeEvent("response.output_audio.delta", delta=audio_delta),
                    FakeEvent("response.output_audio.done"),
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

    head_wobbler = MagicMock()
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        head_wobbler=head_wobbler,
    )
    handler = HuggingFaceRealtimeHandler(deps, gradio_mode=True)
    handler.client = FakeClient()

    start_up = MagicMock()
    shutdown = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_up", start_up)
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", shutdown)

    await handler._run_realtime_session()

    head_wobbler.feed_pcm.assert_called_once()
    assert head_wobbler.feed_pcm.call_args.args[1] == handler.output_sample_rate
    head_wobbler.request_reset_after_current_audio.assert_called_once()


@pytest.mark.asyncio
async def test_emit_skips_idle_signal_while_response_active(monkeypatch: Any) -> None:
    """Idle tools should not trigger while a response is still active."""
    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    handler = HuggingFaceRealtimeHandler(deps)
    handler.last_activity_time = asyncio.get_running_loop().time() - 60.0
    handler._response_done_event.clear()

    send_idle_signal = AsyncMock()
    monkeypatch.setattr(handler, "send_idle_signal", send_idle_signal)
    monkeypatch.setattr(base_rt_mod, "wait_for_item", AsyncMock(return_value=None))

    result = await handler.emit()

    assert result is None
    send_idle_signal.assert_not_awaited()


def test_handler_uses_hf_startup_voice_at_startup(monkeypatch: Any) -> None:
    """Hugging Face startup should restore persisted HF voices."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")

    handler = HuggingFaceRealtimeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="Aiden",
    )

    assert handler.get_current_voice() == "Aiden"


def test_handler_ignores_unsupported_hf_profile_voice(monkeypatch: Any) -> None:
    """OpenAI/Gemini profile voices should not be sent to the Hugging Face backend."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "cedar")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == HF_DEFAULT_VOICE
    session = handler._get_session_config([])
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_handler_normalizes_hf_voice_case(monkeypatch: Any) -> None:
    """Lowercase Hugging Face speaker names should resolve to the curated UI value."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "serena")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == "Serena"


@pytest.mark.asyncio
async def test_start_up_hf_gradio_does_not_wait_for_api_key(monkeypatch: Any) -> None:
    """Hugging Face backend should not wait for gradio key input."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-openai-secret")

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = hf_mod.HuggingFaceRealtimeHandler(deps, gradio_mode=True)

    build_client = AsyncMock(return_value=MagicMock())
    run_realtime_session = AsyncMock(return_value=None)
    wait_for_args = AsyncMock(side_effect=AssertionError("wait_for_args should not be called"))

    monkeypatch.setattr(handler, "_build_realtime_client", build_client)
    monkeypatch.setattr(handler, "_run_realtime_session", run_realtime_session)
    monkeypatch.setattr(handler, "wait_for_args", wait_for_args)

    await handler.start_up()

    wait_for_args.assert_not_awaited()
    build_client.assert_awaited_once_with()
    run_realtime_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_realtime_session_uses_default_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Use the backend default speaker when no profile voice is selected for the hf LB."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_active_tool_specs", lambda _: [])
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

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

        async def __aexit__(self, *_args: Any) -> bool:
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

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = HuggingFaceRealtimeHandler(deps)
    fake_client = FakeClient()
    handler.client = fake_client

    await handler._run_realtime_session()

    session = captured_update["session"]
    # HF at 16 kHz passes None so the backend uses its optimal default (16 kHz).
    assert session["audio"]["input"]["format"]["rate"] is None
    assert session["audio"]["output"]["format"]["rate"] is None
    output = session["audio"]["output"]
    assert output["voice"] == HF_DEFAULT_VOICE


@pytest.mark.asyncio
async def test_run_realtime_session_passes_allocated_session_query(monkeypatch: Any) -> None:
    """Hugging Face sessions must forward the allocated session token to the websocket connect call."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_active_tool_specs", lambda _: [])

    captured_connect: dict[str, Any] = {}

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

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **kwargs: Any) -> FakeConn:
            captured_connect.update(kwargs)
            return FakeConn()

    class FakeClient:
        def __init__(self) -> None:
            self.realtime = FakeRealtime()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    fake_client = FakeClient()
    handler.client = fake_client
    handler._realtime_connect_query = {"session_token": "abc123"}

    await handler._run_realtime_session()

    assert "model" not in captured_connect
    assert captured_connect["extra_query"] == {"session_token": "abc123"}


@pytest.mark.asyncio
async def test_build_realtime_client_uses_direct_hf_ws_url(monkeypatch: Any) -> None:
    """Hugging Face direct websocket mode should bypass the session allocator."""
    captured_client_kwargs: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_client_kwargs.update(kwargs)

    def _unexpected_async_client(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("session allocator should not be called in direct websocket mode")

    monkeypatch.setattr(hf_mod, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _unexpected_async_client)
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setattr(config, "HF_TOKEN", None)
    monkeypatch.setattr(
        config,
        "HF_REALTIME_WS_URL",
        "ws://127.0.0.1:8765/v1/realtime?session_token=abc123&model=ignored-by-sdk",
    )

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert captured_client_kwargs["api_key"] == "DUMMY"
    assert captured_client_kwargs["base_url"] == "http://127.0.0.1:8765/v1"
    assert captured_client_kwargs["websocket_base_url"] == "ws://127.0.0.1:8765/v1"
    assert handler._realtime_connect_query == {"session_token": "abc123"}


@pytest.mark.asyncio
async def test_build_realtime_client_uses_deployed_mode_even_when_direct_hf_ws_url_is_saved(
    monkeypatch: Any,
) -> None:
    """Explicit deployed mode should let .env recover from a stale local websocket URL."""
    captured_client_kwargs: dict[str, Any] = {}
    requested_session_urls: list[str] = []
    requested_session_headers: list[dict[str, str] | None] = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_client_kwargs.update(kwargs)

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {
                "session_id": "session-123",
                "connect_url": "wss://hf.example.test/v1/realtime?session_token=allocated",
            }

    class FakeAsyncClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def post(self, url: str, headers: dict[str, str] | None = None) -> FakeResponse:
            requested_session_urls.append(url)
            requested_session_headers.append(headers)
            return FakeResponse()

    monkeypatch.setattr(hf_mod, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setattr(config, "HF_TOKEN", "hf-secret")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert requested_session_urls == ["https://lb.example.test/session"]
    assert requested_session_headers == [{"Authorization": "Bearer hf-secret"}]
    assert captured_client_kwargs["api_key"] == "hf-secret"
    assert captured_client_kwargs["base_url"] == "https://hf.example.test/v1"
    assert captured_client_kwargs["websocket_base_url"] == "wss://hf.example.test/v1"
    assert handler._realtime_connect_query == {"session_token": "allocated"}


@pytest.mark.asyncio
async def test_build_realtime_client_does_not_send_openai_key_to_hf_allocator(monkeypatch: Any) -> None:
    """Hugging Face allocator auth should use HF_TOKEN only."""
    captured_client_kwargs: dict[str, Any] = {}
    requested_session_headers: list[dict[str, str] | None] = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_client_kwargs.update(kwargs)

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {
                "session_id": "session-123",
                "connect_url": "wss://hf.example.test/v1/realtime?session_token=allocated",
            }

    class FakeAsyncClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def post(self, _url: str, headers: dict[str, str] | None = None) -> FakeResponse:
            requested_session_headers.append(headers)
            return FakeResponse()

    monkeypatch.setattr(hf_mod, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setattr(config, "HF_TOKEN", None)

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert requested_session_headers == [None]
    assert captured_client_kwargs["api_key"] == "DUMMY"


@pytest.mark.asyncio
async def test_apply_personality_uses_selected_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Live personality updates should honor the selected Qwen CustomVoice speaker."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda: "new instructions")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Serena")
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
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

    result = await handler.apply_personality("example")

    assert "restarted realtime session" in result.lower()
    session = captured_update["session"]
    assert session["instructions"] == "new instructions"
    assert session["audio"]["output"]["voice"] == "Serena"


def test_huggingface_response_cost_defaults_to_zero() -> None:
    """Hugging Face should not inherit OpenAI pricing from the shared base handler."""
    usage = _make_usage(audio_in=1000, text_in=2000, image_in=500, audio_out=800, text_out=300)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler._compute_response_cost(usage) == 0.0
