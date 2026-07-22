"""Tests for the headless console stream."""

import asyncio
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.config import HF_AVAILABLE_VOICES, config
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.startup_settings import (
    StartupSettings,
    load_startup_settings_into_runtime,
)
from reachy_mini_conversation_app.personality_routes import (
    RouteError,
    build_personality_ops,
)


def _rpc_call(app: FastAPI, method: str, params: Any = None) -> dict[str, Any]:
    """Send one JSON-RPC request over /rpc and return the response envelope."""
    with TestClient(app).websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        return ws.receive_json()


async def _wait_until(predicate: Any, timeout: float = 1.0) -> None:
    """Wait until a test predicate becomes true."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


def test_clear_audio_queue_prefers_clear_player() -> None:
    """clear_player() is the canonical flush and is used whenever available."""
    handler = MagicMock()
    handler.output_queue = asyncio.Queue()
    handler.output_queue.put_nowait((24000, np.zeros(4, dtype=np.int16)))
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_player.assert_called_once()
    audio.clear_output_buffer.assert_not_called()
    assert handler.output_queue.empty()


def test_clear_audio_queue_falls_back_to_output_buffer() -> None:
    """Older SDKs without clear_player() still flush via clear_output_buffer()."""
    handler = MagicMock()
    handler.output_queue = asyncio.Queue()
    audio = SimpleNamespace(clear_output_buffer=MagicMock())  # no clear_player
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    assert handler.output_queue.empty()


def test_clear_audio_queue_drains_queue_in_place() -> None:
    """The output queue is drained in place, not replaced with a new object."""
    handler = MagicMock()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    queue.put_nowait((24000, np.zeros(4, dtype=np.int16)))
    queue.put_nowait((24000, np.zeros(4, dtype=np.int16)))
    handler.output_queue = queue
    audio = SimpleNamespace(clear_player=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    assert handler.output_queue is queue  # same object, not replaced
    assert queue.empty()


def test_mic_reports_and_toggles_mute_state_over_rpc() -> None:
    """The mic starts live; conversation.mic exposes and flips the pause state."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()

    assert _rpc_call(app, "conversation.mic")["result"] == {"muted": False}
    assert _rpc_call(app, "conversation.mic", {"muted": True})["result"] == {"muted": True}
    assert stream._mic_muted is True
    assert _rpc_call(app, "conversation.mic", {"muted": False})["result"] == {"muted": False}
    assert stream._mic_muted is False

    # headless streams keep the mic live
    assert LocalStream(MagicMock(), robot)._mic_muted is False


def test_rest_api_is_removed_in_favor_of_rpc() -> None:
    """The /api/v1 REST + SSE surface is gone; control is JSON-RPC over /rpc."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    client = TestClient(app)

    for path in ("/api/v1/status", "/api/v1/mic", "/api/v1/personalities", "/api/v1/voices"):
        assert client.get(path).status_code == 404
    assert client.get("/api/v1/conversation_events").status_code == 404

    # ...but /rpc drives it fine.
    assert _rpc_call(app, "conversation.status")["result"]["backend"]


def test_settings_ui_detaches_framework_catch_all_before_own_routes() -> None:
    """Framework fallback routes should not shadow the UI or the /rpc endpoint."""
    app = FastAPI()

    @app.get("/{path:path}")
    def _framework_fallback(path: str) -> None:
        raise HTTPException(status_code=404)

    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/static/js/api.js").status_code == 200
    assert _rpc_call(app, "conversation.status")["result"]["backend"]


@pytest.mark.asyncio
async def test_activity_from_rebuilt_handler_reaches_rpc_clients() -> None:
    """Activity from a rebuilt handler must still reach /rpc subscribers."""

    class FakeHandler:
        def __init__(self) -> None:
            self.observer: Any = None

        def set_activity_observer(self, observer: Any) -> None:
            self.observer = observer

    rebuilt = FakeHandler()
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(FakeHandler(), robot, settings_app=app, handler_factory=lambda voice: rebuilt)
    stream._init_settings_ui_if_needed()
    stream._build_handler_for_current_backend()  # rebuild re-wires the observer

    with TestClient(app).websocket_connect("/rpc") as ws:
        rebuilt.observer("assistant_audio_delta")
        # First frame is conversation.activity (raw reason).
        msg = ws.receive_json()
    assert msg["method"] == "conversation.activity"
    assert msg["params"] == {"reason": "assistant_audio_delta"}


def test_backend_config_requests_in_process_restart_with_handler_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild-capable LocalStream should reconnect in process after a connection change."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.delenv("HF_REALTIME_WS_URL", raising=False)

    app = FastAPI()
    handler = MagicMock()
    handler.shutdown = AsyncMock()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(
        handler,
        robot,
        settings_app=app,
        instance_path=str(tmp_path),
        handler_factory=lambda _voice: handler,
    )
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "backend.config", {"hf_mode": "local", "hf_host": "localhost", "hf_port": 8765})["result"]

    assert data["ok"] is True
    assert data["message"] == "Connection saved. Reconnecting backend."
    assert data["backend"] == "huggingface"
    assert data["requires_restart"] is False
    assert data["can_proceed"] is True
    assert data["backend_connection_state"] == "connecting"
    assert stream._restart_requested.is_set()


def test_backend_config_persists_local_hf_selection_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should persist a direct Hugging Face websocket target."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.delenv("HF_REALTIME_SESSION_URL", raising=False)
    monkeypatch.delenv("HF_REALTIME_WS_URL", raising=False)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "backend.config", {"hf_mode": "local", "hf_host": "localhost", "hf_port": 8765})["result"]

    assert data["ok"] is True
    assert data["backend"] == "huggingface"
    assert data["has_hf_ws_url"] is True
    assert data["has_hf_connection"] is True
    assert data["hf_connection_mode"] == "local"
    assert data["hf_direct_host"] == "localhost"
    assert data["hf_direct_port"] == 8765

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "HF_REALTIME_CONNECTION_MODE=local" in env_text
    assert "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime" in env_text


def test_backend_config_persists_deployed_mode_without_clearing_local_hf_ws_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving deployed mode should make env selection explicit and remove stale allocator URLs."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "HF_REALTIME_SESSION_URL=https://lb.example.test/session\n"
        "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://localhost:8765/v1/realtime")
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.setenv("HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setenv("HF_REALTIME_WS_URL", "ws://localhost:8765/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "backend.config", {"hf_mode": "deployed"})["result"]

    assert data["ok"] is True
    assert data["has_hf_session_url"] is True
    assert data["has_hf_ws_url"] is True
    assert data["hf_connection_mode"] == "deployed"

    env_text = env_path.read_text(encoding="utf-8")
    assert "HF_REALTIME_CONNECTION_MODE=deployed" in env_text
    assert "HF_REALTIME_SESSION_URL=" not in env_text
    assert "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime" in env_text


def test_backend_config_switches_to_saved_local_hf_connection_without_payload_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching back to a saved local Hugging Face backend should reuse the persisted target."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "HF_REALTIME_CONNECTION_MODE=local\nHF_REALTIME_WS_URL=ws://192.168.1.42:8766/v1/realtime\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://192.168.1.42:8766/v1/realtime")
    monkeypatch.setenv("HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setenv("HF_REALTIME_WS_URL", "ws://192.168.1.42:8766/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "backend.config", {})["result"]

    assert data["ok"] is True
    assert data["backend"] == "huggingface"
    assert data["hf_connection_mode"] == "local"
    assert data["hf_direct_host"] == "192.168.1.42"
    assert data["hf_direct_port"] == 8766

    env_text = env_path.read_text(encoding="utf-8")
    assert "HF_REALTIME_CONNECTION_MODE=local" in env_text
    assert "HF_REALTIME_WS_URL=ws://192.168.1.42:8766/v1/realtime" in env_text


def test_backend_config_rejects_invalid_hf_port_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should reject invalid local Hugging Face ports from direct callers."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    resp = _rpc_call(
        app,
        "backend.config",
        {"backend": "huggingface", "hf_mode": "local", "hf_host": "localhost", "hf_port": 0},
    )

    assert resp["error"]["data"]["reason"] == "invalid_hf_port"


def test_status_reports_direct_hf_ws_url_as_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should treat a direct Hugging Face websocket as a valid configuration."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "conversation.status")["result"]

    assert data["backend"] == "huggingface"
    assert data["has_hf_session_url"] is False
    assert data["has_hf_ws_url"] is True
    assert data["has_hf_connection"] is True
    assert data["hf_connection_mode"] == "local"
    assert data["can_proceed_with_hf"] is True


def test_status_reports_backend_connection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should expose backend connection failures without hiding controls."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    app = FastAPI()
    handler = MagicMock()
    handler.connection = None
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(handler, robot, settings_app=app, instance_path=str(tmp_path))
    stream._set_backend_connection_state("disconnected", RuntimeError("connect failed"))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "conversation.status")["result"]
    assert data["backend"] == "huggingface"
    assert data["backend_connected"] is False
    assert data["backend_connection_state"] == "disconnected"
    assert data["backend_error"] == "RuntimeError: connect failed"
    assert data["can_proceed"] is True
    assert data["can_proceed_with_hf"] is True


def test_backend_startup_failure_is_recorded_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend startup failures should become status state instead of killing LocalStream."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    app = FastAPI()
    handler = MagicMock()
    handler.connection = None
    handler.shutdown = AsyncMock()
    media = SimpleNamespace(
        audio=None,
        backend=None,
        start_recording=MagicMock(),
        start_playing=MagicMock(),
    )
    robot = SimpleNamespace(media=media)
    stream = LocalStream(handler, robot, settings_app=app, instance_path=str(tmp_path))
    stream._backend_retry_delay = 0
    stream.record_loop = AsyncMock(return_value=None)  # type: ignore[method-assign]
    stream.play_loop = AsyncMock(return_value=None)  # type: ignore[method-assign]
    monkeypatch.setattr("reachy_mini_conversation_app.console.apply_audio_startup_config", MagicMock())

    async def fail_and_stop() -> None:
        stream._stop_event.set()
        raise RuntimeError("local server unavailable")

    handler.start_up = AsyncMock(side_effect=fail_and_stop)

    try:
        stream.launch()
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())

    handler.start_up.assert_awaited_once()
    data = _rpc_call(app, "conversation.status")["result"]
    assert data["backend_connected"] is False
    assert data["backend_connection_state"] == "disconnected"
    assert data["backend_error"] == "RuntimeError: local server unavailable"


@pytest.mark.asyncio
async def test_startup_loop_rebuilds_handler_on_restart_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """LocalStream should shut down and rebuild the handler when a restart is requested."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    class FakeHandler:
        def __init__(self) -> None:
            self.connection = None
            self.output_queue = asyncio.Queue()
            self.started = asyncio.Event()
            self.stopped = asyncio.Event()
            self.shutdown_calls = 0

        async def start_up(self) -> None:
            self.connection = object()
            self.started.set()
            await self.stopped.wait()
            self.connection = None

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            self.stopped.set()

        async def receive(self, _frame: Any) -> None:
            return None

        async def emit(self) -> None:
            return None

    handlers: list[FakeHandler] = []

    def handler_factory(_voice: str | None) -> FakeHandler:
        handler = FakeHandler()
        handlers.append(handler)
        return handler

    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    initial_handler = handler_factory(None)
    stream = LocalStream(initial_handler, robot, handler_factory=handler_factory)
    stream._backend_retry_delay = 0.01

    startup_task = asyncio.create_task(stream._run_handler_startup_loop())
    try:
        await _wait_until(lambda: initial_handler.started.is_set())

        await stream.request_backend_restart("backend_config_changed")

        await _wait_until(lambda: len(handlers) == 2 and handlers[1].started.is_set())

        assert initial_handler.shutdown_calls >= 1
        assert stream.handler is handlers[1]
        assert stream._backend_connected() is True
    finally:
        stream._stop_event.set()
        await stream._shutdown_active_handler()
        startup_task.cancel()
        try:
            await startup_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_personality_ops_return_hf_voices() -> None:
    """With no running loop, voices() falls back to the Hugging Face catalog."""
    ops = build_personality_ops(MagicMock(), lambda: None)
    assert await ops.voices() == HF_AVAILABLE_VOICES


def test_personality_ops_delete_builtin_is_not_deletable() -> None:
    """Deleting a built-in personality raises not_deletable (was REST 404)."""
    ops = build_personality_ops(MagicMock(), lambda: None)
    with pytest.raises(RouteError) as ei:
        ops.delete("mad_scientist_assistant")
    assert ei.value.reason == "not_deletable"


def test_personality_ops_load_builtin_default_tools() -> None:
    """load('(built-in default)') exposes the built-in default tools."""
    ops = build_personality_ops(MagicMock(), lambda: None)
    data = ops.load("(built-in default)")
    assert data["tools_text"]
    assert "dance" in data["enabled_tools"]
    assert "camera" in data["enabled_tools"]


@pytest.mark.asyncio
async def test_personality_ops_apply_voice() -> None:
    """apply_voice delegates to the handler and reports the status."""
    handler = MagicMock()
    handler.change_voice = AsyncMock(return_value="Voice changed to cedar.")
    ops = build_personality_ops(handler, lambda: asyncio.get_running_loop())

    result = await ops.apply_voice("cedar")

    assert result == {"ok": True, "status": "Voice changed to cedar."}
    handler.change_voice.assert_awaited_once_with("cedar")


@pytest.mark.asyncio
async def test_personality_ops_persist_startup_with_voice_override() -> None:
    """Applying with persist=True saves the active manual voice override."""
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="Applied personality and restarted realtime session.")
    handler.get_current_voice = MagicMock(return_value="shimmer")
    persist_personality = MagicMock()
    ops = build_personality_ops(handler, lambda: asyncio.get_running_loop(), persist_personality=persist_personality)

    result = await ops.apply("sorry_bro", persist=True)

    assert result["ok"] is True
    handler.apply_personality.assert_awaited_once_with("sorry_bro")
    persist_personality.assert_called_once_with("sorry_bro", "shimmer")


@pytest.mark.asyncio
async def test_personality_ops_apply_same_profile_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-applying the active personality is a no-op for the realtime handler."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "sorry_bro")
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="should not be called")
    handler.get_current_voice = MagicMock(return_value="shimmer")
    ops = build_personality_ops(handler, lambda: None)

    result = await ops.apply("sorry_bro")

    assert result["status"] == "Personality unchanged."
    handler.apply_personality.assert_not_awaited()
    handler.get_current_voice.assert_not_called()


def test_personality_ops_startup_choice_survives_runtime_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime profile switching should not redefine the saved startup personality."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "captain_circuit")
    ops = build_personality_ops(MagicMock(), lambda: None)

    first = ops.get_choices()
    assert first["current"] == "captain_circuit"
    assert first["startup"] == "captain_circuit"

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "chess_coach")

    second = ops.get_choices()
    assert second["current"] == "chess_coach"
    assert second["startup"] == "captain_circuit"


@pytest.mark.asyncio
async def test_personality_ops_use_apply_callback() -> None:
    """Apply delegates to the injected apply_personality callback, not the handler."""
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="handler should not be called")
    apply_personality = AsyncMock(return_value="Applied personality and restarting backend.")
    get_current_voice = MagicMock(return_value="cedar")
    ops = build_personality_ops(
        handler,
        lambda: asyncio.get_running_loop(),
        apply_personality=apply_personality,
        get_current_voice=get_current_voice,
    )

    result = await ops.apply("sorry_bro")

    assert result["status"] == "Applied personality and restarting backend."
    apply_personality.assert_awaited_once_with("sorry_bro")
    handler.apply_personality.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_personality_propagates_restart_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation during backend restart should not be converted into a status string."""
    monkeypatch.setattr("reachy_mini_conversation_app.config.set_custom_profile", lambda _profile: None)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.prompts.get_session_instructions", lambda _instance_path=None: "instructions"
    )
    monkeypatch.setattr("reachy_mini_conversation_app.prompts.get_session_voice", lambda default: default)

    stream = LocalStream(MagicMock(), MagicMock())

    async def cancel_restart(_reason: str) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(stream, "request_backend_restart", cancel_restart)

    with pytest.raises(asyncio.CancelledError):
        await stream.apply_personality("sorry_bro")


@pytest.mark.asyncio
async def test_local_stream_change_voice_delegates_without_backend_restart() -> None:
    """LocalStream voice changes should update the active handler without rebuilding it."""
    handler = MagicMock()
    handler.change_voice = AsyncMock(return_value="Voice changed to Serena.")
    handler.get_current_voice = MagicMock(return_value="Serena")
    stream = LocalStream(handler, MagicMock())

    status = await stream.change_voice("Serena")

    assert status == "Voice changed to Serena."
    handler.change_voice.assert_awaited_once_with("Serena")
    assert stream._voice_override == "Serena"
    assert not stream._restart_requested.is_set()


def test_local_stream_persist_personality_stores_voice_override(tmp_path) -> None:
    """Persisting startup settings should write both profile and voice override."""
    stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

    stream._persist_personality("sorry_bro", "shimmer")

    settings_path = tmp_path / "startup_settings.json"
    assert settings_path.exists()
    assert settings_path.read_text(encoding="utf-8") == '{\n  "profile": "sorry_bro",\n  "voice": "shimmer"\n}\n'
    assert stream._read_persisted_personality() == "sorry_bro"


def test_local_stream_persist_personality_clears_legacy_startup_env_overrides(tmp_path, monkeypatch) -> None:
    """Saving startup settings should remove legacy `.env` profile and voice overrides."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "HF_TOKEN=test-token\n"
        "REACHY_MINI_CUSTOM_PROFILE=mad_scientist_assistant\n"
        "REACHY_MINI_VOICE_OVERRIDE=shimmer\n",
        encoding="utf-8",
    )
    stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

    stream._persist_personality(None, "Aiden")

    env_text = env_path.read_text(encoding="utf-8")
    assert "HF_TOKEN=test-token" in env_text
    assert "REACHY_MINI_CUSTOM_PROFILE=" not in env_text
    assert "REACHY_MINI_VOICE_OVERRIDE=" not in env_text

    applied_profiles: list[str | None] = []
    monkeypatch.delenv("REACHY_MINI_CUSTOM_PROFILE", raising=False)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(voice="Aiden")
    assert applied_profiles == [None]


def test_local_stream_launch_waits_for_missing_hf_target_without_starting_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup should wait for settings input when the Hugging Face target is missing."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)

    media = SimpleNamespace(
        start_recording=MagicMock(),
        start_playing=MagicMock(),
    )
    robot = SimpleNamespace(media=media)
    stream = LocalStream(MagicMock(), robot, settings_app=FastAPI(), instance_path=str(tmp_path))

    init_settings_ui = MagicMock()
    monkeypatch.setattr(stream, "_init_settings_ui_if_needed", init_settings_ui)
    monkeypatch.setattr("reachy_mini_conversation_app.console.time.sleep", MagicMock(side_effect=KeyboardInterrupt))

    stream.launch()

    init_settings_ui.assert_called_once()
    media.start_recording.assert_not_called()
    media.start_playing.assert_not_called()


def _rpc_robot() -> SimpleNamespace:
    """Return a robot mock whose audio pipeline supports clear_audio_queue()."""
    audio = SimpleNamespace(clear_player=MagicMock(), clear_output_buffer=MagicMock())
    return SimpleNamespace(media=SimpleNamespace(audio=audio))


def test_rpc_status_and_mic_over_websocket() -> None:
    """conversation.status/mic are reachable over the /rpc JSON-RPC WebSocket."""
    app = FastAPI()
    stream = LocalStream(MagicMock(), _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    client = TestClient(app)
    with client.websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": "conversation.status"})
        resp = ws.receive_json()
        assert resp["id"] == "1"
        assert "result" in resp

        ws.send_json({"jsonrpc": "2.0", "id": "2", "method": "conversation.mic", "params": {"muted": True}})
        resp = ws.receive_json()
        assert resp["result"] == {"muted": True}
    assert stream._mic_muted is True


def test_rpc_interrupt_broadcasts_turn_listening() -> None:
    """conversation.interrupt clears playback and pushes a turn:listening event."""
    handler = MagicMock()
    handler.output_queue = asyncio.Queue()
    handler._is_connected.return_value = True
    app = FastAPI()
    stream = LocalStream(handler, _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    with TestClient(app).websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": "conversation.interrupt"})
        msgs = [ws.receive_json(), ws.receive_json()]
    results = [m for m in msgs if "result" in m]
    notes = [m for m in msgs if m.get("method") == "conversation.turn"]
    assert results and results[0]["result"] == {"ok": True}
    assert notes and notes[0]["params"] == {"state": "listening", "reason": "interrupted"}


def test_rpc_say_requires_active_session() -> None:
    """conversation.say fails with not_running when no session is connected."""
    handler = MagicMock()
    handler._is_connected.return_value = False
    app = FastAPI()
    stream = LocalStream(handler, _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    with TestClient(app).websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": "conversation.say", "params": {"text": "hi"}})
        resp = ws.receive_json()
    assert resp["error"]["data"]["reason"] == "not_running"


def test_rpc_transcript_notification_broadcast() -> None:
    """The handler's transcript observer pushes conversation.transcript events."""
    app = FastAPI()
    stream = LocalStream(MagicMock(), _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    with TestClient(app).websocket_connect("/rpc") as ws:
        stream._dispatch_transcript("assistant", "hello there", True)
        msg = ws.receive_json()
    assert msg["method"] == "conversation.transcript"
    assert msg["params"] == {"role": "assistant", "text": "hello there", "final": True}


def test_rpc_personalities_and_voices_methods() -> None:
    """personalities.* / voices.* are reachable over /rpc (same ops as REST)."""
    app = FastAPI()
    stream = LocalStream(MagicMock(), _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    with TestClient(app).websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": "personalities.list"})
        r1 = ws.receive_json()
        ws.send_json({"jsonrpc": "2.0", "id": "2", "method": "voices.list"})
        r2 = ws.receive_json()
    assert "choices" in r1["result"] and "current" in r1["result"]
    assert isinstance(r2["result"], list)
