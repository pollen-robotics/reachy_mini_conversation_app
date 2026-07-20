"""Tests for the headless console stream."""

import time
import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from collections.abc import Callable

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.config import HF_AVAILABLE_VOICES, config
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.startup_settings import (
    StartupSettings,
    load_startup_settings_into_runtime,
)
from reachy_mini_conversation_app.personality_routes import mount_personality_routes


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


def test_mic_endpoints_report_and_toggle_mute_state() -> None:
    """The mic starts live; the settings API exposes and flips the pause state."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    client = TestClient(app)

    assert client.get("/api/v1/mic").json() == {"muted": False}

    assert client.post("/api/v1/mic", json={"muted": True}).json() == {"muted": True}
    assert stream._mic_muted is True

    assert client.post("/api/v1/mic", json={"muted": False}).json() == {"muted": False}
    assert stream._mic_muted is False

    # headless streams keep the mic live
    assert LocalStream(MagicMock(), robot)._mic_muted is False


def test_settings_api_uses_versioned_routes_only() -> None:
    """Settings clients should use the versioned API paths."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    client = TestClient(app)

    assert client.get("/api/v1/mic").json() == {"muted": False}

    response = client.post("/api/v1/mic", json={"muted": True})

    assert response.status_code == 200
    assert response.json() == {"muted": True}
    assert stream._mic_muted is True
    assert client.get("/api/v1/status").json()["backend"]
    assert client.get("/api/v1/ready").status_code == 404
    assert client.get("/status").status_code == 404
    assert client.get("/ready").status_code == 404
    assert client.post("/backend_config", json={"backend": "openai"}).status_code == 404
    assert client.get("/mic").status_code == 404
    assert client.post("/mic", json={"muted": False}).status_code == 404
    assert client.get("/conversation_events").status_code == 404


def test_settings_ui_detaches_framework_catch_all_before_api_routes() -> None:
    """Framework fallback routes should not shadow the settings API."""
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
    assert client.get("/api/v1/mic").status_code == 200
    assert client.get("/api/v1/personalities").status_code == 200


@pytest.mark.asyncio
async def test_conversation_events_survive_handler_rebuild() -> None:
    """Activity from a rebuilt handler must reach subscribers of the original event bus."""

    class FakeHandler:
        def __init__(self) -> None:
            self.observer = None

        def set_activity_observer(self, observer: Any) -> None:
            self.observer = observer

    rebuilt = FakeHandler()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(FakeHandler(), robot, settings_app=FastAPI(), handler_factory=lambda voice: rebuilt)

    queue, unsubscribe = stream._event_bus.subscribe()
    stream._build_handler_for_current_backend()
    rebuilt.observer("assistant_audio_delta")

    assert await asyncio.wait_for(queue.get(), timeout=1.0) == "assistant_audio_delta"
    unsubscribe()


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

    response = TestClient(app).post(
        "/api/v1/backend_config",
        json={"hf_mode": "local", "hf_host": "localhost", "hf_port": 8765},
    )

    assert response.status_code == 200
    data = response.json()
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

    client = TestClient(app)
    response = client.post(
        "/api/v1/backend_config",
        json={
            "hf_mode": "local",
            "hf_host": "localhost",
            "hf_port": 8765,
        },
    )

    assert response.status_code == 200
    data = response.json()
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

    client = TestClient(app)
    response = client.post(
        "/api/v1/backend_config",
        json={
            "hf_mode": "deployed",
        },
    )

    assert response.status_code == 200
    data = response.json()
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

    client = TestClient(app)
    response = client.post(
        "/api/v1/backend_config",
        json={},
    )

    assert response.status_code == 200
    data = response.json()
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

    client = TestClient(app)
    response = client.post(
        "/api/v1/backend_config",
        json={
            "backend": "huggingface",
            "hf_mode": "local",
            "hf_host": "localhost",
            "hf_port": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_hf_port"


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

    client = TestClient(app)
    response = client.get("/api/v1/status")

    assert response.status_code == 200
    data = response.json()
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

    client = TestClient(app)
    response = client.get("/api/v1/status")

    assert response.status_code == 200
    data = response.json()
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
    client = TestClient(app)
    response = client.get("/api/v1/status")

    assert response.status_code == 200
    data = response.json()
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


def test_personality_routes_return_hf_voices() -> None:
    """Headless personality UI should expose the Hugging Face voices."""
    app = FastAPI()
    handler = MagicMock()
    mount_personality_routes(app, handler, lambda: None)

    client = TestClient(app)
    response = client.get("/voices")

    assert response.status_code == 200
    assert response.json() == HF_AVAILABLE_VOICES


def test_personality_routes_mount_versioned_paths() -> None:
    """Personality and voice endpoints should use the configured API prefix."""
    app = FastAPI()
    handler = MagicMock()

    mount_personality_routes(
        app,
        handler,
        lambda: None,
        api_prefix="/api/v1",
    )

    client = TestClient(app)
    response = client.get("/api/v1/voices")
    delete_response = client.delete("/api/v1/personalities", params={"name": "mad_scientist_assistant"})

    assert response.status_code == 200
    assert delete_response.status_code == 404
    assert delete_response.json()["error"] == "not_deletable"
    assert client.get("/personalities").status_code == 404
    assert client.get("/voices").status_code == 404
    assert client.delete("/personalities", params={"name": "mad_scientist_assistant"}).status_code == 404
    assert client.post("/voices/apply?voice=cedar").status_code == 404


def test_personality_routes_load_builtin_default_tools() -> None:
    """Headless personality UI should expose built-in default tools on initial load."""
    app = FastAPI()
    handler = MagicMock()
    mount_personality_routes(app, handler, lambda: None)

    client = TestClient(app)
    response = client.get("/personalities/load", params={"name": "(built-in default)"})

    assert response.status_code == 200
    data = response.json()
    assert data["tools_text"]
    assert "dance" in data["enabled_tools"]
    assert "camera" in data["enabled_tools"]


def test_personality_routes_apply_voice_accepts_query_param() -> None:
    """Headless personality UI should apply a voice change from a POST query param."""
    app = FastAPI()
    handler = MagicMock()
    handler.change_voice = AsyncMock(return_value="Voice changed to cedar.")

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    started.wait(timeout=1.0)

    try:
        mount_personality_routes(app, handler, lambda: loop, api_prefix="/api/v1")

        client = TestClient(app)
        response = client.post("/api/v1/voices/apply?voice=cedar")

        assert response.status_code == 200
        assert response.json() == {"ok": True, "status": "Voice changed to cedar."}
        handler.change_voice.assert_awaited_once_with("cedar")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_personality_routes_persist_startup_with_voice_override() -> None:
    """Saving a startup personality should persist the active manual voice override."""
    app = FastAPI()
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="Applied personality and restarted realtime session.")
    handler.get_current_voice = MagicMock(return_value="shimmer")
    persist_personality = MagicMock()

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    started.wait(timeout=1.0)

    try:
        mount_personality_routes(app, handler, lambda: loop, persist_personality=persist_personality)

        client = TestClient(app)
        response = client.post("/personalities/apply", json={"name": "sorry_bro", "persist": True})

        assert response.status_code == 200
        assert response.json()["ok"] is True
        handler.apply_personality.assert_awaited_once_with("sorry_bro")
        persist_personality.assert_called_once_with("sorry_bro", "shimmer")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_personality_routes_apply_same_profile_does_not_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-applying the active personality should be a no-op for the realtime handler."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "sorry_bro")
    app = FastAPI()
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="should not be called")
    handler.get_current_voice = MagicMock(return_value="shimmer")
    mount_personality_routes(app, handler, lambda: None)

    response = TestClient(app).post("/personalities/apply", json={"name": "sorry_bro"})

    assert response.status_code == 200
    assert response.json()["status"] == "Personality unchanged."
    handler.apply_personality.assert_not_awaited()
    handler.get_current_voice.assert_not_called()


def test_personality_routes_startup_choice_survives_runtime_profile_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime profile switching should not redefine the saved startup personality."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "captain_circuit")
    app = FastAPI()
    handler = MagicMock()
    mount_personality_routes(app, handler, lambda: None)
    client = TestClient(app)

    initial_response = client.get("/personalities")
    assert initial_response.status_code == 200
    assert initial_response.json()["current"] == "captain_circuit"
    assert initial_response.json()["startup"] == "captain_circuit"

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "chess_coach")

    switched_response = client.get("/personalities")
    assert switched_response.status_code == 200
    assert switched_response.json()["current"] == "chess_coach"
    assert switched_response.json()["startup"] == "captain_circuit"


def test_headless_personality_routes_can_use_stream_callbacks() -> None:
    """Headless personality routes can delegate apply/restart ownership to LocalStream."""
    app = FastAPI()
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="handler should not be called")
    apply_personality = AsyncMock(return_value="Applied personality and restarting backend.")
    get_current_voice = MagicMock(return_value="cedar")

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    started.wait(timeout=1.0)

    try:
        mount_personality_routes(
            app,
            handler,
            lambda: loop,
            apply_personality=apply_personality,
            get_current_voice=get_current_voice,
        )

        response = TestClient(app).post("/personalities/apply", json={"name": "sorry_bro"})

        assert response.status_code == 200
        assert response.json()["status"] == "Applied personality and restarting backend."
        apply_personality.assert_awaited_once_with("sorry_bro")
        handler.apply_personality.assert_not_awaited()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


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


def _bare_stream() -> LocalStream:
    """Return a LocalStream with a no-audio robot, enough for helper-method tests."""
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    return LocalStream(MagicMock(), robot)


def test_read_env_lines_prefers_existing_file(tmp_path: Path) -> None:
    """An existing .env is read verbatim, ignoring the template."""
    env_path = tmp_path / ".env"
    env_path.write_text("A=1\nB=2\n", encoding="utf-8")

    assert _bare_stream()._read_env_lines(env_path) == ["A=1", "B=2"]


def test_read_env_lines_falls_back_to_example_template(tmp_path: Path) -> None:
    """When no .env exists, the sibling .env.example is used as the template."""
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\n", encoding="utf-8")

    assert _bare_stream()._read_env_lines(tmp_path / ".env") == ["OPENAI_API_KEY="]


def test_seconds_since_activity_reads_handler() -> None:
    """seconds_since_activity is measured from the handler's last activity time."""
    stream = _bare_stream()
    stream.handler.last_activity_time = time.monotonic() - 5.0

    assert stream.seconds_since_activity() >= 5.0


def test_get_current_voice_prefers_override() -> None:
    """A manual voice override wins over the profile voice."""
    stream = _bare_stream()
    stream._voice_override = "Serena"

    assert stream.get_current_voice() == "Serena"


@pytest.mark.asyncio
async def test_change_voice_reports_handler_failure() -> None:
    """A failing handler voice change is surfaced as an error string, not raised."""
    handler = MagicMock()
    handler.change_voice = AsyncMock(side_effect=RuntimeError("backend down"))
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(handler, robot)

    result = await stream.change_voice("Serena")

    assert "Failed to change voice" in result


def _audio_robot(**media_attrs: Any) -> SimpleNamespace:
    """Return a robot whose media exposes only the attributes a test drives."""
    return SimpleNamespace(media=SimpleNamespace(audio=None, backend=None, **media_attrs))


def _stop_after(stream: LocalStream, value: Any) -> Callable[[], Any]:
    """Return a side effect that stops the stream after one iteration, yielding `value`."""

    def _side_effect() -> Any:
        stream._stop_event.set()
        return value

    return _side_effect


@pytest.mark.asyncio
async def test_record_loop_forwards_unmuted_frames() -> None:
    """A recorded frame is forwarded to the handler with the input sample rate."""
    frame = np.zeros(4, dtype=np.int16)
    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=16000), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    robot.media.get_audio_sample.side_effect = _stop_after(stream, frame)

    await stream.record_loop()

    handler.receive.assert_awaited_once_with((16000, frame))


@pytest.mark.asyncio
async def test_record_loop_skips_frames_while_muted() -> None:
    """No frames are forwarded while the mic is muted."""
    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=16000), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    stream._mic_muted = True
    robot.media.get_audio_sample.side_effect = _stop_after(stream, np.zeros(4, dtype=np.int16))

    await stream.record_loop()

    handler.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_loop_skips_missing_frames() -> None:
    """A None frame from the recorder is not forwarded."""
    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=16000), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    robot.media.get_audio_sample.side_effect = _stop_after(stream, None)

    await stream.record_loop()

    handler.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_play_loop_logs_text_outputs() -> None:
    """Text outputs are logged, not pushed to the speaker."""
    robot = _audio_robot(push_audio_sample=MagicMock())
    handler = MagicMock()
    stream = LocalStream(handler, robot)
    output = AdditionalOutputs({"role": "assistant", "content": "hi"})
    handler.emit = AsyncMock(side_effect=_stop_after(stream, output))

    await stream.play_loop()

    robot.media.push_audio_sample.assert_not_called()


@pytest.mark.asyncio
async def test_play_loop_pushes_mono_audio_as_float32() -> None:
    """A mono int16 frame is pushed to the speaker as float32."""
    robot = _audio_robot(push_audio_sample=MagicMock())
    handler = MagicMock()
    stream = LocalStream(handler, robot)
    handler.emit = AsyncMock(side_effect=_stop_after(stream, (24000, np.zeros(4, dtype=np.int16))))

    await stream.play_loop()

    robot.media.push_audio_sample.assert_called_once()
    pushed = robot.media.push_audio_sample.call_args.args[0]
    assert pushed.ndim == 1
    assert pushed.dtype == np.float32


@pytest.mark.asyncio
async def test_play_loop_downmixes_stereo_before_pushing() -> None:
    """A stereo frame is reduced to a single mono channel before playback."""
    robot = _audio_robot(push_audio_sample=MagicMock())
    handler = MagicMock()
    stream = LocalStream(handler, robot)
    stereo = np.zeros((4, 2), dtype=np.int16)
    handler.emit = AsyncMock(side_effect=_stop_after(stream, (24000, stereo)))

    await stream.play_loop()

    pushed = robot.media.push_audio_sample.call_args.args[0]
    assert pushed.ndim == 1


@pytest.mark.asyncio
async def test_play_loop_skips_empty_audio() -> None:
    """An empty audio frame is skipped, not pushed."""
    robot = _audio_robot(push_audio_sample=MagicMock())
    handler = MagicMock()
    stream = LocalStream(handler, robot)
    handler.emit = AsyncMock(side_effect=_stop_after(stream, (24000, np.array([], dtype=np.int16))))

    await stream.play_loop()

    robot.media.push_audio_sample.assert_not_called()


def test_close_without_running_loop_stops_media() -> None:
    """Closing without a running loop stops the media pipelines and sets the stop event."""
    robot = _audio_robot(stop_recording=MagicMock(), stop_playing=MagicMock())
    stream = LocalStream(MagicMock(), robot)
    stream._asyncio_loop = None

    stream.close()

    robot.media.stop_recording.assert_called_once()
    robot.media.stop_playing.assert_called_once()
    assert stream._stop_event.is_set()


def test_drain_output_queue_empties_in_place() -> None:
    """The output queue is drained without being replaced."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    queue.put_nowait("a")
    queue.put_nowait("b")
    handler = MagicMock()
    handler.output_queue = queue
    stream = LocalStream(handler, _audio_robot())

    stream._drain_output_queue()

    assert stream.handler.output_queue is queue
    assert queue.empty()


def test_drain_output_queue_tolerates_missing_queue() -> None:
    """Draining is a no-op when the handler has no output queue."""
    handler = MagicMock()
    handler.output_queue = None
    stream = LocalStream(handler, _audio_robot())

    stream._drain_output_queue()  # must not raise
