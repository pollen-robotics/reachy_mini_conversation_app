"""Tests for the headless console stream."""

import sys
import json
import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from numpy.typing import NDArray
from fastapi.testclient import TestClient

from reachy_mini.media.media_manager import MediaBackend
from robot_comic.config import GEMINI_AVAILABLE_VOICES, config
from robot_comic.console import LOCAL_PLAYER_BACKEND, LocalStream
from robot_comic.startup_settings import (
    StartupSettings,
    load_startup_settings_into_runtime,
)
from robot_comic.headless_personality_ui import mount_personality_routes


def test_clear_audio_queue_prefers_clear_player_when_available() -> None:
    """Local GStreamer audio should use the lower-level player flush when available."""
    handler = MagicMock()
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=LOCAL_PLAYER_BACKEND))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_player.assert_called_once()
    audio.clear_output_buffer.assert_not_called()
    assert isinstance(handler.output_queue, asyncio.Queue)
    assert handler.output_queue.empty()


def test_clear_audio_queue_uses_output_buffer_for_webrtc() -> None:
    """WebRTC audio should flush queued playback via the output buffer API."""
    handler = MagicMock()
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.WEBRTC))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    audio.clear_player.assert_not_called()
    assert isinstance(handler.output_queue, asyncio.Queue)
    assert handler.output_queue.empty()


def test_clear_audio_queue_falls_back_when_backend_is_unknown() -> None:
    """Unknown backends should still best-effort flush pending playback."""
    handler = MagicMock()
    audio = SimpleNamespace(clear_output_buffer=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=None))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    assert isinstance(handler.output_queue, asyncio.Queue)
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_play_loop_feeds_head_wobbler_with_local_playback_delay() -> None:
    """Local playback should drive speech wobble using the queued player delay."""
    head_wobbler = MagicMock()
    chunk = np.array([1, -2, 3, -4], dtype=np.int16)

    class Handler:
        def __init__(self) -> None:
            self.deps = SimpleNamespace(head_wobbler=head_wobbler)
            self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
            self._emitted = False

        async def emit(self) -> tuple[int, NDArray[np.int16]] | None:
            if not self._emitted:
                self._emitted = True
                return (24000, chunk.copy())
            return None

    audio = SimpleNamespace(
        _playback_next_pts_ns=1_500_000_000,
        _get_playback_running_time_ns=lambda: 500_000_000,
    )
    media = SimpleNamespace(
        audio=audio,
        backend=LOCAL_PLAYER_BACKEND,
        get_output_audio_samplerate=lambda: 24000,
        push_audio_sample=MagicMock(),
    )
    robot = SimpleNamespace(media=media)
    handler = Handler()
    stream = LocalStream(handler, robot)

    async def stop_soon() -> None:
        await asyncio.sleep(0.01)
        stream._stop_event.set()

    stopper = asyncio.create_task(stop_soon())
    try:
        await asyncio.wait_for(stream.play_loop(), timeout=1.0)
    finally:
        await stopper

    head_wobbler.feed_pcm.assert_called_once()
    args, kwargs = head_wobbler.feed_pcm.call_args
    assert np.array_equal(args[0], chunk.reshape(1, -1))
    assert args[1] == 24000
    assert kwargs["start_delay_s"] == pytest.approx(1.0)
    media.push_audio_sample.assert_called_once()


def test_backend_config_persists_gemini_selection_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should persist Gemini pipeline choice and token."""
    monkeypatch.setattr(config, "PIPELINE_MODE", "openai_realtime")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "openai_realtime_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "openai_realtime_output")
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)

    response = client.post(
        "/backend_config",
        json={"pipeline_mode": "gemini_live", "api_key": "gem-test-token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["pipeline_mode"] == "gemini_live"
    assert data["active_backend"] == "openai"
    assert data["has_gemini_key"] is True
    assert data["has_key"] is False
    assert data["can_proceed"] is False
    assert data["can_proceed_with_openai"] is False
    assert data["can_proceed_with_gemini"] is True
    assert data["requires_restart"] is True

    status = client.get("/status")
    assert status.status_code == 200
    status_data = status.json()
    assert status_data["pipeline_mode"] == "gemini_live"
    assert status_data["active_backend"] == "openai"
    assert status_data["has_gemini_key"] is True
    assert status_data["can_proceed"] is False
    assert status_data["can_proceed_with_openai"] is False
    assert status_data["can_proceed_with_gemini"] is True

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "REACHY_MINI_PIPELINE_MODE=gemini_live" in env_text
    assert "GEMINI_API_KEY=gem-test-token" in env_text
    # Phase 4f: the retired dials are no longer written as active assignments.
    legacy_dial_prefixes = ("BACKEND_PROVI" + "DER=", "LOCAL_STT_RESPONSE_BAC" + "KEND=")  # split to keep grep clean
    legacy_assignment_lines = [line for line in env_text.splitlines() if line.startswith(legacy_dial_prefixes)]
    assert legacy_assignment_lines == []


def test_backend_config_does_not_write_model_name_when_saving_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 4f: ``_persist_pipeline_choice`` no longer manages ``MODEL_NAME``."""
    monkeypatch.setattr(config, "PIPELINE_MODE", "openai_realtime")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "openai_realtime_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "openai_realtime_output")
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={"pipeline_mode": "openai_realtime", "api_key": "openai-test-key"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["can_proceed"] is True
    assert data["can_proceed_with_openai"] is True
    assert data["can_proceed_with_gemini"] is False

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "REACHY_MINI_PIPELINE_MODE=openai_realtime" in env_text
    assert "OPENAI_API_KEY=openai-test-key" in env_text
    # The pipeline-save path never writes a MODEL_NAME assignment in the new world.
    model_assignment_lines = [line for line in env_text.splitlines() if line.startswith("MODEL_NAME=")]
    assert model_assignment_lines == []


def test_backend_config_persists_local_stt_selection_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should persist composable-pipeline options and reuse OpenAI credentials."""
    monkeypatch.setattr(config, "PIPELINE_MODE", "openai_realtime")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "openai_realtime_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "openai_realtime_output")
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(config, "LOCAL_STT_CACHE_DIR", "./cache/moonshine_voice")
    monkeypatch.setattr(config, "LOCAL_STT_LANGUAGE", "en")
    monkeypatch.setattr(config, "LOCAL_STT_MODEL", "tiny_streaming")
    monkeypatch.setattr(config, "LOCAL_STT_UPDATE_INTERVAL", 0.35)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={
            "pipeline_mode": "composable",
            "audio_input_backend": "moonshine",
            "audio_output_backend": "openai_realtime_output",
            "api_key": "openai-test-key",
            "local_stt_cache_dir": "./cache/moonshine_voice",
            "local_stt_language": "en",
            "local_stt_model": "small_streaming",
            "local_stt_update_interval": 0.5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["pipeline_mode"] == "composable"
    assert data["audio_input_backend"] == "moonshine"
    assert data["audio_output_backend"] == "openai_realtime_output"
    assert data["active_backend"] == "openai"
    assert data["has_local_stt_key"] is True
    assert data["can_proceed_with_local_stt"] is True
    assert data["local_stt_cache_dir"] == "./cache/moonshine_voice"
    assert data["local_stt_language"] == "en"
    assert data["local_stt_model"] == "small_streaming"
    assert data["local_stt_update_interval"] == 0.5
    assert data["requires_restart"] is True

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "REACHY_MINI_PIPELINE_MODE=composable" in env_text
    assert "REACHY_MINI_AUDIO_INPUT_BACKEND=moonshine" in env_text
    assert "REACHY_MINI_AUDIO_OUTPUT_BACKEND=openai_realtime_output" in env_text
    assert "OPENAI_API_KEY=openai-test-key" in env_text
    assert "LOCAL_STT_PROVIDER=moonshine" in env_text
    assert "LOCAL_STT_CACHE_DIR=./cache/moonshine_voice" in env_text
    assert "LOCAL_STT_LANGUAGE=en" in env_text
    assert "LOCAL_STT_MODEL=small_streaming" in env_text
    assert "LOCAL_STT_UPDATE_INTERVAL=0.50" in env_text


def test_backend_config_rejects_invalid_local_stt_update_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should reject local STT update intervals that would hurt UX."""
    monkeypatch.setattr(config, "PIPELINE_MODE", "composable")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "moonshine")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "chatterbox")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "openai-test-key")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={
            "pipeline_mode": "composable",
            "audio_input_backend": "moonshine",
            "audio_output_backend": "chatterbox",
            "local_stt_model": "tiny_streaming",
            "local_stt_update_interval": 9.0,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_local_stt_update_interval"


def test_backend_config_persists_local_hf_selection_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should persist a direct Hugging Face websocket target."""
    monkeypatch.setattr(config, "PIPELINE_MODE", "openai_realtime")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "openai_realtime_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "openai_realtime_output")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.delenv("HF_REALTIME_SESSION_URL", raising=False)
    monkeypatch.delenv("HF_REALTIME_WS_URL", raising=False)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={
            "pipeline_mode": "hf_realtime",
            "hf_mode": "local",
            "hf_host": "localhost",
            "hf_port": 8765,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["pipeline_mode"] == "hf_realtime"
    assert data["active_backend"] == "openai"
    assert data["has_hf_ws_url"] is True
    assert data["has_hf_connection"] is True
    assert data["hf_connection_mode"] == "local"
    assert data["hf_direct_host"] == "localhost"
    assert data["hf_direct_port"] == 8765
    assert data["requires_restart"] is True

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "REACHY_MINI_PIPELINE_MODE=hf_realtime" in env_text
    assert "HF_REALTIME_CONNECTION_MODE=local" in env_text
    assert "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime" in env_text


def test_backend_config_persists_deployed_mode_without_clearing_local_hf_ws_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving deployed mode should make env selection explicit and remove stale allocator URLs."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "REACHY_MINI_PIPELINE_MODE=hf_realtime\n"
        "HF_REALTIME_SESSION_URL=https://lb.example.test/session\n"
        "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "PIPELINE_MODE", "hf_realtime")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "hf_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "hf_output")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://localhost:8765/v1/realtime")
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.setenv("HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setenv("HF_REALTIME_WS_URL", "ws://localhost:8765/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={
            "pipeline_mode": "hf_realtime",
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
        "REACHY_MINI_PIPELINE_MODE=openai_realtime\n"
        "HF_REALTIME_CONNECTION_MODE=local\n"
        "HF_REALTIME_WS_URL=ws://192.168.1.42:8766/v1/realtime\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "PIPELINE_MODE", "openai_realtime")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "openai_realtime_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "openai_realtime_output")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://192.168.1.42:8766/v1/realtime")
    monkeypatch.setenv("HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setenv("HF_REALTIME_WS_URL", "ws://192.168.1.42:8766/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={"pipeline_mode": "hf_realtime"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["pipeline_mode"] == "hf_realtime"
    assert data["hf_connection_mode"] == "local"
    assert data["hf_direct_host"] == "192.168.1.42"
    assert data["hf_direct_port"] == 8766

    env_text = env_path.read_text(encoding="utf-8")
    assert "REACHY_MINI_PIPELINE_MODE=hf_realtime" in env_text
    assert "HF_REALTIME_CONNECTION_MODE=local" in env_text
    assert "HF_REALTIME_WS_URL=ws://192.168.1.42:8766/v1/realtime" in env_text


def test_backend_config_rejects_invalid_hf_port_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should reject invalid local Hugging Face ports from direct callers."""
    monkeypatch.setattr(config, "PIPELINE_MODE", "hf_realtime")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "hf_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "hf_output")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={
            "pipeline_mode": "hf_realtime",
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
    monkeypatch.setattr(config, "PIPELINE_MODE", "hf_realtime")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "hf_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "hf_output")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.get("/status")

    assert response.status_code == 200
    data = response.json()
    assert data["pipeline_mode"] == "hf_realtime"
    assert data["active_backend"] == "huggingface"
    assert data["has_hf_session_url"] is False
    assert data["has_hf_ws_url"] is True
    assert data["has_hf_connection"] is True
    assert data["hf_connection_mode"] == "local"
    assert data["can_proceed_with_hf"] is True


def test_status_reports_local_crowd_history_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should expose where crowd-work session JSON is stored."""
    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / ".comedy_sessions"
    session_dir.mkdir()
    session_file = session_dir / "session_20260506_204006.json"
    session_file.write_text(json.dumps({"name": "Tony"}), encoding="utf-8")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.get("/status")

    assert response.status_code == 200
    data = response.json()
    assert data["crowd_history_dir"] == str(session_dir.resolve())
    assert data["crowd_history_count"] == 1
    assert data["crowd_history_latest"] == str(session_file.resolve())


def test_crowd_history_clear_removes_session_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin API should clear persisted crowd-work history without touching unrelated files."""
    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / ".comedy_sessions"
    session_dir.mkdir()
    first = session_dir / "session_20260506_204006.json"
    second = session_dir / "session_20260506_214915.json"
    unrelated = session_dir / "notes.txt"
    first.write_text(json.dumps({"name": "Tony"}), encoding="utf-8")
    second.write_text(json.dumps({"name": "Ari"}), encoding="utf-8")
    unrelated.write_text("keep me", encoding="utf-8")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream.init_admin_ui()

    client = TestClient(app)
    response = client.post("/crowd_history/clear")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["removed"] == 2
    assert data["crowd_history_count"] == 0
    assert not first.exists()
    assert not second.exists()
    assert unrelated.exists()


def test_headless_personality_routes_return_gemini_voices_when_backend_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headless personality UI should expose Gemini voices when Gemini is selected."""
    monkeypatch.setattr(config, "PIPELINE_MODE", "gemini_live")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "gemini_live_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "gemini_live_output")
    monkeypatch.setattr(config, "MODEL_NAME", "gemini-3.1-flash-live-preview")

    app = FastAPI()
    handler = MagicMock()
    mount_personality_routes(app, handler, lambda: None)

    client = TestClient(app)
    response = client.get("/voices")

    assert response.status_code == 200
    assert response.json() == GEMINI_AVAILABLE_VOICES


def test_headless_personality_routes_load_builtin_default_tools() -> None:
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


def test_headless_personality_routes_apply_voice_accepts_query_param() -> None:
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
        mount_personality_routes(app, handler, lambda: loop)

        client = TestClient(app)
        response = client.post("/voices/apply?voice=cedar")

        assert response.status_code == 200
        assert response.json() == {"ok": True, "status": "Voice changed to cedar."}
        handler.change_voice.assert_awaited_once_with("cedar")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_headless_personality_routes_persist_startup_with_voice_override() -> None:
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
        response = client.post("/personalities/apply?name=house_comedian&persist=1")

        assert response.status_code == 200
        assert response.json()["ok"] is True
        handler.apply_personality.assert_awaited_once_with("house_comedian")
        persist_personality.assert_called_once_with("house_comedian", "shimmer")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_local_stream_persist_personality_stores_voice_override(tmp_path) -> None:
    """Persisting startup settings should write both profile and voice override."""
    stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

    stream._persist_personality("house_comedian", "shimmer")

    settings_path = tmp_path / "startup_settings.json"
    assert settings_path.exists()
    assert settings_path.read_text(encoding="utf-8") == '{\n  "profile": "house_comedian",\n  "voice": "shimmer"\n}\n'
    assert stream._read_persisted_personality() == "house_comedian"


def test_local_stream_persist_personality_clears_legacy_startup_env_overrides(tmp_path, monkeypatch) -> None:
    """Saving startup settings should remove legacy `.env` profile and voice overrides."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=test-key\nREACHY_MINI_CUSTOM_PROFILE=house_comedian\nREACHY_MINI_VOICE_OVERRIDE=shimmer\n",
        encoding="utf-8",
    )
    stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

    stream._persist_personality(None, "Aiden")

    env_text = env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=test-key" in env_text
    assert "REACHY_MINI_CUSTOM_PROFILE=" not in env_text
    assert "REACHY_MINI_VOICE_OVERRIDE=" not in env_text

    applied_profiles: list[str | None] = []
    monkeypatch.delenv("REACHY_MINI_CUSTOM_PROFILE", raising=False)
    monkeypatch.setattr(
        "robot_comic.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(voice="Aiden")
    assert applied_profiles == [None]


def test_local_stream_launch_waits_for_manual_openai_key_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI startup should wait for settings input instead of claiming a bundled key."""
    monkeypatch.setattr(config, "PIPELINE_MODE", "openai_realtime")
    monkeypatch.setattr(config, "AUDIO_INPUT_BACKEND", "openai_realtime_input")
    monkeypatch.setattr(config, "AUDIO_OUTPUT_BACKEND", "openai_realtime_output")
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    fake_client_ctor = MagicMock(side_effect=AssertionError("launch() should not try to download an OpenAI key"))
    monkeypatch.setitem(sys.modules, "gradio_client", SimpleNamespace(Client=fake_client_ctor))

    media = SimpleNamespace(
        start_recording=MagicMock(),
        start_playing=MagicMock(),
    )
    robot = SimpleNamespace(media=media)
    stream = LocalStream(MagicMock(), robot, settings_app=FastAPI(), instance_path=str(tmp_path))
    stream._active_backend_name = "openai"

    init_settings_ui = MagicMock()
    monkeypatch.setattr(stream, "init_admin_ui", init_settings_ui)
    monkeypatch.setattr(stream, "_has_required_key", MagicMock(side_effect=[False, False]))
    monkeypatch.setattr("robot_comic.console.time.sleep", MagicMock(side_effect=KeyboardInterrupt))

    stream.launch()

    fake_client_ctor.assert_not_called()
    init_settings_ui.assert_called_once()
    media.start_recording.assert_not_called()
    media.start_playing.assert_not_called()


# ---------------------------------------------------------------------------
# role=user_partial INFO-log dedup (#302)
# ---------------------------------------------------------------------------


def _new_stream_for_log_dedup() -> LocalStream:
    """Construct a bare LocalStream suitable for poking ``_log_handler_message``."""
    handler = MagicMock()
    handler._clear_queue = None
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    return LocalStream(handler, robot)


def test_log_handler_message_dedups_identical_user_partials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """5 identical user_partial rows -> 1 INFO + 4 DEBUG (issue #302)."""
    stream = _new_stream_for_log_dedup()

    with caplog.at_level("DEBUG", logger="robot_comic.console"):
        for _ in range(5):
            stream._log_handler_message("user_partial", "hello there")

    partial_records = [
        r for r in caplog.records if r.name == "robot_comic.console" and "user_partial" in r.getMessage()
    ]
    info_count = sum(1 for r in partial_records if r.levelname == "INFO")
    debug_count = sum(1 for r in partial_records if r.levelname == "DEBUG")
    assert info_count == 1, f"expected exactly 1 INFO, got {info_count}: {[r.getMessage() for r in partial_records]}"
    assert debug_count == 4, f"expected exactly 4 DEBUG, got {debug_count}"


def test_log_handler_message_emits_info_for_each_progressive_partial(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each user_partial with new text re-INFOs (no dedup)."""
    stream = _new_stream_for_log_dedup()
    progressive = ["he", "hello", "hello th", "hello there", "hello there friend"]

    with caplog.at_level("DEBUG", logger="robot_comic.console"):
        for text in progressive:
            stream._log_handler_message("user_partial", text)

    partial_records = [
        r for r in caplog.records if r.name == "robot_comic.console" and "user_partial" in r.getMessage()
    ]
    info_count = sum(1 for r in partial_records if r.levelname == "INFO")
    debug_count = sum(1 for r in partial_records if r.levelname == "DEBUG")
    assert info_count == 5
    assert debug_count == 0


def test_log_handler_message_resets_dedup_on_utterance_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Final role=user row resets the tracker so the next utterance's first partial INFOs."""
    stream = _new_stream_for_log_dedup()

    with caplog.at_level("DEBUG", logger="robot_comic.console"):
        # First utterance: identical partials collapse, then final user row closes it.
        stream._log_handler_message("user_partial", "hello there")
        stream._log_handler_message("user_partial", "hello there")
        stream._log_handler_message("user", "hello there")
        # New utterance: the *very first* partial happens to be the same text
        # as the previous utterance's last partial — it must still INFO.
        stream._log_handler_message("user_partial", "hello there")

    msgs = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "robot_comic.console"]
    partial_infos = [m for m in msgs if m[0] == "INFO" and "role=user_partial" in m[1]]
    partial_debugs = [m for m in msgs if m[0] == "DEBUG" and "role=user_partial" in m[1]]
    final_user = [m for m in msgs if m[0] == "INFO" and "role=user content" in m[1]]
    assert len(partial_infos) == 2, f"first partial of each utterance should INFO, got: {partial_infos}"
    assert len(partial_debugs) == 1, f"only the repeat within first utterance should DEBUG, got: {partial_debugs}"
    assert len(final_user) == 1


def test_log_handler_message_truncates_long_content_and_dedups_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Truncation must not break dedup: two long identical contents -> 1 INFO + 1 DEBUG."""
    stream = _new_stream_for_log_dedup()
    long_text = "a" * 2000

    with caplog.at_level("DEBUG", logger="robot_comic.console"):
        stream._log_handler_message("user_partial", long_text)
        stream._log_handler_message("user_partial", long_text)

    partial_records = [
        r for r in caplog.records if r.name == "robot_comic.console" and "user_partial" in r.getMessage()
    ]
    info_count = sum(1 for r in partial_records if r.levelname == "INFO")
    debug_count = sum(1 for r in partial_records if r.levelname == "DEBUG")
    assert info_count == 1
    assert debug_count == 1
    # Sanity: the INFO line is truncated.
    info_msg = next(r.getMessage() for r in partial_records if r.levelname == "INFO")
    assert info_msg.endswith("…")
