"""Tests for the language.* and vision.* JSON-RPC methods."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.vision_routes import register_vision_methods
from reachy_mini_conversation_app.language_routes import register_language_methods
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _rpc_call(client: TestClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = websocket.receive_json()
        return response


def _language_client(applied: list[str]) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()

    def _set_language(language: str) -> str:
        applied.append(language)
        config.REALTIME_TRANSCRIPTION_LANGUAGE = language
        return "Saved."

    register_language_methods(rpc, set_language=_set_language)
    rpc.mount(app)
    return TestClient(app)


def _vision_client(deps: ToolDependencies, persisted: list[bool]) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()
    register_vision_methods(rpc, deps, persisted.append)
    rpc.mount(app)
    return TestClient(app)


def test_language_get_reports_the_active_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client reads the transcription language it is about to change."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "en")

    assert _rpc_call(_language_client([]), "language.get")["result"] == {"language": "en"}


@pytest.mark.parametrize("language", ["fr", "pt-BR"])
def test_language_set_accepts_client_codes(language: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ids the clients send are accepted and normalised."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "en")
    applied: list[str] = []

    result = _rpc_call(_language_client(applied), "language.set", {"language": language})["result"]

    assert applied == [language.lower()]
    assert result["language"] == language.lower()


@pytest.mark.parametrize("language", ["", "english", "e", 42])
def test_language_set_rejects_junk(language: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad code must not reach the transcription config."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "en")
    applied: list[str] = []

    error = _rpc_call(_language_client(applied), "language.set", {"language": language})["error"]

    assert error["data"]["reason"] == "invalid_language"
    assert applied == []


def test_vision_toggle_is_live_and_persisted() -> None:
    """Tools read camera_enabled per call, so the switch bites immediately."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_enabled=True)
    persisted: list[bool] = []
    client = _vision_client(deps, persisted)

    assert _rpc_call(client, "vision.get")["result"] == {"enabled": True}
    assert _rpc_call(client, "vision.set", {"enabled": False})["result"] == {"enabled": False}
    assert deps.camera_enabled is False
    assert persisted == [False]


def test_vision_set_rejects_non_boolean() -> None:
    """A malformed toggle leaves the camera as it was."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_enabled=True)
    persisted: list[bool] = []

    error = _rpc_call(_vision_client(deps, persisted), "vision.set", {"enabled": "off"})["error"]

    assert error["data"]["reason"] == "invalid_params"
    assert deps.camera_enabled is True
    assert persisted == []
