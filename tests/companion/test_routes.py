"""Tests for the global background-assistant RPC settings."""

from types import SimpleNamespace
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import reachy_mini_conversation_app.companion.routes as routes_module
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.companion.settings import (
    CompanionSettings,
    write_companion_settings,
)
from reachy_mini_conversation_app.companion.provisioner import AssistantNamespace, AssistantNamespaceKind


def _rpc_call(app: FastAPI, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    with TestClient(app).websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: object = websocket.receive_json()
        assert isinstance(response, dict)
        return response


def test_companion_setting_is_global_and_never_exposes_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RPC persists one boolean and returns no connection secrets."""
    monkeypatch.setenv("SMOL_ASSISTANT_API_TOKEN", "a" * 32)
    monkeypatch.setattr(config, "COMPANION_ENABLED", True)
    saved_settings = CompanionSettings(
        api_url="https://alice-smolagents-assistant-reachy-mini.hf.space",
        api_token="b" * 32,
    )
    write_companion_settings(tmp_path, saved_settings)
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(
        MagicMock(),
        robot,
        settings_app=app,
        instance_path=str(tmp_path),
        companion_tasks=MagicMock(),
    )
    stream._init_settings_ui_if_needed()

    initial = _rpc_call(app, "companion.config.get")["result"]
    saved = _rpc_call(app, "companion.config.save", {"enabled": False})["result"]

    assert initial["configured"] is True
    assert initial["enabled"] is True
    assert initial["setup"]["state"] == "ready"
    assert saved["configured"] is True
    assert saved["enabled"] is False
    assert set(saved) == {"configured", "enabled", "setup", "message"}
    assert saved_settings.api_token not in str((initial, saved))


def test_missing_saved_assistant_is_not_reported_as_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deleted managed Space fails closed and drops its stale ownership links."""
    monkeypatch.delenv("SMOL_ASSISTANT_API_URL", raising=False)
    monkeypatch.delenv("SMOL_ASSISTANT_API_TOKEN", raising=False)
    monkeypatch.setattr(config, "COMPANION_ENABLED", True)
    monkeypatch.setattr(config, "COMPANION_CONFIGURED", True)
    monkeypatch.setattr(config, "HF_TOKEN", None)
    monkeypatch.setattr(routes_module, "get_token", lambda: "hf_test_credential")
    write_companion_settings(
        tmp_path,
        CompanionSettings(
            api_url="https://alice-smolagents-assistant-reachy-mini.hf.space",
            api_token="b" * 32,
        ),
    )
    unavailable = routes_module._companion_error(
        "companion_invalid_response",
        "The background assistant returned invalid JSON.",
    )
    run_on_companion_loop = AsyncMock(side_effect=unavailable)
    monkeypatch.setattr(routes_module, "_run_on_companion_loop", run_on_companion_loop)
    namespaces = (
        AssistantNamespace("alice", AssistantNamespaceKind.PERSONAL),
        AssistantNamespace("pollen-robotics", AssistantNamespaceKind.ORGANIZATION),
    )
    monkeypatch.setattr(routes_module, "list_assistant_namespaces", MagicMock(return_value=namespaces))
    setup = routes_module.CompanionSetup(tmp_path)
    start_setup = MagicMock()
    monkeypatch.setattr(setup, "start", start_setup)
    monkeypatch.setattr(routes_module, "CompanionSetup", lambda _instance_path: setup)
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(
        MagicMock(),
        robot,
        settings_app=app,
        instance_path=str(tmp_path),
        companion_tasks=MagicMock(),
    )
    stream._init_settings_ui_if_needed()

    result = _rpc_call(app, "companion.config.get")["result"]
    listed = _rpc_call(app, "companion.setup.namespaces")["result"]
    _rpc_call(app, "companion.setup.start", {"namespace": "pollen-robotics"})

    assert result["configured"] is False
    assert result["enabled"] is False
    assert result["setup"]["state"] == "failed"
    assert set(result["setup"]) == {"state", "message"}
    assert config.COMPANION_CONFIGURED is False
    assert listed["namespaces"] == [
        {"name": "alice", "kind": "personal"},
        {"name": "pollen-robotics", "kind": "organization"},
    ]
    start_setup.assert_called_once_with("hf_test_credential", "pollen-robotics")


def test_setup_uses_local_token_and_selected_organization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The browser selects a server-verified namespace without handling credentials."""
    monkeypatch.delenv("SMOL_ASSISTANT_API_URL", raising=False)
    monkeypatch.delenv("SMOL_ASSISTANT_API_TOKEN", raising=False)
    monkeypatch.setattr(config, "HF_TOKEN", "hf_server_credential")
    monkeypatch.setattr(config, "COMPANION_ENABLED", False)
    monkeypatch.setattr(config, "COMPANION_CONFIGURED", False)
    namespaces = (
        AssistantNamespace("alice", AssistantNamespaceKind.PERSONAL),
        AssistantNamespace("pollen-robotics", AssistantNamespaceKind.ORGANIZATION),
    )
    list_namespaces = MagicMock(return_value=namespaces)
    monkeypatch.setattr(routes_module, "list_assistant_namespaces", list_namespaces)
    setup = routes_module.CompanionSetup(tmp_path)
    monkeypatch.setattr(setup, "start", MagicMock())
    monkeypatch.setattr(routes_module, "CompanionSetup", lambda _instance_path: setup)
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(
        MagicMock(),
        robot,
        settings_app=app,
        instance_path=str(tmp_path),
        companion_tasks=None,
    )
    stream._init_settings_ui_if_needed()

    listed = _rpc_call(app, "companion.setup.namespaces")["result"]
    tampered = _rpc_call(
        app,
        "companion.setup.start",
        {"namespace": "pollen-robotics", "hf_token": "client-supplied"},
    )
    started = _rpc_call(app, "companion.setup.start", {"namespace": "pollen-robotics"})["result"]

    assert listed == {
        "namespaces": [
            {"name": "alice", "kind": "personal"},
            {"name": "pollen-robotics", "kind": "organization"},
        ]
    }
    assert tampered["error"]["data"]["reason"] == "invalid_companion_namespace"
    list_namespaces.assert_called_once_with("hf_server_credential")
    setup.start.assert_called_once_with("hf_server_credential", "pollen-robotics")
    assert started["configured"] is False
    assert "hf_server_credential" not in str((listed, tampered, started))
    assert "client-supplied" not in str((listed, tampered, started))
