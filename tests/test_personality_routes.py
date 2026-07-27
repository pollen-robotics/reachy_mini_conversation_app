"""Tests for personality editing routes."""

import asyncio
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
import reachy_mini_conversation_app.personality as personality_mod
from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_mini_conversation_app.profile_store import (
    write_profile,
    read_profile_from_directory,
    read_packaged_default_profile,
)
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_tool_override,
    write_profile_tool_override,
)
from reachy_mini_conversation_app.personality_routes import (
    build_personality_ops,
    register_personality_methods,
)
from reachy_mini_conversation_app.profile_tool_routes import register_profile_tool_methods


def _rpc_call(client: TestClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = websocket.receive_json()
        return response


def _client() -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()
    register_personality_methods(rpc, build_personality_ops(MagicMock(), lambda: None))
    rpc.mount(app)
    return TestClient(app)


def test_new_personality_inherits_packaged_default_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creating a personality should start from the bundled tool baseline."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    response = _rpc_call(
        _client(),
        "personalities.save",
        {"name": "guide", "instructions": "Be a concise guide.", "greeting": "Hello there."},
    )

    result = response["result"]
    assert result["value"] == "user_personalities/guide"
    assert "user_personalities/guide" in result["choices"]
    profile = read_profile_from_directory("guide", tmp_path / "user_personalities" / "guide")
    assert profile.instructions == "Be a concise guide."
    assert profile.greeting == "Hello there."
    assert profile.voice == "Aiden"
    assert profile.default_tools == read_packaged_default_profile().default_tools
    loaded = _rpc_call(_client(), "personalities.load", {"name": "user_personalities/guide"})["result"]
    assert {field: loaded[field] for field in ("instructions", "greeting", "voice")} == {
        "instructions": "Be a concise guide.",
        "greeting": "Hello there.",
        "voice": "Aiden",
    }
    assert loaded["enabled_tools"] == list(profile.default_tools)


def test_personality_creation_does_not_overwrite_existing_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creating a duplicate personality should preserve the existing profile."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    client = _client()
    first = _rpc_call(client, "personalities.save", {"name": "guide", "instructions": "Original."})

    duplicate = _rpc_call(client, "personalities.save", {"name": "guide", "instructions": "Replacement."})

    assert "result" in first
    assert duplicate["error"]["data"]["reason"] == "profile_exists"
    profile = read_profile_from_directory("guide", tmp_path / "user_personalities" / "guide")
    assert profile.instructions == "Original."


def test_personality_save_rejects_blank_instructions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty instructions should be a client error, not a failed storage write."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)

    response = _rpc_call(_client(), "personalities.save", {"name": "guide", "instructions": "   "})

    assert response["error"]["data"]["reason"] == "invalid_instructions"
    assert not (tmp_path / "user_personalities" / "guide").exists()


def test_personality_save_rejects_unsafe_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web API should enforce the same safe names as headless creation."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)

    response = _rpc_call(_client(), "personalities.save", {"name": "../guide", "instructions": "Unsafe."})

    assert response["error"]["data"]["reason"] == "invalid_name"
    assert not (tmp_path / "guide").exists()


def test_editing_personality_preserves_tool_defaults_and_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt edits must not change either layer of personality tool access."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    profile_directory = tmp_path / "user_personalities" / "guide"
    write_profile("guide", profile_directory, "Old instructions.", ["dance"])
    write_profile_tool_override("user_personalities/guide", ["camera"], tmp_path)

    response = _rpc_call(
        _client(),
        "personalities.save",
        {
            "name": "guide",
            "instructions": "New instructions.",
            "greeting": "Hello there.",
            "overwrite": True,
        },
    )

    assert "result" in response
    profile = read_profile_from_directory("guide", profile_directory)
    assert profile.instructions == "New instructions."
    assert profile.greeting == "Hello there."
    assert profile.default_tools == ("dance",)
    assert read_profile_tool_override("user_personalities/guide", tmp_path) == ["camera"]


def test_personality_save_materializes_submitted_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitted profile tools should remain portable without a local override."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    profile_directory = tmp_path / "user_personalities" / "guide"
    write_profile("guide", profile_directory, "Old instructions.", ["dance"])
    write_profile_tool_override("user_personalities/guide", ["camera"], tmp_path)

    response = _rpc_call(
        _client(),
        "personalities.save",
        {
            "name": "guide",
            "instructions": "New instructions.",
            "tools_text": "go_to_sleep\n",
            "overwrite": True,
        },
    )

    assert "result" in response
    profile = read_profile_from_directory("guide", profile_directory)
    assert profile.default_tools == ("go_to_sleep",)
    assert read_profile_tool_override("user_personalities/guide", tmp_path) is None
    loaded = _rpc_call(_client(), "personalities.load", {"name": "user_personalities/guide"})["result"]
    assert loaded["enabled_tools"] == ["go_to_sleep"]
    assert loaded["tools_text"] == "go_to_sleep\n"


def test_personality_save_rolls_back_tool_override_if_profile_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed profile write must restore the previous tool override."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    profile_directory = tmp_path / "user_personalities" / "guide"
    write_profile(
        "guide",
        profile_directory,
        "Old instructions.",
        ["dance"],
        greeting="Old greeting.",
        hidden=True,
    )
    write_profile_tool_override("user_personalities/guide", ["camera"], tmp_path)

    monkeypatch.setattr(personality_mod, "write_profile", MagicMock(side_effect=OSError("profile write failed")))

    response = _rpc_call(
        _client(),
        "personalities.save",
        {
            "name": "guide",
            "instructions": "New instructions.",
            "greeting": "New greeting.",
            "tools_text": "go_to_sleep\n",
            "overwrite": True,
        },
    )

    assert response["error"]["data"]["reason"] == "profile_save_failed"
    profile = read_profile_from_directory("guide", profile_directory)
    assert profile.instructions == "Old instructions."
    assert profile.greeting == "Old greeting."
    assert profile.hidden is True
    assert read_profile_tool_override("user_personalities/guide", tmp_path) == ["camera"]


def test_external_profiles_keep_canonical_packaged_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default choice should remain available without an external default directory."""
    external_profiles_root = tmp_path / "external_profiles"
    write_profile("guide", external_profiles_root / "guide", "Be a guide.", ["dance"])
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path / "instance")
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    client = _client()
    listing = _rpc_call(client, "personalities.list")["result"]
    loaded = _rpc_call(client, "personalities.load", {"name": "default"})["result"]

    assert listing["choices"] == ["default", "guide"]
    assert listing["current"] == "default"
    assert listing["startup"] == "default"
    assert "Reachy Mini" in loaded["instructions"]
    assert not (external_profiles_root / "default").exists()


def test_profile_load_failure_is_not_returned_as_editable_content() -> None:
    """Missing profile content should produce a proper API error."""
    response = _rpc_call(_client(), "personalities.load", {"name": "missing"})

    assert response["error"]["data"]["reason"] == "profile_unavailable"
    assert "result" not in response


def test_applying_default_persists_runtime_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical default ID should map to no custom runtime profile."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    app = FastAPI()
    handler = MagicMock()
    handler.get_current_voice.return_value = "Aiden"
    persist_personality = MagicMock()
    rpc = JsonRpcServer()
    ops = build_personality_ops(
        handler,
        lambda: None,
        persist_personality=persist_personality,
    )
    register_personality_methods(rpc, ops)
    rpc.mount(app)

    response = _rpc_call(TestClient(app), "personalities.apply", {"name": "default", "persist": True})

    assert response["result"]["startup"] == "default"
    persist_personality.assert_called_once_with(None, "Aiden")


def test_force_reloads_active_personality(monkeypatch: pytest.MonkeyPatch) -> None:
    """An active profile edit must reload the running conversation."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/guide")
    app = FastAPI()
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="Personality reloaded.")
    ops = build_personality_ops(handler, lambda: asyncio.get_running_loop())
    rpc = JsonRpcServer()
    register_personality_methods(rpc, ops)
    rpc.mount(app)

    response = _rpc_call(
        TestClient(app),
        "personalities.apply",
        {"name": "user_personalities/guide", "force": True},
    )

    assert response["result"]["status"] == "Personality reloaded."
    handler.apply_personality.assert_awaited_once_with("user_personalities/guide")


def test_external_tools_are_available_without_autoload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autoload should not control whether an external tool can be selected."""
    external_tools_root = tmp_path / "external_tools"
    external_tools_root.mkdir()
    (external_tools_root / "ext_ping.py").write_text("# selectable external tool\n", encoding="utf-8")
    (external_tools_root / "_private.py").write_text("# ignored\n", encoding="utf-8")
    (external_tools_root / "bad-name.py").write_text("# ignored\n", encoding="utf-8")
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    monkeypatch.setattr(config, "TOOLS_DIRECTORY", external_tools_root)
    monkeypatch.setattr(config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    app = FastAPI()
    rpc = JsonRpcServer()
    register_profile_tool_methods(
        rpc,
        lambda: None,
        AsyncMock(),
        instance_path=tmp_path,
    )
    rpc.mount(app)

    response = _rpc_call(TestClient(app), "profile_tools.get", {"profile": "default"})["result"]

    external_tools = [tool for tool in response["available_tools"] if tool["kind"] == "external"]
    assert external_tools == [
        {
            "id": "ext_ping",
            "kind": "external",
            "source": "External",
            "description": "",
        }
    ]
