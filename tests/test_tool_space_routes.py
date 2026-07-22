"""Tests for Hugging Face Space and profile-tool management methods."""

import asyncio
import threading
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_mini_conversation_app.tool_spaces import (
    InstalledToolSpace,
    InstalledToolSpaceTool,
    InstalledToolSpacesManifest,
    read_installed_tool_spaces,
    write_installed_tool_spaces,
)
from reachy_mini_conversation_app.profile_store import write_profile
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_tool_names,
    read_profile_tool_override,
)
from reachy_mini_conversation_app.tool_space_routes import register_tool_space_methods
from reachy_mini_conversation_app.profile_tool_routes import register_profile_tool_methods


SPACE_SLUG = "example/search-tool"
SPACE_ALIAS = "example_search_tool"
TOOL_NAME = f"{SPACE_ALIAS}__search_web"


def _resolved_space() -> InstalledToolSpace:
    return InstalledToolSpace(
        slug=SPACE_SLUG,
        alias=SPACE_ALIAS,
        mcp_url="https://example-search-tool.hf.space/gradio_api/mcp/",
        private=False,
        tools=[
            InstalledToolSpaceTool(
                local_name=TOOL_NAME,
                client_tool_name=f"{SPACE_ALIAS}__search_tool_search_web",
                remote_name="search_tool_search_web",
                description="Search the web",
                parameters_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
    )


def _configure_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    instance_path = tmp_path / "instance"
    profiles_root = tmp_path / "profiles"
    write_profile("default", profiles_root / "default", "Default profile.", ["dance"])
    write_profile("guide", profiles_root / "guide", "Guide profile.", ["camera"])
    write_installed_tool_spaces(instance_path, InstalledToolSpacesManifest(spaces=[]))
    monkeypatch.setattr(config, "INSTANCE_PATH", instance_path)
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr("reachy_mini_conversation_app.profile_store.DEFAULT_PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.resolve_tool_space_sync",
        lambda slug: _resolved_space(),
    )
    return instance_path, profiles_root


def _rpc_call(client: TestClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = websocket.receive_json()
        return response


def _mount_rpc(
    instance_path: Path,
    get_loop: MagicMock,
    restart_conversation: AsyncMock,
) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()
    register_tool_space_methods(
        rpc,
        get_loop,
        restart_conversation,
        instance_path=instance_path,
    )
    register_profile_tool_methods(
        rpc,
        get_loop,
        restart_conversation,
        instance_path=instance_path,
    )
    rpc.mount(app)
    return TestClient(app)


def test_web_install_adds_global_inventory_without_enabling_a_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installing from the web should make tools available without granting profile access."""
    instance_path, profiles_root = _configure_profiles(tmp_path, monkeypatch)
    default_profile_text = (profiles_root / "default" / "profile.md").read_text(encoding="utf-8")
    initialize_tools = MagicMock()
    monkeypatch.setattr("reachy_mini_conversation_app.tool_settings.initialize_tools", initialize_tools)
    restart_conversation = AsyncMock()
    client = _mount_rpc(instance_path, MagicMock(return_value=None), restart_conversation)

    response = _rpc_call(client, "tool_spaces.add", {"slug": SPACE_SLUG})

    added = response["result"]
    assert added["spaces"] == [{"slug": SPACE_SLUG, "private": False, "tool_count": 1}]
    assert added["editable"] is True
    assert "ready to assign to personalities" in added["message"]
    assert read_profile_tool_override("default", instance_path) is None
    assert (profiles_root / "default" / "profile.md").read_text(encoding="utf-8") == default_profile_text
    initialize_tools.assert_not_called()
    restart_conversation.assert_not_called()

    profile_tools = _rpc_call(client, "profile_tools.get", {"profile": "default"})["result"]
    assert profile_tools["enabled_tools"] == ["dance"]
    assert TOOL_NAME in {tool["id"] for tool in profile_tools["available_tools"]}
    assert _rpc_call(client, "tool_spaces.list")["result"] == {"spaces": added["spaces"], "editable": True}


def test_preinstalled_space_tools_are_available_to_every_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundled Space inventory should be selectable without leaking default-profile access."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "mars_rover")
    client = _mount_rpc(tmp_path, MagicMock(return_value=None), AsyncMock())

    response = _rpc_call(client, "profile_tools.get", {"profile": "mars_rover"})

    payload = response["result"]
    available_ids = {tool["id"] for tool in payload["available_tools"]}
    preinstalled_tool_ids = {
        "pollen_robotics_reachy_mini_search_tool__search_web",
        "pollen_robotics_reachy_mini_weather_tool__get_weather",
        "pollen_robotics_reachy_mini_time_tool__get_time",
    }
    assert preinstalled_tool_ids <= available_ids
    assert preinstalled_tool_ids.isdisjoint(payload["enabled_tools"])


def test_profile_tools_save_and_reset_control_one_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile-tool updates should persist independently and reset to authored defaults."""
    instance_path, profiles_root = _configure_profiles(tmp_path, monkeypatch)
    guide_profile_text = (profiles_root / "guide" / "profile.md").read_text(encoding="utf-8")
    initialize_tools = MagicMock()
    monkeypatch.setattr("reachy_mini_conversation_app.tool_settings.initialize_tools", initialize_tools)
    client = _mount_rpc(instance_path, MagicMock(return_value=None), AsyncMock())
    assert "result" in _rpc_call(client, "tool_spaces.add", {"slug": SPACE_SLUG})

    update_response = _rpc_call(
        client,
        "profile_tools.save",
        {"profile": "guide", "enabled_tools": ["camera", TOOL_NAME, TOOL_NAME]},
    )

    updated = update_response["result"]
    assert updated["profile"] == "guide"
    assert updated["is_active"] is False
    assert updated["overridden"] is True
    assert updated["enabled_tools"] == ["camera", TOOL_NAME]
    assert "next time this personality is selected" in updated["message"]
    assert read_profile_tool_names("default", instance_path) == ["dance"]
    assert (profiles_root / "guide" / "profile.md").read_text(encoding="utf-8") == guide_profile_text
    initialize_tools.assert_not_called()

    reset_response = _rpc_call(client, "profile_tools.reset", {"profile": "guide"})

    reset = reset_response["result"]
    assert reset["overridden"] is False
    assert reset["enabled_tools"] == ["camera"]
    assert "next time this personality is selected" in reset["message"]
    assert read_profile_tool_override("guide", instance_path) is None
    initialize_tools.assert_not_called()


def test_remove_tool_space_disables_its_tools_in_every_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a Space should clean its tool IDs from every profile selection."""
    instance_path, _ = _configure_profiles(tmp_path, monkeypatch)
    initialize_tools = MagicMock()
    monkeypatch.setattr("reachy_mini_conversation_app.tool_settings.initialize_tools", initialize_tools)
    client = _mount_rpc(instance_path, MagicMock(return_value=None), AsyncMock())
    assert "result" in _rpc_call(client, "tool_spaces.add", {"slug": SPACE_SLUG})
    assert "result" in _rpc_call(
        client,
        "profile_tools.save",
        {"profile": "default", "enabled_tools": ["dance", TOOL_NAME]},
    )
    assert "result" in _rpc_call(
        client,
        "profile_tools.save",
        {"profile": "guide", "enabled_tools": ["camera", TOOL_NAME]},
    )
    initialize_tools.reset_mock()

    remove_response = _rpc_call(client, "tool_spaces.remove", {"slug": SPACE_SLUG})

    removed = remove_response["result"]
    assert removed["spaces"] == []
    assert "Disabled 2 tools across personalities" in removed["message"]
    assert read_profile_tool_names("default", instance_path) == ["dance"]
    assert read_profile_tool_names("guide", instance_path) == ["camera"]
    initialize_tools.assert_called_once_with(instance_path=instance_path, force=True)
    assert read_installed_tool_spaces(instance_path).spaces == []


def test_add_tool_space_rejects_invalid_slug_without_network_access(tmp_path: Path) -> None:
    """The UI method should only accept Hugging Face owner/Space slugs."""
    app = FastAPI()
    rpc = JsonRpcServer()
    register_tool_space_methods(
        rpc,
        lambda: None,
        AsyncMock(),
        instance_path=tmp_path,
    )
    rpc.mount(app)

    response = _rpc_call(TestClient(app), "tool_spaces.add", {"slug": "https://example.com/mcp"})

    assert response["error"]["data"]["reason"] == "invalid_tool_space_slug"


def test_locked_mode_exposes_inventory_but_rejects_tool_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Locked variants should keep tool settings visible and make every mutation read-only."""
    instance_path, _ = _configure_profiles(tmp_path, monkeypatch)
    monkeypatch.setattr("reachy_mini_conversation_app.tool_space_routes.LOCKED_PROFILE", "default")
    monkeypatch.setattr("reachy_mini_conversation_app.profile_tool_routes.LOCKED_PROFILE", "default")
    client = _mount_rpc(instance_path, MagicMock(return_value=None), AsyncMock())

    assert _rpc_call(client, "tool_spaces.list")["result"]["editable"] is False
    assert _rpc_call(client, "profile_tools.get", {"profile": "default"})["result"]["editable"] is False
    assert _rpc_call(client, "tool_spaces.add", {"slug": SPACE_SLUG})["error"]["data"]["reason"] == "profile_locked"
    assert _rpc_call(client, "tool_spaces.remove", {"slug": SPACE_SLUG})["error"]["data"]["reason"] == "profile_locked"
    assert (
        _rpc_call(client, "profile_tools.save", {"profile": "default", "enabled_tools": []})["error"]["data"]["reason"]
        == "profile_locked"
    )
    assert (
        _rpc_call(client, "profile_tools.reset", {"profile": "default"})["error"]["data"]["reason"] == "profile_locked"
    )


def test_active_profile_tool_update_restarts_a_running_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing active profile tools should reconnect the running conversation."""
    instance_path, _ = _configure_profiles(tmp_path, monkeypatch)
    monkeypatch.setattr("reachy_mini_conversation_app.tool_settings.initialize_tools", MagicMock())
    conversation_loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=conversation_loop.run_forever)
    loop_thread.start()
    restart_called = threading.Event()

    async def _restart_conversation(reason: str) -> None:
        assert reason == "profile_tools_changed"
        restart_called.set()

    try:
        app = FastAPI()
        rpc = JsonRpcServer()
        register_tool_space_methods(
            rpc,
            lambda: conversation_loop,
            _restart_conversation,
            instance_path=instance_path,
        )
        register_profile_tool_methods(
            rpc,
            lambda: conversation_loop,
            _restart_conversation,
            instance_path=instance_path,
        )
        rpc.mount(app)
        client = TestClient(app)
        assert "result" in _rpc_call(client, "tool_spaces.add", {"slug": SPACE_SLUG})

        response = _rpc_call(
            client,
            "profile_tools.save",
            {"profile": "default", "enabled_tools": ["dance", TOOL_NAME]},
        )

        assert "Reconnecting the conversation" in response["result"]["message"]
        assert restart_called.wait(timeout=1.0)
    finally:
        conversation_loop.call_soon_threadsafe(conversation_loop.stop)
        loop_thread.join(timeout=1.0)
        conversation_loop.close()


def test_saved_tool_change_reports_success_when_runtime_reload_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persisted selection should not be reported as a failed save when live reload fails."""
    instance_path, _ = _configure_profiles(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_settings.initialize_tools",
        MagicMock(side_effect=RuntimeError("reload failed")),
    )
    client = _mount_rpc(instance_path, MagicMock(return_value=None), AsyncMock())

    response = _rpc_call(
        client,
        "profile_tools.save",
        {"profile": "default", "enabled_tools": ["dance"]},
    )

    assert "Restart the conversation app" in response["result"]["message"]
    assert read_profile_tool_override("default", instance_path) == ["dance"]
