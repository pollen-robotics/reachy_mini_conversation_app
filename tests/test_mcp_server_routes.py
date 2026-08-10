"""Tests for the custom MCP server management methods."""

import dataclasses
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.mcp_servers import (
    McpServerAuth,
    InstalledMcpServer,
    InstalledMcpServersManifest,
    read_mcp_servers,
    write_mcp_servers,
)
from reachy_mini_conversation_app.tool_spaces import InstalledToolSpacesManifest, write_installed_tool_spaces
from reachy_mini_conversation_app.profile_store import write_profile
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_names
from reachy_mini_conversation_app.mcp_server_routes import register_mcp_server_methods
from reachy_mini_conversation_app.remote_tool_sources import CachedRemoteTool


SERVER_ALIAS = "acme"
SERVER_URL = "https://mcp.example.com/mcp"
TOOL_NAME = f"{SERVER_ALIAS}__ping"
TOKEN_ENV = "ACME_MCP_TOKEN"


def _resolved_server(auth: McpServerAuth | None = None) -> InstalledMcpServer:
    return InstalledMcpServer(
        alias=SERVER_ALIAS,
        url=SERVER_URL,
        auth=auth,
        tools=[
            CachedRemoteTool(
                local_name=TOOL_NAME,
                client_tool_name=TOOL_NAME,
                remote_name="ping",
                description="Ping the server",
                parameters_schema={"type": "object", "properties": {}, "required": []},
            )
        ],
    )


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    instance_path = tmp_path / "instance"
    profiles_root = tmp_path / "profiles"
    write_profile("default", profiles_root / "default", "Default profile.", ["dance"])
    write_installed_tool_spaces(instance_path, InstalledToolSpacesManifest(spaces=[]))
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr("reachy_mini_conversation_app.profile_store.DEFAULT_PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.mcp_servers.resolve_mcp_server_sync",
        lambda server: dataclasses.replace(server, tools=_resolved_server().tools),
    )
    return instance_path


def _rpc_call(client: TestClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = websocket.receive_json()
        return response


def _mount_rpc(
    instance_path: Path | None,
    persist_env_values: Any = None,
    get_loop: MagicMock | None = None,
    restart_conversation: AsyncMock | None = None,
) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()
    register_mcp_server_methods(
        rpc,
        get_loop or MagicMock(return_value=None),
        restart_conversation or AsyncMock(),
        persist_env_values or (lambda updates: "persisted"),
        instance_path=instance_path,
    )
    rpc.mount(app)
    return TestClient(app)


def test_add_list_remove_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a server publishes it to the inventory without enabling its tools anywhere."""
    instance_path = _configure(tmp_path, monkeypatch)
    client = _mount_rpc(instance_path)

    added = _rpc_call(client, "mcp_servers.add", {"alias": SERVER_ALIAS, "url": SERVER_URL})["result"]
    assert added["servers"] == [
        {
            "alias": SERVER_ALIAS,
            "url": SERVER_URL,
            "tool_count": 1,
            "token_env": None,
            "token_set": False,
        }
    ]
    assert "ready to assign to personalities" in added["message"]
    # install_only: the tool exists in the inventory but no profile enables it yet.
    assert TOOL_NAME not in read_profile_tool_names("default", instance_path)
    assert _rpc_call(client, "mcp_servers.list")["result"]["servers"] == added["servers"]

    removed = _rpc_call(client, "mcp_servers.remove", {"alias": SERVER_ALIAS})["result"]
    assert removed["servers"] == []
    assert read_mcp_servers(instance_path).servers == []


def test_add_reports_token_requirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A server configured with a token variable reports whether that variable is set."""
    instance_path = _configure(tmp_path, monkeypatch)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    client = _mount_rpc(instance_path)

    added = _rpc_call(
        client,
        "mcp_servers.add",
        {"alias": SERVER_ALIAS, "url": SERVER_URL, "token_env": TOKEN_ENV},
    )["result"]
    assert added["servers"][0]["token_env"] == TOKEN_ENV
    assert added["servers"][0]["token_set"] is False

    monkeypatch.setenv(TOKEN_ENV, "a-real-token")
    assert _rpc_call(client, "mcp_servers.list")["result"]["servers"][0]["token_set"] is True


def test_save_token_persists_and_rebuilds_the_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving a token writes the named variable and reloads tools, since a client reads its token when built."""
    instance_path = _configure(tmp_path, monkeypatch)
    saved: dict[str, str] = {}

    def _persist(updates: dict[str, str]) -> str:
        saved.update(updates)
        monkeypatch.setenv(TOKEN_ENV, updates[TOKEN_ENV])
        return "persisted"

    initialized: list[bool] = []
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_settings.initialize_tools",
        lambda instance_path=None, force=False: initialized.append(force),
    )
    client = _mount_rpc(instance_path, persist_env_values=_persist)
    write_mcp_servers(
        instance_path,
        InstalledMcpServersManifest(
            servers=[_resolved_server(auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )

    response = _rpc_call(client, "mcp_servers.save_token", {"alias": SERVER_ALIAS, "token": "s3cr#t"})["result"]
    assert saved == {TOKEN_ENV: "s3cr#t"}
    assert response["servers"][0]["token_set"] is True
    assert initialized == [True]


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        pytest.param({"alias": SERVER_ALIAS, "token": "   "}, "empty_token", id="blank-token"),
        pytest.param({"alias": SERVER_ALIAS}, "empty_token", id="missing-token"),
        pytest.param({"alias": "", "token": "x"}, "invalid_mcp_alias", id="blank-alias"),
        pytest.param({"alias": "nope", "token": "x"}, "unknown_mcp_server", id="unconfigured-server"),
    ],
)
def test_save_token_rejects_bad_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, object],
    reason: str,
) -> None:
    """Bad token saves fail with a stable reason instead of writing anything."""
    instance_path = _configure(tmp_path, monkeypatch)
    saved: dict[str, str] = {}
    client = _mount_rpc(instance_path, persist_env_values=lambda updates: saved.update(updates) or "persisted")
    write_mcp_servers(
        instance_path,
        InstalledMcpServersManifest(
            servers=[_resolved_server(auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )

    assert _rpc_call(client, "mcp_servers.save_token", params)["error"]["data"]["reason"] == reason
    assert saved == {}


def test_save_token_reports_a_value_that_cannot_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token the instance `.env` cannot represent is refused with actionable copy."""
    instance_path = _configure(tmp_path, monkeypatch)

    def _refuse(updates: dict[str, str]) -> str:
        raise ValueError("Values must not contain line breaks or '${'.")

    client = _mount_rpc(instance_path, persist_env_values=_refuse)
    write_mcp_servers(
        instance_path,
        InstalledMcpServersManifest(
            servers=[_resolved_server(auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )

    error = _rpc_call(client, "mcp_servers.save_token", {"alias": SERVER_ALIAS, "token": "a\nb"})["error"]
    assert error["data"]["reason"] == "invalid_token"


def test_save_token_warns_when_it_could_not_be_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token applied to the process but not written to disk must say it will not survive a restart."""
    instance_path = _configure(tmp_path, monkeypatch)
    client = _mount_rpc(instance_path, persist_env_values=lambda updates: "failed")
    write_mcp_servers(
        instance_path,
        InstalledMcpServersManifest(
            servers=[_resolved_server(auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )

    message = _rpc_call(client, "mcp_servers.save_token", {"alias": SERVER_ALIAS, "token": "x"})["result"]["message"]
    assert "will not survive a restart" in message


def test_save_token_reports_session_only_in_terminal_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no managed instance there is no `.env` to write, so the token lasts for the session only."""
    _configure(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    client = _mount_rpc(None, persist_env_values=lambda updates: "session")
    write_mcp_servers(
        None,
        InstalledMcpServersManifest(
            servers=[_resolved_server(auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )

    message = _rpc_call(client, "mcp_servers.save_token", {"alias": SERVER_ALIAS, "token": "x"})["result"]["message"]
    assert "for this session" in message


def test_list_reports_an_unreadable_manifest_instead_of_an_empty_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A damaged manifest must surface as an error, not as 'no servers configured'."""
    instance_path = _configure(tmp_path, monkeypatch)
    instance_path.mkdir(parents=True, exist_ok=True)
    (instance_path / "mcp_servers.json").write_text("{not valid json", encoding="utf-8")
    client = _mount_rpc(instance_path)

    error = _rpc_call(client, "mcp_servers.list")["error"]
    assert error["data"]["reason"] == "mcp_servers_unavailable"


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        pytest.param({"url": SERVER_URL}, "invalid_mcp_alias", id="missing-alias"),
        pytest.param({"alias": SERVER_ALIAS}, "invalid_mcp_url", id="missing-url"),
        pytest.param({"alias": "bad__alias", "url": SERVER_URL}, "invalid_mcp_server", id="ambiguous-alias"),
        pytest.param(
            {"alias": SERVER_ALIAS, "url": "http://example.com/mcp"},
            "invalid_mcp_server",
            id="public-plain-http",
        ),
        pytest.param(
            {"alias": SERVER_ALIAS, "url": SERVER_URL, "token_env": "BAD NAME"},
            "invalid_mcp_token_env",
            id="invalid-token-env",
        ),
        pytest.param(
            {"alias": SERVER_ALIAS, "url": SERVER_URL, "request_timeout_s": 0},
            "invalid_mcp_timeout",
            id="non-positive-timeout",
        ),
    ],
)
def test_add_rejects_invalid_input_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, object],
    reason: str,
) -> None:
    """Invalid configuration is refused with a stable reason and nothing is written."""
    instance_path = _configure(tmp_path, monkeypatch)
    client = _mount_rpc(instance_path)

    assert _rpc_call(client, "mcp_servers.add", params)["error"]["data"]["reason"] == reason
    assert read_mcp_servers(instance_path).servers == []


def test_add_rejects_an_alias_already_claimed_by_a_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aliases namespace tool IDs across both sources, so a cross-source collision is refused."""
    instance_path = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.mcp_servers._space_aliases",
        lambda path: {SERVER_ALIAS},
    )
    client = _mount_rpc(instance_path)

    error = _rpc_call(client, "mcp_servers.add", {"alias": SERVER_ALIAS, "url": SERVER_URL})["error"]
    assert error["data"]["reason"] == "mcp_server_alias_conflict"
    assert read_mcp_servers(instance_path).servers == []


def test_remove_rejects_a_server_that_is_not_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing an unknown alias reports it rather than silently succeeding."""
    instance_path = _configure(tmp_path, monkeypatch)
    client = _mount_rpc(instance_path)

    error = _rpc_call(client, "mcp_servers.remove", {"alias": "ghost"})["error"]
    assert error["data"]["reason"] == "mcp_server_not_configured"


def test_locked_mode_exposes_the_inventory_but_rejects_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A locked profile may read the server list but may not change it."""
    instance_path = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr("reachy_mini_conversation_app.mcp_server_routes.LOCKED_PROFILE", "kiosk")
    client = _mount_rpc(instance_path)

    assert _rpc_call(client, "mcp_servers.list")["result"]["editable"] is False
    for method, params in (
        ("mcp_servers.add", {"alias": SERVER_ALIAS, "url": SERVER_URL}),
        ("mcp_servers.remove", {"alias": SERVER_ALIAS}),
        ("mcp_servers.save_token", {"alias": SERVER_ALIAS, "token": "x"}),
    ):
        assert _rpc_call(client, method, params)["error"]["data"]["reason"] == "profile_locked"
