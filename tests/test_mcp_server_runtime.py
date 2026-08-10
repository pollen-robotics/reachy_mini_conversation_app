import sys
import json
import importlib
from types import ModuleType
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.mcp_servers as mcp_servers_mod
from reachy_mini_conversation_app.mcp_servers import (
    McpServerAuth,
    InstalledMcpServer,
    InstalledMcpServersManifest,
    write_mcp_servers,
)
from reachy_mini_conversation_app.tool_spaces import (
    InstalledToolSpace,
    InstalledToolSpacesManifest,
    write_installed_tool_spaces,
)
from reachy_mini_conversation_app.profile_store import write_profile
from reachy_mini_conversation_app.remote_tool_sources import CachedRemoteTool


SERVER_ALIAS = "example"
SERVER_URL = "http://192.168.1.50:8000/mcp"
TOOL_ID = f"{SERVER_ALIAS}__do_thing"
TOKEN_ENV = "MCP_SERVER_RUNTIME_TOKEN"


def _reload_core_tools() -> ModuleType:
    for module_name in list(sys.modules):
        if module_name.startswith("reachy_mini_conversation_app.tools."):
            sys.modules.pop(module_name, None)

    sys.modules.pop("reachy_mini_conversation_app.tools.core_tools", None)
    return importlib.import_module("reachy_mini_conversation_app.tools.core_tools")


def _installed_server(auth: McpServerAuth | None = None) -> InstalledMcpServer:
    return InstalledMcpServer(
        alias=SERVER_ALIAS,
        url=SERVER_URL,
        auth=auth,
        tools=[
            CachedRemoteTool(
                local_name=TOOL_ID,
                client_tool_name=TOOL_ID,
                remote_name="do_thing",
                description="Remote tool do_thing",
                parameters_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            )
        ],
    )


def _profile_enabling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    tool_ids: list[str],
) -> None:
    """Point config at an external profile whose authored defaults enable tool_ids."""
    external_profiles_root = tmp_path / "external_profiles"
    write_profile(profile_name, external_profiles_root / profile_name, "hello", tool_ids)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", profile_name)
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)


def _mcp_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _profile_enabling(tmp_path, monkeypatch, "mcp_profile", [TOOL_ID])


@pytest.mark.asyncio
async def test_initialize_tools_loads_enabled_generic_mcp_tools_and_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled generic MCP server tools should register from the cached manifest and dispatch."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    client = AsyncMock()
    client.call_tool.return_value = {
        "status": "ok",
        "server_alias": SERVER_ALIAS,
        "remote_tool_name": "do_thing",
        "namespaced_tool_name": TOOL_ID,
        "content_blocks": [],
        "text": "hello",
    }
    captured_servers: list[InstalledMcpServer] = []

    def _build_generic_remote_client(server: InstalledMcpServer) -> AsyncMock:
        captured_servers.append(server)
        return client

    monkeypatch.setattr(mcp_servers_mod, "build_generic_remote_client", _build_generic_remote_client)

    write_mcp_servers(None, InstalledMcpServersManifest(servers=[_installed_server()]))

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    assert TOOL_ID in core_tools_mod.ALL_TOOLS
    assert [server.alias for server in captured_servers] == [SERVER_ALIAS]
    assert captured_servers[0].tools == _installed_server().tools
    assert any(spec["name"] == TOOL_ID for spec in core_tools_mod.get_tool_specs())

    result = await core_tools_mod.dispatch_tool_call(
        TOOL_ID,
        json.dumps({"message": "hello"}),
        core_tools_mod.ToolDependencies(
            reachy_mini=object(),
            movement_manager=object(),
        ),
    )

    assert result["namespaced_tool_name"] == TOOL_ID
    assert result["mcp_server_alias"] == SERVER_ALIAS
    assert "tool_space_slug" not in result
    client.call_tool.assert_awaited_once_with(TOOL_ID, {"message": "hello"})


def test_initialize_tools_skips_generic_tools_with_missing_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing bearer token skips the server's tools with a warning: registering tools whose every call would fail unauthenticated only hides the problem."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    write_mcp_servers(
        None,
        InstalledMcpServersManifest(
            servers=[_installed_server(auth=McpServerAuth(type="bearer", token_env=TOKEN_ENV))]
        ),
    )

    core_tools_mod = _reload_core_tools()
    with caplog.at_level("WARNING"):
        core_tools_mod.initialize_tools()

    assert TOOL_ID not in core_tools_mod.ALL_TOOLS
    assert any(TOKEN_ENV in record.getMessage() for record in caplog.records)


def test_initialize_tools_warns_when_enabled_generic_tool_missing_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool enabled in the profile but absent from the cached manifest is skipped with a refresh hint."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    write_mcp_servers(
        None,
        InstalledMcpServersManifest(servers=[InstalledMcpServer(alias=SERVER_ALIAS, url=SERVER_URL)]),
    )

    core_tools_mod = _reload_core_tools()
    with caplog.at_level("WARNING"):
        core_tools_mod.initialize_tools()

    assert TOOL_ID not in core_tools_mod.ALL_TOOLS
    assert any("mcp-servers add" in record.message for record in caplog.records)


def test_initialize_tools_skips_generic_servers_on_corrupt_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A corrupt mcp_servers.json must not take down the whole tool registry at boot."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    manifest_path = tmp_path / "external_content" / "mcp_servers.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{not valid json", encoding="utf-8")

    core_tools_mod = _reload_core_tools()
    with caplog.at_level("ERROR"):
        core_tools_mod.initialize_tools()

    assert TOOL_ID not in core_tools_mod.ALL_TOOLS
    assert any("mcp_servers.json" in record.getMessage() for record in caplog.records)


def test_initialize_tools_skips_spaces_on_corrupt_manifest_but_keeps_generic_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A corrupt installed_tool_spaces.json must not take down the registry: generic MCP tools still load."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    spaces_path = tmp_path / "external_content" / "installed_tool_spaces.json"
    spaces_path.parent.mkdir(parents=True, exist_ok=True)
    spaces_path.write_text("{not valid json", encoding="utf-8")
    write_mcp_servers(None, InstalledMcpServersManifest(servers=[_installed_server()]))

    core_tools_mod = _reload_core_tools()
    with caplog.at_level("ERROR"):
        core_tools_mod.initialize_tools()

    assert TOOL_ID in core_tools_mod.ALL_TOOLS
    assert any("installed_tool_spaces.json" in record.getMessage() for record in caplog.records)


def test_initialize_tools_fails_on_name_collision_between_space_and_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Space tool and a generic server tool claiming the same local name must hard-fail.

    The add-time guards prevent this alias overlap, but manifests edited by hand can
    still collide; the registry is the backstop.
    """
    monkeypatch.chdir(tmp_path)
    # Both sources claim the alias "owner_example" (the Space's alias derives from its slug),
    # so the same enabled tool ID resolves through both.
    colliding_alias = "owner_example"
    colliding_tool_id = f"{colliding_alias}__do_thing"

    _profile_enabling(tmp_path, monkeypatch, "collision_profile", [colliding_tool_id])

    colliding_tool = CachedRemoteTool(
        local_name=colliding_tool_id,
        client_tool_name=colliding_tool_id,
        remote_name="do_thing",
        description="Colliding tool",
        parameters_schema={},
    )
    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(
            spaces=[
                InstalledToolSpace(
                    slug="owner/example",
                    alias=colliding_alias,
                    mcp_url="https://owner-example.hf.space/gradio_api/mcp/",
                    private=False,
                    tools=[colliding_tool],
                )
            ]
        ),
    )
    write_mcp_servers(
        None,
        InstalledMcpServersManifest(
            servers=[InstalledMcpServer(alias=colliding_alias, url=SERVER_URL, tools=[colliding_tool])]
        ),
    )

    core_tools_mod = _reload_core_tools()
    with pytest.raises(RuntimeError, match="Duplicate Tool.name"):
        core_tools_mod.initialize_tools()


def test_initialize_tools_survives_a_spaces_manifest_with_an_invalid_slug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The Space manifest validators raise ValueError, not RuntimeError, so the registry must tolerate both."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    spaces_path = tmp_path / "external_content" / "installed_tool_spaces.json"
    spaces_path.parent.mkdir(parents=True, exist_ok=True)
    spaces_path.write_text(
        json.dumps(
            {
                "version": 2,
                "spaces": [
                    {
                        "slug": "not a valid slug!",
                        "alias": "x",
                        "mcp_url": "https://x-y.hf.space/gradio_api/mcp/",
                        "private": False,
                        "tools": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_mcp_servers(None, InstalledMcpServersManifest(servers=[_installed_server()]))

    core_tools_mod = _reload_core_tools()
    with caplog.at_level("ERROR"):
        core_tools_mod.initialize_tools()

    # The bad Space is skipped, and the unrelated MCP server still registers.
    assert any("Skipping installed tool Spaces" in record.getMessage() for record in caplog.records)
    assert TOOL_ID in core_tools_mod.ALL_TOOLS
