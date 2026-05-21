from __future__ import annotations
import sys
import json
import importlib
from types import ModuleType
from pathlib import Path
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.tool_spaces as tool_spaces_mod
from reachy_mini_conversation_app.tool_spaces import (
    InstalledToolSpace,
    InstalledToolSpaceTool,
    ResolvedInstalledToolSpace,
    InstalledToolSpacesManifest,
    read_installed_tool_spaces,
    write_installed_tool_spaces,
)


def _reload_core_tools() -> ModuleType:
    for module_name in list(sys.modules):
        if module_name.startswith("reachy_mini_conversation_app.tools."):
            sys.modules.pop(module_name, None)

    sys.modules.pop("reachy_mini_conversation_app.tools.core_tools", None)
    return importlib.import_module("reachy_mini_conversation_app.tools.core_tools")


def _resolved_remote_space(client: AsyncMock) -> ResolvedInstalledToolSpace:
    return ResolvedInstalledToolSpace(
        slug="alozowski/reachy-mini-search-tool",
        alias="alozowski_reachy_mini_search_tool",
        mcp_url="https://alozowski-reachy-mini-search-tool.hf.space/gradio_api/mcp/",
        tags=["mcp", "reachy-mini-tool"],
        tools=[
            InstalledToolSpaceTool(
                local_name="alozowski_reachy_mini_search_tool__search_web",
                client_tool_name="alozowski_reachy_mini_search_tool__reachy_mini_search_tool_search_web",
                remote_name="reachy_mini_search_tool_search_web",
                description="Search the web",
                parameters_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
        client=client,
    )


@pytest.mark.asyncio
async def test_initialize_tools_loads_enabled_installed_remote_tools_and_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled public Space tools should join the registry and dispatch through the normal path."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "mcp_profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text("alozowski_reachy_mini_search_tool__search_web\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "mcp_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    client = AsyncMock()
    client.call_tool.return_value = {
        "status": "ok",
        "server_alias": "alozowski_reachy_mini_search_tool",
        "remote_tool_name": "reachy_mini_search_tool_search_web",
        "namespaced_tool_name": "alozowski_reachy_mini_search_tool__reachy_mini_search_tool_search_web",
        "content_blocks": [],
        "text": "hello",
    }
    monkeypatch.setattr(tool_spaces_mod, "resolve_public_tool_space_sync", lambda slug: _resolved_remote_space(client))

    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(
            spaces=[
                InstalledToolSpace(slug="alozowski/reachy-mini-search-tool", alias="alozowski_reachy_mini_search_tool")
            ]
        ),
    )

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    assert "alozowski_reachy_mini_search_tool__search_web" in core_tools_mod.ALL_TOOLS
    tool_specs = core_tools_mod.get_tool_specs()
    assert any(spec["name"] == "alozowski_reachy_mini_search_tool__search_web" for spec in tool_specs)

    result = await core_tools_mod.dispatch_tool_call(
        "alozowski_reachy_mini_search_tool__search_web",
        json.dumps({"query": "hello"}),
        core_tools_mod.ToolDependencies(
            reachy_mini=object(),
            movement_manager=object(),
        ),
    )

    assert result["namespaced_tool_name"] == "alozowski_reachy_mini_search_tool__search_web"
    assert result["tool_space_slug"] == "alozowski/reachy-mini-search-tool"
    client.call_tool.assert_awaited_once_with(
        "alozowski_reachy_mini_search_tool__reachy_mini_search_tool_search_web",
        {"query": "hello"},
    )


@pytest.mark.asyncio
async def test_tool_spaces_install_enable_and_dispatch_remote_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installing a Space, enabling its tool in a profile, and dispatching it should work end to end."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "mcp_profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text("alozowski_reachy_mini_search_tool__search_web\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "mcp_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    client = AsyncMock()
    client.call_tool.return_value = {
        "status": "ok",
        "server_alias": "alozowski_reachy_mini_search_tool",
        "remote_tool_name": "reachy_mini_search_tool_search_web",
        "namespaced_tool_name": "alozowski_reachy_mini_search_tool__reachy_mini_search_tool_search_web",
        "content_blocks": [],
        "text": "hello from installed space",
    }
    monkeypatch.setattr(tool_spaces_mod, "resolve_public_tool_space_sync", lambda slug: _resolved_remote_space(client))

    exit_code = tool_spaces_mod.handle_tool_spaces_command(
        Namespace(tool_spaces_command="add", space_slug="alozowski/reachy-mini-search-tool", install_only=True, profile=None)
    )
    assert exit_code == 0
    assert read_installed_tool_spaces(None).spaces == [
        InstalledToolSpace(
            slug="alozowski/reachy-mini-search-tool",
            alias="alozowski_reachy_mini_search_tool",
        )
    ]

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    result = await core_tools_mod.dispatch_tool_call(
        "alozowski_reachy_mini_search_tool__search_web",
        json.dumps({"query": "hello"}),
        core_tools_mod.ToolDependencies(
            reachy_mini=object(),
            movement_manager=object(),
        ),
    )

    assert result["status"] == "ok"
    assert result["text"] == "hello from installed space"
    assert result["namespaced_tool_name"] == "alozowski_reachy_mini_search_tool__search_web"
    assert result["tool_space_slug"] == "alozowski/reachy-mini-search-tool"
    client.call_tool.assert_awaited_once_with(
        "alozowski_reachy_mini_search_tool__reachy_mini_search_tool_search_web",
        {"query": "hello"},
    )


def test_initialize_tools_fails_when_enabled_remote_tool_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup should fail when the active profile enables a missing installed remote tool."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "remote_profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text("alozowski_reachy_mini_search_tool__search_web\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "remote_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    monkeypatch.setattr(
        tool_spaces_mod, "resolve_public_tool_space_sync", lambda slug: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(
            spaces=[
                InstalledToolSpace(slug="alozowski/reachy-mini-search-tool", alias="alozowski_reachy_mini_search_tool")
            ]
        ),
    )

    core_tools_mod = _reload_core_tools()
    with pytest.raises(
        RuntimeError, match="Enabled remote tools from 'alozowski/reachy-mini-search-tool' are unavailable"
    ):
        core_tools_mod.initialize_tools()


def test_initialize_tools_skips_unused_installed_remote_tool_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unused installed Spaces should not be resolved during profile tool loading."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    resolver = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(tool_spaces_mod, "resolve_public_tool_space_sync", resolver)

    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(
            spaces=[
                InstalledToolSpace(slug="alozowski/reachy-mini-search-tool", alias="alozowski_reachy_mini_search_tool")
            ]
        ),
    )

    core_tools_mod = _reload_core_tools()
    with caplog.at_level("WARNING"):
        core_tools_mod.initialize_tools()

    resolver.assert_not_called()
    assert not any("unavailable" in record.message for record in caplog.records)
    assert "dance" in core_tools_mod.ALL_TOOLS


def test_initialize_tools_inherits_default_tools_txt_for_profile_without_local_tool_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiles without a local tools.txt should inherit the built-in default tool set."""
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "inherit_default"
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "inherit_default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    assert "dance" in core_tools_mod.ALL_TOOLS
