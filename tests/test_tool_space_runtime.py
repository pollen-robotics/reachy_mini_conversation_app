import sys
import json
import asyncio
import importlib
import threading
from types import ModuleType
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.tool_spaces as tool_spaces_mod
from reachy_mini_conversation_app.mcp_client import McpToolTimeoutError, McpToolInvocationError
from reachy_mini_conversation_app.tool_spaces import (
    InstalledToolSpace,
    InstalledToolSpaceTool,
    InstalledToolSpacesManifest,
    write_installed_tool_spaces,
)
from reachy_mini_conversation_app.profile_store import write_profile


SEARCH_SPACE_SLUG = "example/search-tool"
SEARCH_ALIAS = "example_search_tool"
SEARCH_TOOL_ID = f"{SEARCH_ALIAS}__search_web"
SEARCH_CLIENT_TOOL_ID = f"{SEARCH_ALIAS}__search_tool_search_web"
SEARCH_MCP_URL = "https://example-search-tool.hf.space/gradio_api/mcp/"


def _reload_core_tools() -> ModuleType:
    for module_name in list(sys.modules):
        if module_name.startswith("reachy_mini_conversation_app.tools."):
            sys.modules.pop(module_name, None)

    sys.modules.pop("reachy_mini_conversation_app.tools.core_tools", None)
    return importlib.import_module("reachy_mini_conversation_app.tools.core_tools")


def _installed_search_space() -> InstalledToolSpace:
    return InstalledToolSpace(
        slug=SEARCH_SPACE_SLUG,
        alias=SEARCH_ALIAS,
        mcp_url=SEARCH_MCP_URL,
        private=False,
        tools=[
            InstalledToolSpaceTool(
                local_name=SEARCH_TOOL_ID,
                client_tool_name=SEARCH_CLIENT_TOOL_ID,
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


@pytest.mark.asyncio
async def test_initialize_tools_loads_enabled_installed_remote_tools_and_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled public Space tools should join the registry and dispatch through the normal path."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "mcp_profile"
    write_profile("mcp_profile", profile_dir, "hello", [SEARCH_TOOL_ID])

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "mcp_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    client = AsyncMock()
    client.call_tool.return_value = {
        "status": "ok",
        "server_alias": SEARCH_ALIAS,
        "remote_tool_name": "reachy_mini_search_tool_search_web",
        "namespaced_tool_name": SEARCH_CLIENT_TOOL_ID,
        "content_blocks": [],
        "text": "hello",
    }
    captured_cached_tools: list[InstalledToolSpaceTool] | None = None

    def _build_remote_client(
        alias: str,
        mcp_url: str,
        *,
        private: bool,
        cached_tools: list[InstalledToolSpaceTool],
    ) -> AsyncMock:
        nonlocal captured_cached_tools
        assert alias == SEARCH_ALIAS
        assert mcp_url == SEARCH_MCP_URL
        assert private is False
        captured_cached_tools = cached_tools
        return client

    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", _build_remote_client)

    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(spaces=[_installed_search_space()]),
    )

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    assert SEARCH_TOOL_ID in core_tools_mod.ALL_TOOLS
    assert captured_cached_tools == _installed_search_space().tools
    tool_specs = core_tools_mod.get_tool_specs()
    assert any(spec["name"] == SEARCH_TOOL_ID for spec in tool_specs)

    result = await core_tools_mod.dispatch_tool_call(
        SEARCH_TOOL_ID,
        json.dumps({"query": "hello"}),
        core_tools_mod.ToolDependencies(
            reachy_mini=object(),
            movement_manager=object(),
        ),
    )

    assert result["namespaced_tool_name"] == SEARCH_TOOL_ID
    assert result["tool_space_slug"] == SEARCH_SPACE_SLUG
    client.call_tool.assert_awaited_once_with(SEARCH_CLIENT_TOOL_ID, {"query": "hello"})


def test_initialize_tools_warns_when_enabled_tool_missing_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool enabled in the profile but absent from the cached manifest is skipped with a warning."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "remote_profile"
    write_profile("remote_profile", profile_dir, "hello", [SEARCH_TOOL_ID])

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "remote_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(
            spaces=[
                InstalledToolSpace(
                    slug=SEARCH_SPACE_SLUG,
                    alias=SEARCH_ALIAS,
                    mcp_url=SEARCH_MCP_URL,
                    private=False,
                ),
            ]
        ),
    )

    core_tools_mod = _reload_core_tools()
    with caplog.at_level("WARNING"):
        core_tools_mod.initialize_tools()

    assert any(SEARCH_SPACE_SLUG in record.message for record in caplog.records)
    assert SEARCH_TOOL_ID not in core_tools_mod.ALL_TOOLS


def test_initialize_tools_respects_empty_profile_tool_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict profile with no authored tools should not inherit another profile's tools."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "inherit_default"
    write_profile("inherit_default", profile_dir, "hello", [])

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "inherit_default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    assert "dance" not in core_tools_mod.ALL_TOOLS


def _mcp_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "mcp_profile"
    write_profile("mcp_profile", profile_dir, "hello", [SEARCH_TOOL_ID])
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "mcp_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)


@pytest.mark.asyncio
async def test_remote_tool_retries_once_after_transport_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient remote transport failures should get one fast retry."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    client = AsyncMock()
    client.call_tool.side_effect = [
        McpToolInvocationError("connection reset"),
        {
            "status": "ok",
            "server_alias": SEARCH_ALIAS,
            "remote_tool_name": "reachy_mini_search_tool_search_web",
            "namespaced_tool_name": SEARCH_CLIENT_TOOL_ID,
            "content_blocks": [],
            "text": "hello",
        },
    ]
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *a, **k: client)
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(spaces=[_installed_search_space()]))

    core_tools_mod = _reload_core_tools()
    monkeypatch.setattr(core_tools_mod, "_REMOTE_TOOL_RETRY_DELAY_S", 0.0)
    core_tools_mod.initialize_tools()

    result = await core_tools_mod.dispatch_tool_call(
        SEARCH_TOOL_ID,
        json.dumps({"query": "hello"}),
        core_tools_mod.ToolDependencies(reachy_mini=object(), movement_manager=object()),
    )

    assert result["status"] == "ok"
    assert client.call_tool.await_count == 2


@pytest.mark.asyncio
async def test_remote_tool_does_not_retry_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote timeouts should fail once instead of doubling the user wait."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    client = AsyncMock()
    client.call_tool.side_effect = McpToolTimeoutError("slow tool")
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *a, **k: client)
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(spaces=[_installed_search_space()]))

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    result = await core_tools_mod.dispatch_tool_call(
        SEARCH_TOOL_ID,
        json.dumps({"query": "hello"}),
        core_tools_mod.ToolDependencies(reachy_mini=object(), movement_manager=object()),
    )

    assert "error" in result
    assert client.call_tool.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_remote_tool_plays_thinking_move_until_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    """A thinking move runs during the remote call and is cancelled when it returns, even on failure."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    queued = asyncio.Event()

    async def call_tool(*args: object) -> dict[str, str]:
        await asyncio.wait_for(queued.wait(), timeout=1.0)
        if fails:
            raise McpToolTimeoutError("slow tool")
        return {"status": "ok", "text": "hello"}

    client = AsyncMock()
    client.call_tool.side_effect = call_tool
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *a, **k: client)
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(spaces=[_installed_search_space()]))

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()
    move = object()
    monkeypatch.setattr(core_tools_mod, "thinking_move", lambda: move)
    movement_manager = MagicMock()
    movement_manager.queue_move.side_effect = lambda _move, **kwargs: queued.set()

    result = await core_tools_mod.dispatch_tool_call(
        SEARCH_TOOL_ID,
        json.dumps({"query": "hello"}),
        core_tools_mod.ToolDependencies(reachy_mini=object(), movement_manager=movement_manager),
    )

    assert ("error" in result) is fails
    movement_manager.queue_move.assert_called_once_with(move, pause_head_tracking=True)
    movement_manager.cancel_move.assert_called_once_with(move)


@pytest.mark.asyncio
async def test_remote_tool_does_not_wait_for_thinking_move_or_queue_it_after_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncached feedback library cannot delay a lookup or animate after its answer."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)
    loading = asyncio.Event()
    finished_loading = asyncio.Event()
    release_loader = threading.Event()
    loop = asyncio.get_running_loop()

    def load_thinking_move() -> object:
        loop.call_soon_threadsafe(loading.set)
        try:
            if not release_loader.wait(timeout=2.0):
                raise TimeoutError("Thinking loader was not released")
            return object()
        finally:
            loop.call_soon_threadsafe(finished_loading.set)

    async def call_tool(*args: object) -> dict[str, str]:
        await asyncio.wait_for(loading.wait(), timeout=1.0)
        return {"status": "ok", "text": "hello"}

    client = AsyncMock()
    client.call_tool.side_effect = call_tool
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *a, **k: client)
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(spaces=[_installed_search_space()]))
    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()
    monkeypatch.setattr(core_tools_mod, "thinking_move", load_thinking_move)
    movement_manager = MagicMock()

    try:
        result = await asyncio.wait_for(
            core_tools_mod.dispatch_tool_call(
                SEARCH_TOOL_ID,
                json.dumps({"query": "hello"}),
                core_tools_mod.ToolDependencies(reachy_mini=object(), movement_manager=movement_manager),
            ),
            timeout=1.0,
        )
        assert result["text"] == "hello"
        assert not release_loader.is_set()
        client.call_tool.assert_awaited_once()
    finally:
        release_loader.set()
        await asyncio.wait_for(finished_loading.wait(), timeout=1.0)

    movement_manager.queue_move.assert_not_called()
    movement_manager.cancel_move.assert_not_called()


@pytest.mark.asyncio
async def test_remote_tool_cancellation_stops_thinking_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling a pending lookup removes its feedback without waiting for an answer."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)
    client = AsyncMock()
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *a, **k: client)
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(spaces=[_installed_search_space()]))
    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()
    move = object()
    monkeypatch.setattr(core_tools_mod, "thinking_move", lambda: move)
    queued = asyncio.Event()
    movement_manager = MagicMock()
    movement_manager.queue_move.side_effect = lambda _move, **kwargs: queued.set()

    async def call_tool(*args: object) -> None:
        await asyncio.Future[None]()

    client.call_tool.side_effect = call_tool
    lookup = asyncio.create_task(
        core_tools_mod.dispatch_tool_call(
            SEARCH_TOOL_ID,
            json.dumps({"query": "hello"}),
            core_tools_mod.ToolDependencies(reachy_mini=object(), movement_manager=movement_manager),
        )
    )
    try:
        await asyncio.wait_for(queued.wait(), timeout=1.0)
    finally:
        lookup.cancel()
        result = await asyncio.wait_for(lookup, timeout=1.0)

    assert result == {"error": "Tool cancelled"}
    movement_manager.cancel_move.assert_called_once_with(move)


def test_initialize_tools_survives_a_corrupt_installed_spaces_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A damaged manifest should cost the Space tools, not the whole tool registry.

    `read_installed_tool_spaces` raises RuntimeError for a malformed payload and
    ValueError straight out of `validate_space_slug` / `validate_space_mcp_url`
    for a bad entry (see the reader's own tests). Either one used to propagate
    out of tool initialization, so a single bad character in the manifest left
    the app with no tools at all instead of just no Space tools.
    """
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "remote_profile"
    write_profile("remote_profile", profile_dir, "hello", [SEARCH_TOOL_ID])

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "remote_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    # A well-formed envelope whose entry trips the URL validator (ValueError),
    # which is the path that no exception handler covered.
    manifest_path = tmp_path / "external_content" / "installed_tool_spaces.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "spaces": [
                    {
                        "slug": SEARCH_SPACE_SLUG,
                        "alias": SEARCH_ALIAS,
                        "mcp_url": "https://attacker.example/gradio_api/mcp/",
                        "private": True,
                        "tools": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    core_tools_mod = _reload_core_tools()
    with caplog.at_level("ERROR"):
        core_tools_mod.initialize_tools()

    assert any("Skipping installed tool Spaces" in record.message for record in caplog.records)
    assert SEARCH_TOOL_ID not in core_tools_mod.ALL_TOOLS
    # The registry still came up: built-in tools are unaffected by a bad manifest.
    assert core_tools_mod.ALL_TOOLS
