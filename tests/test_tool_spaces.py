from __future__ import annotations
import sys
import json
from types import SimpleNamespace
from pathlib import Path
from argparse import Namespace

import pytest

from reachy_mini_conversation_app.main import main
from reachy_mini_conversation_app.mcp_client import RemoteToolSpec
from reachy_mini_conversation_app.tool_spaces import (
    InstalledToolSpace,
    InstalledToolSpacesManifest,
    handle_tool_spaces_command,
    read_installed_tool_spaces,
    write_installed_tool_spaces,
)


def _mock_public_space_info(slug: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=slug,
        private=False,
        disabled=False,
        sdk="gradio",
        host=None,
        subdomain=slug.replace("/", "-"),
        tags=["reachy-mini-tool", "mcp"],
    )


async def _mock_list_tool_specs(self: object) -> list[RemoteToolSpec]:
    return [
        RemoteToolSpec(
            server_alias="alozowski_reachy_mini_search_tool",
            remote_name="reachy_mini_search_tool_search_web",
            namespaced_name="alozowski_reachy_mini_search_tool__reachy_mini_search_tool_search_web",
            description="Search the web",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    return int(exc.value.code)


def test_tool_spaces_add_list_remove_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should install, list, and remove a public Space tool source cleanly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.HfApi.space_info",
        lambda self, slug, **kwargs: _mock_public_space_info(slug),
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.RemoteMcpToolClient.list_tool_specs",
        _mock_list_tool_specs,
    )

    exit_code = _run_cli(
        monkeypatch,
        [
            "reachy-mini-conversation-app",
            "tool-spaces",
            "add",
            "alozowski/reachy-mini-search-tool",
        ],
    )
    assert exit_code == 0
    add_output = capsys.readouterr().out
    assert "Installed Space tool source: alozowski/reachy-mini-search-tool" in add_output
    assert "alozowski_reachy_mini_search_tool__search_web" in add_output

    manifest_path = tmp_path / "external_content" / "installed_tool_spaces.json"
    assert manifest_path.is_file()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload == {
        "version": 1,
        "spaces": [
            {
                "alias": "alozowski_reachy_mini_search_tool",
                "slug": "alozowski/reachy-mini-search-tool",
            }
        ],
    }

    exit_code = _run_cli(
        monkeypatch,
        [
            "reachy-mini-conversation-app",
            "tool-spaces",
            "list",
        ],
    )
    assert exit_code == 0
    list_output = capsys.readouterr().out
    assert "Manifest:" in list_output
    assert "alozowski/reachy-mini-search-tool (alozowski_reachy_mini_search_tool)" in list_output
    assert "alozowski_reachy_mini_search_tool__search_web" in list_output

    exit_code = _run_cli(
        monkeypatch,
        [
            "reachy-mini-conversation-app",
            "tool-spaces",
            "remove",
            "alozowski/reachy-mini-search-tool",
        ],
    )
    assert exit_code == 0
    remove_output = capsys.readouterr().out
    assert "Removed Space tool source: alozowski/reachy-mini-search-tool" in remove_output
    assert read_installed_tool_spaces(None).spaces == []


def test_tool_spaces_add_rejects_non_public_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should reject non-public Spaces before writing the manifest."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.HfApi.space_info",
        lambda self, slug, **kwargs: SimpleNamespace(
            id=slug,
            private=True,
            disabled=False,
            sdk="gradio",
            host=None,
            subdomain=slug.replace("/", "-"),
            tags=[],
        ),
    )

    exit_code = _run_cli(
        monkeypatch,
        [
            "reachy-mini-conversation-app",
            "tool-spaces",
            "add",
            "alozowski/private-space",
        ],
    )
    assert exit_code == 1
    output = capsys.readouterr()
    assert "is not public" in output.err


def test_tool_spaces_manifest_uses_instance_path_when_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Managed instance paths should store the manifest beside other instance-local state."""
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.HfApi.space_info",
        lambda self, slug, **kwargs: _mock_public_space_info(slug),
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.RemoteMcpToolClient.list_tool_specs",
        _mock_list_tool_specs,
    )

    args = Namespace(
        tool_spaces_command="add",
        space_slug="alozowski/reachy-mini-search-tool",
    )
    exit_code = handle_tool_spaces_command(args, instance_path=tmp_path)
    assert exit_code == 0
    assert (tmp_path / "installed_tool_spaces.json").is_file()
    assert not (tmp_path / "external_content" / "installed_tool_spaces.json").exists()
    output = capsys.readouterr().out
    assert f"Manifest: {tmp_path / 'installed_tool_spaces.json'}" in output


def test_write_and_read_installed_tool_spaces_round_trip_for_instance_path(tmp_path: Path) -> None:
    """Persisted manifests should round-trip through the instance-local path."""
    manifest = InstalledToolSpacesManifest(
        spaces=[InstalledToolSpace(slug="owner/space", alias="owner_space")],
    )

    manifest_path = write_installed_tool_spaces(tmp_path, manifest)

    assert manifest_path == tmp_path / "installed_tool_spaces.json"
    assert read_installed_tool_spaces(tmp_path) == manifest
