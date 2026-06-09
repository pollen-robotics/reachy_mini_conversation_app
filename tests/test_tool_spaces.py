from __future__ import annotations
import sys
import json
from types import SimpleNamespace
from pathlib import Path
from argparse import Namespace

import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.main import main
from reachy_mini_conversation_app.mcp_client import RemoteToolSpec
from reachy_mini_conversation_app.tool_spaces import (
    handle_tool_spaces_command,
    read_installed_tool_spaces,
)


SEARCH_SPACE_SLUG = "pollen-robotics/reachy-mini-search-tool"
COLLIDING_SEARCH_SPACE_SLUG = "pollen_robotics/reachy-mini-search-tool"
PRIVATE_SPACE_SLUG = "pollen-robotics/private-space"
SEARCH_ALIAS = "pollen_robotics_reachy_mini_search_tool"
SEARCH_TOOL_ID = f"{SEARCH_ALIAS}__search_web"
SEARCH_CLIENT_TOOL_ID = f"{SEARCH_ALIAS}__reachy_mini_search_tool_search_web"


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
            server_alias=SEARCH_ALIAS,
            remote_name="reachy_mini_search_tool_search_web",
            namespaced_name=SEARCH_CLIENT_TOOL_ID,
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

    assert (
        _run_cli(
            monkeypatch,
            [
                "reachy-mini-conversation-app",
                "tool-spaces",
                "add",
                SEARCH_SPACE_SLUG,
                "--install-only",
            ],
        )
        == 0
    )

    manifest_path = tmp_path / "external_content" / "installed_tool_spaces.json"
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "spaces": [{"alias": SEARCH_ALIAS, "slug": SEARCH_SPACE_SLUG}],
    }

    assert _run_cli(monkeypatch, ["reachy-mini-conversation-app", "tool-spaces", "list"]) == 0

    assert _run_cli(monkeypatch, ["reachy-mini-conversation-app", "tool-spaces", "remove", SEARCH_SPACE_SLUG]) == 0
    assert read_installed_tool_spaces(None).spaces == []


def test_tool_spaces_add_rejects_non_public_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    assert _run_cli(monkeypatch, ["reachy-mini-conversation-app", "tool-spaces", "add", PRIVATE_SPACE_SLUG]) == 1
    assert not (tmp_path / "external_content" / "installed_tool_spaces.json").exists()


def test_tool_spaces_manifest_uses_instance_path_when_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        space_slug=SEARCH_SPACE_SLUG,
        install_only=True,
        profile=None,
    )
    assert handle_tool_spaces_command(args, instance_path=tmp_path) == 0
    assert (tmp_path / "installed_tool_spaces.json").is_file()
    assert not (tmp_path / "external_content" / "installed_tool_spaces.json").exists()


def test_read_installed_tool_spaces_raises_on_alias_collision_in_manifest(tmp_path: Path) -> None:
    """A manifest with two slugs that normalize to the same alias must be rejected on read."""
    payload = {
        "version": 1,
        "spaces": [
            {"slug": "owner/my-tool", "alias": "owner_my_tool"},
            {"slug": "owner/my_tool", "alias": "owner_my_tool"},
        ],
    }
    (tmp_path / "installed_tool_spaces.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="alias collision"):
        read_installed_tool_spaces(tmp_path)


def test_tool_spaces_add_rejects_alias_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second Space whose slug normalizes to the same alias must be rejected."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.HfApi.space_info",
        lambda self, slug, **kwargs: _mock_public_space_info(slug),
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.RemoteMcpToolClient.list_tool_specs",
        _mock_list_tool_specs,
    )

    assert (
        _run_cli(
            monkeypatch,
            ["app", "tool-spaces", "add", SEARCH_SPACE_SLUG, "--install-only"],
        )
        == 0
    )

    # The owner separator style differs, but both slugs normalize to the same alias.
    assert (
        _run_cli(
            monkeypatch,
            ["app", "tool-spaces", "add", COLLIDING_SEARCH_SPACE_SLUG, "--install-only"],
        )
        == 1
    )


def _setup_profile(tmp_path: Path, profile: str, existing_tools: list[str] | None = None) -> Path:
    """Create a profile directory with an optional tools.txt."""
    profile_dir = tmp_path / profile
    profile_dir.mkdir(parents=True)
    tools_txt = profile_dir / "tools.txt"
    tools_txt.write_text("\n".join(existing_tools or []) + "\n" if existing_tools else "", encoding="utf-8")
    return tools_txt


def _mock_add(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.HfApi.space_info",
        lambda self, slug, **kwargs: _mock_public_space_info(slug),
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.RemoteMcpToolClient.list_tool_specs",
        _mock_list_tool_specs,
    )


def test_tool_spaces_add_enables_in_active_profile_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add without flags should enable tools in the active profile."""
    _mock_add(monkeypatch, tmp_path)
    tools_txt = _setup_profile(tmp_path, "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", SEARCH_SPACE_SLUG]) == 0

    assert SEARCH_TOOL_ID in tools_txt.read_text(encoding="utf-8")


def test_tool_spaces_add_install_only_skips_tools_txt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--install-only should not modify any profile's tools.txt."""
    _mock_add(monkeypatch, tmp_path)
    tools_txt = _setup_profile(tmp_path, "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", SEARCH_SPACE_SLUG, "--install-only"]) == 0

    assert tools_txt.read_text(encoding="utf-8") == ""


def test_tool_spaces_add_profile_flag_enables_in_specified_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--profile should enable tools in the named profile, not the active one."""
    _mock_add(monkeypatch, tmp_path)
    default_tools_txt = _setup_profile(tmp_path, "default")
    canary_tools_txt = _setup_profile(tmp_path, "canary")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", SEARCH_SPACE_SLUG, "--profile", "canary"]) == 0

    assert SEARCH_TOOL_ID in canary_tools_txt.read_text(encoding="utf-8")
    assert default_tools_txt.read_text(encoding="utf-8") == ""
