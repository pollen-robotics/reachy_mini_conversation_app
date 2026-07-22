import sys
import json
import threading
from types import SimpleNamespace
from pathlib import Path
from argparse import Namespace
from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.main import main
from reachy_mini_conversation_app.mcp_client import RemoteToolSpec, RemoteMcpToolClient
from reachy_mini_conversation_app.tool_spaces import (
    ToolSpaceProfileUpdateError,
    remove_tool_space,
    install_tool_space,
    resolve_tool_space_sync,
    handle_tool_spaces_command,
    read_installed_tool_spaces,
)
from reachy_mini_conversation_app.profile_store import write_profile
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_tool_names,
    read_profile_tool_override,
    write_profile_tool_override,
)


SEARCH_SPACE_SLUG = "example/search-tool"
COLLIDING_SEARCH_SPACE_SLUG = "example/search_tool"
PRIVATE_SPACE_SLUG = "example/private-space"
SEARCH_ALIAS = "example_search_tool"
SEARCH_REMOTE_NAME = "search_tool_search_web"
SEARCH_TOOL_ID = f"{SEARCH_ALIAS}__search_web"
SEARCH_CLIENT_TOOL_ID = f"{SEARCH_ALIAS}__{SEARCH_REMOTE_NAME}"


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


def _mock_private_space_info(slug: str) -> SimpleNamespace:
    info = _mock_public_space_info(slug)
    info.private = True
    return info


async def _mock_list_tool_specs(self: RemoteMcpToolClient) -> list[RemoteToolSpec]:
    alias = self.server.alias
    return [
        RemoteToolSpec(
            server_alias=alias,
            remote_name=SEARCH_REMOTE_NAME,
            namespaced_name=f"{alias}__{SEARCH_REMOTE_NAME}",
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
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["version"] == 2
    added_entry = next(space for space in written["spaces"] if space["slug"] == SEARCH_SPACE_SLUG)
    assert added_entry == {
        "slug": SEARCH_SPACE_SLUG,
        "alias": SEARCH_ALIAS,
        "mcp_url": "https://example-search-tool.hf.space/gradio_api/mcp/",
        "private": False,
        "tools": [
            {
                "local_name": SEARCH_TOOL_ID,
                "client_tool_name": SEARCH_CLIENT_TOOL_ID,
                "remote_name": SEARCH_REMOTE_NAME,
                "description": "Search the web",
                "parameters_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }

    resolve_tool_space = MagicMock(side_effect=AssertionError("list must use cached metadata"))
    monkeypatch.setattr("reachy_mini_conversation_app.tool_spaces.resolve_tool_space_sync", resolve_tool_space)
    assert _run_cli(monkeypatch, ["reachy-mini-conversation-app", "tool-spaces", "list"]) == 0
    resolve_tool_space.assert_not_called()

    assert _run_cli(monkeypatch, ["reachy-mini-conversation-app", "tool-spaces", "remove", SEARCH_SPACE_SLUG]) == 0
    assert SEARCH_SPACE_SLUG not in [space.slug for space in read_installed_tool_spaces(None).spaces]


def test_tool_spaces_add_installs_private_space_with_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A private Space resolves and installs when an HF token is available."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.HfApi.space_info",
        lambda self, slug, **kwargs: _mock_private_space_info(slug),
    )
    monkeypatch.setattr("reachy_mini_conversation_app.tool_spaces.get_token", lambda: "hf_test_token")
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.RemoteMcpToolClient.list_tool_specs",
        _mock_list_tool_specs,
    )

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", PRIVATE_SPACE_SLUG, "--install-only"]) == 0
    assert PRIVATE_SPACE_SLUG in [space.slug for space in read_installed_tool_spaces(None).spaces]


@pytest.mark.parametrize(
    ("private", "configured_token", "login_token", "expected_authorization"),
    [
        (True, None, "hf_test_token", "Bearer hf_test_token"),
        (True, "hf_env_token", None, "Bearer hf_env_token"),
        (False, None, "hf_test_token", None),
    ],
)
def test_resolve_tool_space_sends_auth_only_to_private_spaces(
    monkeypatch: pytest.MonkeyPatch,
    private: bool,
    configured_token: str | None,
    login_token: str | None,
    expected_authorization: str | None,
) -> None:
    """Private Spaces use the configured/login token while public Spaces receive no credentials."""
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.HfApi.space_info",
        lambda self, slug, **kwargs: _mock_private_space_info(slug) if private else _mock_public_space_info(slug),
    )
    monkeypatch.setattr(config_mod.config, "HF_TOKEN", configured_token)
    monkeypatch.setattr("reachy_mini_conversation_app.tool_spaces.get_token", lambda: login_token)
    authorization: str | None = None

    async def _capture_authorization(self: RemoteMcpToolClient) -> list[RemoteToolSpec]:
        nonlocal authorization
        authorization = self.server.headers.get("Authorization")
        return await _mock_list_tool_specs(self)

    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.RemoteMcpToolClient.list_tool_specs",
        _capture_authorization,
    )

    resolve_tool_space_sync(PRIVATE_SPACE_SLUG if private else SEARCH_SPACE_SLUG)

    assert authorization == expected_authorization


def test_tool_spaces_add_private_space_without_token_hints_at_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without a token, adding a private Space fails cleanly and points at HF auth."""
    monkeypatch.chdir(tmp_path)

    def _raise_not_found(self: object, slug: str, **kwargs: object) -> SimpleNamespace:
        raise RepositoryNotFoundError(
            "404 Client Error", response=httpx.Response(404, request=httpx.Request("GET", "https://hf.co"))
        )

    monkeypatch.setattr("reachy_mini_conversation_app.tool_spaces.HfApi.space_info", _raise_not_found)
    monkeypatch.setattr(config_mod.config, "HF_TOKEN", None)
    monkeypatch.setattr("reachy_mini_conversation_app.tool_spaces.get_token", lambda: None)

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", PRIVATE_SPACE_SLUG]) == 1
    assert "hf auth login" in capsys.readouterr().err
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
    mcp_url = "https://example.hf.space/gradio_api/mcp/"
    payload = {
        "version": 2,
        "spaces": [
            {"slug": "owner/my-tool", "alias": "owner_my_tool", "mcp_url": mcp_url, "private": False, "tools": []},
            {"slug": "owner/my_tool", "alias": "owner_my_tool", "mcp_url": mcp_url, "private": False, "tools": []},
        ],
    }
    (tmp_path / "installed_tool_spaces.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="alias collision"):
        read_installed_tool_spaces(tmp_path)


def test_read_installed_tool_spaces_rejects_legacy_manifest(tmp_path: Path) -> None:
    """Only the current strict manifest schema should be accepted for the 1.0 data model."""
    (tmp_path / "installed_tool_spaces.json").write_text(
        json.dumps({"version": 1, "spaces": []}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected 2"):
        read_installed_tool_spaces(tmp_path)


def test_read_installed_tool_spaces_rejects_non_hugging_face_endpoint(tmp_path: Path) -> None:
    """A persisted private Space must not be able to redirect the HF token to another host."""
    payload = {
        "version": 2,
        "spaces": [
            {
                "slug": PRIVATE_SPACE_SLUG,
                "alias": "example_private_space",
                "mcp_url": "https://attacker.example/gradio_api/mcp/",
                "private": True,
                "tools": [],
            }
        ],
    }
    (tmp_path / "installed_tool_spaces.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid Hugging Face Space MCP URL"):
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


def _setup_profile(tmp_path: Path, profile: str, default_tools: list[str] | None = None) -> Path:
    """Create a strict profile document with authored tool defaults."""
    profile_dir = tmp_path / profile
    return write_profile(profile, profile_dir, "Test profile instructions.", default_tools or [])


def _mock_add(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("reachy_mini_conversation_app.profile_store.DEFAULT_PROFILES_DIRECTORY", tmp_path)
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
    profile_path = _setup_profile(tmp_path, "default", ["dance"])
    original_profile = profile_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", SEARCH_SPACE_SLUG]) == 0

    assert read_profile_tool_override("default", None) == ["dance", SEARCH_TOOL_ID]
    assert read_profile_tool_names("default", None) == ["dance", SEARCH_TOOL_ID]
    assert profile_path.read_text(encoding="utf-8") == original_profile


def test_tool_spaces_remove_disables_tools_in_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a Space strips its tool IDs from the profile they were enabled in."""
    _mock_add(monkeypatch, tmp_path)
    profile_path = _setup_profile(tmp_path, "default", ["dance"])
    original_profile = profile_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", SEARCH_SPACE_SLUG]) == 0
    assert read_profile_tool_names("default", None) == ["dance", SEARCH_TOOL_ID]

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "remove", SEARCH_SPACE_SLUG]) == 0
    assert read_profile_tool_override("default", None) == ["dance"]
    assert read_profile_tool_names("default", None) == ["dance"]
    assert profile_path.read_text(encoding="utf-8") == original_profile
    assert SEARCH_SPACE_SLUG not in [space.slug for space in read_installed_tool_spaces(None).spaces]


def test_tool_spaces_add_install_only_skips_profile_toolset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--install-only should not create a profile tool override."""
    _mock_add(monkeypatch, tmp_path)
    profile_path = _setup_profile(tmp_path, "default", ["dance"])
    original_profile = profile_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", SEARCH_SPACE_SLUG, "--install-only"]) == 0

    assert read_profile_tool_override("default", None) is None
    assert read_profile_tool_names("default", None) == ["dance"]
    assert profile_path.read_text(encoding="utf-8") == original_profile


def test_tool_spaces_add_profile_flag_enables_in_specified_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--profile should enable tools in the named profile, not the active one."""
    _mock_add(monkeypatch, tmp_path)
    default_profile_path = _setup_profile(tmp_path, "default", ["dance"])
    canary_profile_path = _setup_profile(tmp_path, "canary", ["move_head"])
    original_default_profile = default_profile_path.read_text(encoding="utf-8")
    original_canary_profile = canary_profile_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")

    assert _run_cli(monkeypatch, ["app", "tool-spaces", "add", SEARCH_SPACE_SLUG, "--profile", "canary"]) == 0

    assert read_profile_tool_override("canary", None) == ["move_head", SEARCH_TOOL_ID]
    assert read_profile_tool_names("canary", None) == ["move_head", SEARCH_TOOL_ID]
    assert read_profile_tool_override("default", None) is None
    assert read_profile_tool_names("default", None) == ["dance"]
    assert default_profile_path.read_text(encoding="utf-8") == original_default_profile
    assert canary_profile_path.read_text(encoding="utf-8") == original_canary_profile


def test_tool_space_install_profile_failure_leaves_manifest_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed profile update must not install a Space globally."""
    _mock_add(monkeypatch, tmp_path)
    _setup_profile(tmp_path, "guide", ["dance"])
    instance_path = tmp_path / "instance"
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.enable_profile_tools",
        MagicMock(side_effect=OSError("profile store unavailable")),
    )

    with pytest.raises(ToolSpaceProfileUpdateError, match="profile store unavailable"):
        install_tool_space(SEARCH_SPACE_SLUG, instance_path, profile="guide")

    assert read_profile_tool_override("guide", instance_path) is None
    assert SEARCH_SPACE_SLUG not in [space.slug for space in read_installed_tool_spaces(instance_path).spaces]


def test_tool_space_install_manifest_failure_rolls_back_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed manifest commit must restore the profile tool selection."""
    _mock_add(monkeypatch, tmp_path)
    _setup_profile(tmp_path, "guide", ["dance"])
    instance_path = tmp_path / "instance"
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.write_installed_tool_spaces",
        MagicMock(side_effect=OSError("manifest store unavailable")),
    )

    with pytest.raises(RuntimeError, match="manifest store unavailable"):
        install_tool_space(SEARCH_SPACE_SLUG, instance_path, profile="guide")

    assert read_profile_tool_override("guide", instance_path) is None
    assert not (instance_path / "installed_tool_spaces.json").exists()


def test_tool_space_install_rollback_does_not_overwrite_concurrent_profile_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile save that waits on a failed install should be applied after rollback."""
    _mock_add(monkeypatch, tmp_path)
    _setup_profile(tmp_path, "guide", ["dance"])
    instance_path = tmp_path / "instance"
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    manifest_write_started = threading.Event()
    release_manifest_write = threading.Event()
    profile_save_started = threading.Event()

    def fail_manifest_write(_instance_path: object, _manifest: object) -> None:
        manifest_write_started.set()
        if not release_manifest_write.wait(timeout=1.0):
            raise TimeoutError("Timed out waiting to fail manifest write")
        raise OSError("manifest store unavailable")

    def save_profile_tools() -> None:
        profile_save_started.set()
        write_profile_tool_override("guide", ["camera"], instance_path)

    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.write_installed_tool_spaces",
        fail_manifest_write,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        install_future = executor.submit(install_tool_space, SEARCH_SPACE_SLUG, instance_path, profile="guide")
        assert manifest_write_started.wait(timeout=1.0)
        profile_save_future = executor.submit(save_profile_tools)
        try:
            assert profile_save_started.wait(timeout=1.0)
            with pytest.raises(TimeoutError):
                profile_save_future.result(timeout=0.1)
        finally:
            release_manifest_write.set()
        with pytest.raises(RuntimeError, match="manifest store unavailable"):
            install_future.result(timeout=1.0)
        profile_save_future.result(timeout=1.0)

    assert read_profile_tool_override("guide", instance_path) == ["camera"]


def test_tool_space_remove_manifest_failure_rolls_back_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed removal commit must restore every affected profile selection."""
    _mock_add(monkeypatch, tmp_path)
    _setup_profile(tmp_path, "default", ["dance"])
    _setup_profile(tmp_path, "guide", ["dance"])
    instance_path = tmp_path / "instance"
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    install_tool_space(SEARCH_SPACE_SLUG, instance_path, profile="guide")
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.write_installed_tool_spaces",
        MagicMock(side_effect=OSError("manifest store unavailable")),
    )

    with pytest.raises(RuntimeError, match="manifest store unavailable"):
        remove_tool_space(SEARCH_SPACE_SLUG, instance_path)

    assert read_profile_tool_override("guide", instance_path) == ["dance", SEARCH_TOOL_ID]
    assert SEARCH_SPACE_SLUG in [space.slug for space in read_installed_tool_spaces(instance_path).spaces]


def test_tool_space_remove_cleans_default_and_orphaned_profile_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removal should clean persisted access even when profiles are outside or absent from the active root."""
    _mock_add(monkeypatch, tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    _setup_profile(external_profiles_root, "guide", ["camera"])
    instance_path = tmp_path / "instance"
    monkeypatch.setattr(config_mod.config, "INSTANCE_PATH", instance_path)
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    install_tool_space(SEARCH_SPACE_SLUG, instance_path, install_only=True)
    write_profile_tool_override("default", ["dance", SEARCH_TOOL_ID], instance_path)
    write_profile_tool_override("deleted_profile", [SEARCH_TOOL_ID], instance_path)

    remove_tool_space(SEARCH_SPACE_SLUG, instance_path)

    assert read_profile_tool_override("default", instance_path) == ["dance"]
    assert read_profile_tool_override("deleted_profile", instance_path) == []


def test_read_installed_tool_spaces_seeds_bundled_pollen_spaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no manifest, the three bundled Pollen Spaces are seeded with their tools cached offline."""
    monkeypatch.chdir(tmp_path)

    spaces = read_installed_tool_spaces(None).spaces
    assert [space.slug for space in spaces] == [
        "pollen-robotics/reachy-mini-search-tool",
        "pollen-robotics/reachy-mini-time-tool",
        "pollen-robotics/reachy-mini-weather-tool",
    ]
    search_tool = spaces[0].tools[0]
    assert search_tool.local_name == "pollen_robotics_reachy_mini_search_tool__search_web"
    assert search_tool.remote_name == "reachy_mini_search_tool_search_web"
    assert spaces[0].private is False


def test_install_tool_space_refreshes_cached_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-adding an installed Space should replace its cached tool metadata."""
    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.HfApi.space_info",
        lambda self, slug, **kwargs: _mock_public_space_info(slug),
    )
    descriptions = ["First description", "Refreshed description"]

    async def _list_tool_specs(self: object) -> list[RemoteToolSpec]:
        return [
            RemoteToolSpec(
                server_alias=SEARCH_ALIAS,
                remote_name=SEARCH_REMOTE_NAME,
                namespaced_name=SEARCH_CLIENT_TOOL_ID,
                description=descriptions.pop(0),
                parameters_schema={"type": "object", "properties": {}},
            )
        ]

    monkeypatch.setattr(
        "reachy_mini_conversation_app.tool_spaces.RemoteMcpToolClient.list_tool_specs",
        _list_tool_specs,
    )

    first_result = install_tool_space(SEARCH_SPACE_SLUG, tmp_path, install_only=True)
    refreshed_result = install_tool_space(SEARCH_SPACE_SLUG, tmp_path, install_only=True)

    assert first_result.refreshed is False
    assert refreshed_result.refreshed is True
    matching_spaces = [space for space in refreshed_result.manifest.spaces if space.slug == SEARCH_SPACE_SLUG]
    assert len(matching_spaces) == 1
    assert matching_spaces[0].tools[0].description == "Refreshed description"
