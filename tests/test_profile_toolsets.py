"""Tests for instance-local personality tool selections."""

import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

import reachy_mini_conversation_app.profile_store as profile_store_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.profile_store import write_profile
from reachy_mini_conversation_app.profile_toolsets import (
    enable_profile_tools,
    read_profile_toolsets,
    read_profile_tool_names,
    get_profile_toolsets_path,
    read_profile_tool_override,
    clear_profile_tool_override,
    write_profile_tool_override,
    disable_profile_tools_by_prefix,
)


SPACE_ALIAS = "example_search_tool"
TOOL_NAME = f"{SPACE_ALIAS}__search_web"


@pytest.fixture
def configured_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Configure two strict profile documents and return their instance path."""
    instance_path = tmp_path / "instance"
    profiles_root = tmp_path / "profiles"
    write_profile("default", profiles_root / "default", "Default profile.", ["dance", TOOL_NAME])
    write_profile(
        "guide",
        profiles_root / "guide",
        "Guide profile.",
        ["camera", TOOL_NAME, "other_space__lookup"],
    )
    monkeypatch.setattr(config, "INSTANCE_PATH", instance_path)
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr(profile_store_mod, "DEFAULT_PROFILES_DIRECTORY", profiles_root)
    return instance_path


def test_profile_tool_override_round_trip_and_reset(configured_profiles: Path) -> None:
    """An explicit override should replace authored defaults until it is reset."""
    instance_path = configured_profiles
    assert read_profile_tool_names("guide", instance_path) == ["camera", TOOL_NAME, "other_space__lookup"]
    assert read_profile_tool_override("guide", instance_path) is None

    settings_path = write_profile_tool_override(
        "guide",
        [" camera ", "# disabled", "", "camera", "other_space__lookup"],
        instance_path,
    )

    assert settings_path == get_profile_toolsets_path(instance_path)
    assert read_profile_tool_override("guide", instance_path) == ["camera", "other_space__lookup"]
    assert read_profile_tool_names("guide", instance_path) == ["camera", "other_space__lookup"]

    write_profile_tool_override("guide", [], instance_path)

    assert read_profile_tool_override("guide", instance_path) == []
    assert read_profile_tool_names("guide", instance_path) == []
    assert clear_profile_tool_override("guide", instance_path) is True
    assert read_profile_tool_override("guide", instance_path) is None
    assert read_profile_tool_names("guide", instance_path) == ["camera", TOOL_NAME, "other_space__lookup"]
    assert not settings_path.exists()
    assert clear_profile_tool_override("guide", instance_path) is False


def test_default_profile_uses_canonical_storage_key(configured_profiles: Path) -> None:
    """An empty runtime selection should use the canonical default override."""
    instance_path = configured_profiles

    write_profile_tool_override(None, ["dance"], instance_path)

    assert read_profile_tool_override("default", instance_path) == ["dance"]
    assert read_profile_toolsets(instance_path).profiles == {"default": ["dance"]}


def test_enabling_an_authored_default_does_not_create_an_override(configured_profiles: Path) -> None:
    """Re-enabling an existing default should leave the profile in default mode."""
    instance_path = configured_profiles

    assert enable_profile_tools("default", ["dance"], instance_path) == []
    assert read_profile_tool_override("default", instance_path) is None
    assert not get_profile_toolsets_path(instance_path).exists()


def test_enabling_tools_does_not_overwrite_a_concurrent_profile_save(
    configured_profiles: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile save that starts after an enable operation should be applied last."""
    instance_path = configured_profiles
    enable_read = threading.Event()
    release_enable = threading.Event()
    profile_save_started = threading.Event()
    original_read = read_profile_tool_names

    def paused_read(profile: str | None, path: str | Path | None) -> list[str]:
        tool_names = original_read(profile, path)
        enable_read.set()
        if not release_enable.wait(timeout=1.0):
            raise TimeoutError("Timed out waiting to continue tool enable")
        return tool_names

    def save_profile_tools() -> None:
        profile_save_started.set()
        write_profile_tool_override("guide", ["camera"], instance_path)

    monkeypatch.setattr("reachy_mini_conversation_app.profile_toolsets.read_profile_tool_names", paused_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        enable_future = executor.submit(enable_profile_tools, "guide", ["new_space__search"], instance_path)
        assert enable_read.wait(timeout=1.0)
        profile_save_future = executor.submit(save_profile_tools)
        try:
            assert profile_save_started.wait(timeout=1.0)
            with pytest.raises(TimeoutError):
                profile_save_future.result(timeout=0.1)
        finally:
            release_enable.set()
        assert enable_future.result(timeout=1.0) == ["new_space__search"]
        profile_save_future.result(timeout=1.0)

    assert read_profile_tool_override("guide", instance_path) == ["camera"]


def test_disabling_space_tools_preserves_other_tools_for_every_profile(configured_profiles: Path) -> None:
    """Space removal should create tombstones for matching authored defaults in all profiles."""
    instance_path = configured_profiles

    disabled = disable_profile_tools_by_prefix(
        ["default", "guide", "default"],
        f"{SPACE_ALIAS}__",
        instance_path,
    )

    assert disabled == [
        ("default", [TOOL_NAME]),
        ("guide", [TOOL_NAME]),
    ]
    assert read_profile_tool_names("default", instance_path) == ["dance"]
    assert read_profile_tool_names("guide", instance_path) == ["camera", "other_space__lookup"]
    assert read_profile_tool_override("default", instance_path) == ["dance"]
    assert read_profile_tool_override("guide", instance_path) == ["camera", "other_space__lookup"]
