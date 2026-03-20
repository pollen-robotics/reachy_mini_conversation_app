from pathlib import Path

import pytest

import reachy_mini_conversation_app.prompts as prompts_mod
from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_mini_conversation_app.headless_personality import (
    list_personalities,
    resolve_profile_dir,
    read_instructions_for,
)


def test_builtin_profiles_keep_legacy_public_names() -> None:
    """Built-in abbreviated folders should still surface legacy profile names."""
    names = list_personalities()

    assert "short_mad_scientist_assistant" in names
    assert "mad_scientist_assistant" not in names


def test_legacy_profile_name_resolves_to_short_storage_dir() -> None:
    """Legacy profile names should resolve to the compact built-in directory."""
    profile_dir = resolve_profile_dir("short_mad_scientist_assistant")

    assert profile_dir.name == "mad_scientist_assistant"
    assert (profile_dir / "inst.txt").is_file()


def test_prompts_load_from_compact_builtin_instructions_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt loading should read compact built-in instructions files transparently."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "short_mad_scientist_assistant")
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", DEFAULT_PROFILES_DIRECTORY)

    expected = (DEFAULT_PROFILES_DIRECTORY / "mad_scientist_assistant" / "inst.txt").read_text(encoding="utf-8").strip()

    assert prompts_mod.get_session_instructions() == expected
    assert read_instructions_for("short_mad_scientist_assistant") == expected


def test_aliased_builtin_profile_paths_stay_compact() -> None:
    """Abbreviated built-in profile files should stay within the path budget."""
    project_root = Path(__file__).resolve().parents[1]
    aliased_profiles = [
        DEFAULT_PROFILES_DIRECTORY / "bored_teenager",
        DEFAULT_PROFILES_DIRECTORY / "captain_circuit",
        DEFAULT_PROFILES_DIRECTORY / "chess_coach",
        DEFAULT_PROFILES_DIRECTORY / "hype_bot",
        DEFAULT_PROFILES_DIRECTORY / "mad_scientist_assistant",
        DEFAULT_PROFILES_DIRECTORY / "nature_documentarian",
        DEFAULT_PROFILES_DIRECTORY / "noir_detective",
        DEFAULT_PROFILES_DIRECTORY / "time_traveler",
        DEFAULT_PROFILES_DIRECTORY / "victorian_butler",
    ]

    longest = max(
        len(str(path.relative_to(project_root)))
        for profile_dir in aliased_profiles
        for path in profile_dir.rglob("*")
        if path.is_file()
    )

    assert longest <= 75
