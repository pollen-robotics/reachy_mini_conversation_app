"""Profiles authored before profile.md must keep working.

Profiles used to be a directory of sidecar files (instructions.txt, tools.txt,
voice.txt, greeting.txt). The profile.md format replaced that and migrated the
bundled profiles, but nothing migrated user profiles under user_personalities/.
Those directories are still on disk and still valid content, so they are read
where they lie instead of being reported as missing.
"""

from pathlib import Path

import pytest

from reachy_mini_conversation_app.profile_store import (
    ProfileFormatError,
    write_profile,
    list_profile_names,
    read_profile_from_directory,
    read_packaged_default_profile,
    profile_directory_has_definition,
)


def _write_legacy(
    directory: Path,
    instructions: str = "tu es un robot",
    *,
    tools: str | None = None,
    voice: str | None = None,
    greeting: str | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "instructions.txt").write_text(instructions, encoding="utf-8")
    if tools is not None:
        (directory / "tools.txt").write_text(tools, encoding="utf-8")
    if voice is not None:
        (directory / "voice.txt").write_text(voice, encoding="utf-8")
    if greeting is not None:
        (directory / "greeting.txt").write_text(greeting, encoding="utf-8")


def test_legacy_profile_is_read_with_all_its_content(tmp_path: Path) -> None:
    """Instructions, tools, voice and greeting all survive."""
    profile_dir = tmp_path / "legacy"
    _write_legacy(
        profile_dir,
        instructions="## IDENTITE\nTu es le robot de la famille.",
        tools="dance\n# a comment\n\nplay_emotion\ncamera\n",
        voice="Serena",
        greeting="Dis bonjour a la famille.",
    )

    profile = read_profile_from_directory("legacy", profile_dir)

    assert profile.instructions == "## IDENTITE\nTu es le robot de la famille."
    assert profile.default_tools == ("dance", "play_emotion", "camera")
    assert profile.voice == "Serena"
    assert profile.greeting == "Dis bonjour a la famille."
    assert profile.hidden is False


def test_legacy_profile_without_tools_inherits_the_default_tool_list(tmp_path: Path) -> None:
    """Cover the shape that actually broke in the wild.

    Builds of that era fell back to the bundled default's tools.txt when a
    profile shipped none, so profiles were authored relying on that fallback and
    carry only instructions.txt and greeting.txt.
    """
    profile_dir = tmp_path / "famille"
    _write_legacy(profile_dir, greeting="Dis bonjour.")

    profile = read_profile_from_directory("famille", profile_dir)

    assert profile.default_tools == read_packaged_default_profile().default_tools
    assert profile.default_tools, "the default profile should contribute a non-empty tool list"
    assert profile.greeting == "Dis bonjour."


def test_legacy_profile_optional_sidecars_may_be_absent(tmp_path: Path) -> None:
    """Only instructions.txt was ever mandatory."""
    profile_dir = tmp_path / "sparse"
    _write_legacy(profile_dir)

    profile = read_profile_from_directory("sparse", profile_dir)

    assert profile.instructions == "tu es un robot"
    assert profile.voice is None
    assert profile.greeting is None


def test_profile_md_wins_when_both_layouts_are_present(tmp_path: Path) -> None:
    """A converted profile must not be shadowed by leftover sidecar files."""
    profile_dir = tmp_path / "converted"
    write_profile("converted", profile_dir, "nouvelle version", ["dance"])
    _write_legacy(profile_dir, instructions="ancienne version", tools="camera")

    profile = read_profile_from_directory("converted", profile_dir)

    assert profile.instructions == "nouvelle version"
    assert profile.default_tools == ("dance",)


def test_empty_legacy_instructions_is_still_an_error(tmp_path: Path) -> None:
    """An empty persona is not something to silently accept."""
    profile_dir = tmp_path / "empty"
    _write_legacy(profile_dir, instructions="   \n")

    with pytest.raises(ProfileFormatError):
        read_profile_from_directory("empty", profile_dir)


def test_directory_with_neither_layout_is_still_missing(tmp_path: Path) -> None:
    """An unrelated directory must not be mistaken for a profile."""
    profile_dir = tmp_path / "not_a_profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "notes.md").write_text("hello", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        read_profile_from_directory("not_a_profile", profile_dir)


def test_legacy_profiles_are_listed_and_selectable(tmp_path: Path) -> None:
    """A profile that loads must also be visible in the profile list."""
    profiles_root = tmp_path / "profiles"
    _write_legacy(profiles_root / "legacy_one")
    write_profile("modern_one", profiles_root / "modern_one", "moderne", ["dance"])
    (profiles_root / "junk").mkdir(parents=True)

    assert list_profile_names(profiles_root) == ["legacy_one", "modern_one"]
    assert profile_directory_has_definition(profiles_root / "legacy_one")
    assert not profile_directory_has_definition(profiles_root / "junk")
