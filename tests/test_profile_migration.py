"""Legacy sidecar-file profiles are converted to profile.md at startup."""

from pathlib import Path

import pytest

from reachy_mini_conversation_app.profile_store import (
    write_profile,
    list_profile_names,
    migrate_legacy_profiles,
    read_profile_from_directory,
    read_packaged_default_profile,
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


def test_migration_converts_all_legacy_content(tmp_path: Path) -> None:
    """Instructions, tools, voice and greeting all land in profile.md."""
    _write_legacy(
        tmp_path / "legacy",
        instructions="Tu es le robot de la famille.",
        tools="dance\n# a comment\nplay_emotion\n",
        voice="Serena",
        greeting="Dis bonjour.",
    )

    assert migrate_legacy_profiles(tmp_path) == ["legacy"]

    profile = read_profile_from_directory("legacy", tmp_path / "legacy")
    assert profile.instructions == "Tu es le robot de la famille."
    assert profile.default_tools == ("dance", "play_emotion")
    assert profile.voice == "Serena"
    assert profile.greeting == "Dis bonjour."
    # Conversion is additive: the sidecar files are left in place.
    assert (tmp_path / "legacy" / "instructions.txt").is_file()


def test_migration_without_tools_txt_inherits_default_tools(tmp_path: Path) -> None:
    """The shape that broke in the wild: instructions.txt and greeting.txt only."""
    _write_legacy(tmp_path / "famille", greeting="Dis bonjour.")

    migrate_legacy_profiles(tmp_path)

    profile = read_profile_from_directory("famille", tmp_path / "famille")
    assert profile.default_tools == read_packaged_default_profile().default_tools
    assert profile.default_tools
    assert profile.greeting == "Dis bonjour."


def test_migration_never_touches_an_existing_profile_document(tmp_path: Path) -> None:
    """profile.md wins over leftover sidecars, and re-running is a no-op."""
    write_profile("converted", tmp_path / "converted", "version actuelle", ["dance"])
    _write_legacy(tmp_path / "converted", instructions="ancienne version")

    assert migrate_legacy_profiles(tmp_path) == []
    assert migrate_legacy_profiles(tmp_path) == []

    profile = read_profile_from_directory("converted", tmp_path / "converted")
    assert profile.instructions == "version actuelle"


def test_migration_loses_a_race_against_a_concurrent_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile.md saved from the UI mid-migration must not be overwritten."""
    import reachy_mini_conversation_app.profile_store as store

    _write_legacy(tmp_path / "racy", instructions="ancienne version")
    real_read = store._read_legacy_profile

    def read_then_save_from_ui(profile_name: str, profile_directory: Path) -> store.ProfileDefinition:
        legacy = real_read(profile_name, profile_directory)
        write_profile(profile_name, profile_directory, "sauvegarde UI", ["dance"])
        return legacy

    monkeypatch.setattr(store, "_read_legacy_profile", read_then_save_from_ui)

    assert migrate_legacy_profiles(tmp_path) == []

    profile = read_profile_from_directory("racy", tmp_path / "racy")
    assert profile.instructions == "sauvegarde UI"


def test_migration_skips_broken_directories_without_raising(tmp_path: Path) -> None:
    """One broken profile must not block the others or startup."""
    _write_legacy(tmp_path / "empty", instructions="   \n")
    _write_legacy(tmp_path / "valid")
    (tmp_path / "junk").mkdir()

    assert migrate_legacy_profiles(tmp_path) == ["valid"]
    assert not (tmp_path / "empty" / "profile.md").exists()


def test_migration_of_a_missing_root_is_a_no_op(tmp_path: Path) -> None:
    """A robot without user profiles has no root to migrate."""
    assert migrate_legacy_profiles(tmp_path / "absent") == []


def test_reader_and_listing_stay_strict_about_profile_md(tmp_path: Path) -> None:
    """Unmigrated legacy directories are not a supported runtime format."""
    _write_legacy(tmp_path / "legacy")

    with pytest.raises(FileNotFoundError):
        read_profile_from_directory("legacy", tmp_path / "legacy")
    assert list_profile_names(tmp_path) == []
