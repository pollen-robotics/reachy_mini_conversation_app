from pathlib import Path

import pytest

from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY
from reachy_mini_conversation_app.profile_store import (
    ProfileFormatError,
    write_profile,
    list_profile_names,
    read_profile_from_directory,
)


def test_profile_document_round_trip(tmp_path: Path) -> None:
    """Profile metadata and Markdown instructions should round-trip losslessly."""
    profile_directory = tmp_path / "guide"

    write_profile(
        "guide",
        profile_directory,
        '# Role\n\nBe concise and say "hello".',
        ["dance", "dance", "", "# disabled", "move_head"],
        voice="Ono_Anna",
        greeting="Open with a \\tiny greeting.",
    )

    profile = read_profile_from_directory("guide", profile_directory)

    assert profile.instructions == '# Role\n\nBe concise and say "hello".'
    assert profile.default_tools == ("dance", "move_head")
    assert profile.voice == "Ono_Anna"
    assert profile.greeting == "Open with a \\tiny greeting."
    assert profile.hidden is False


@pytest.mark.parametrize(
    ("document", "error"),
    [
        ("Legacy instructions only.\n", "expected TOML front matter"),
        ("+++\nschema_version = 1\ndefault_tools = []\n", "missing closing"),
        ("+++\nschema_version = 2\ndefault_tools = []\n+++\nPrompt\n", "Unsupported profile schema"),
        ("+++\nschema_version = 1\ndefault_tools = 'dance'\n+++\nPrompt\n", "expected a list of strings"),
        ("+++\nschema_version = 1\ndefault_tools = []\nvoiec = 'Aiden'\n+++\nPrompt\n", "Unknown profile metadata"),
        ("+++\nschema_version = 1\ndefault_tools = []\n+++\n", "empty instruction body"),
    ],
)
def test_invalid_profile_document_is_rejected(tmp_path: Path, document: str, error: str) -> None:
    """Malformed or legacy profile documents should fail with a useful error."""
    profile_directory = tmp_path / "invalid"
    profile_directory.mkdir()
    (profile_directory / "profile.md").write_text(document, encoding="utf-8")

    with pytest.raises(ProfileFormatError, match=error):
        read_profile_from_directory("invalid", profile_directory)


def test_profile_listing_ignores_directories_that_are_not_profiles(tmp_path: Path) -> None:
    """Arbitrary directories should not be treated as profiles.

    This used to assert that a legacy instructions.txt directory was excluded
    too. That exclusion is what silently unlisted user profiles authored before
    profile.md and, combined with the fatal tool load, stopped the app from
    starting at all. Legacy directories are now readable and therefore listed;
    see test_profile_legacy_layout.py. The original intent, that a directory
    holding no profile of either shape is not a profile, is kept here.
    """
    write_profile("visible", tmp_path / "visible", "Hello.", [])
    unrelated_directory = tmp_path / "not_a_profile"
    unrelated_directory.mkdir()
    (unrelated_directory / "notes.md").write_text("Just notes.\n", encoding="utf-8")

    assert list_profile_names(tmp_path) == ["visible"]


def test_bundled_profiles_enable_head_tracking_by_default() -> None:
    """Every bundled personality should start with head tracking available."""
    for profile_name in list_profile_names(DEFAULT_PROFILES_DIRECTORY):
        profile = read_profile_from_directory(profile_name, DEFAULT_PROFILES_DIRECTORY / profile_name)

        assert "head_tracking" in profile.default_tools, profile_name
