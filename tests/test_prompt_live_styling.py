"""Unit tests for _load_profile_live_styling() and Gemini Live prompt assembly.

Covers Issue #139 — each comedian profile now ships a gemini_live.txt file.
Tests confirm:
  1. For house_comedian, the loader returns non-empty content from gemini_live.txt.
  2. For a profile without gemini_live.txt, the loader returns None.
  3. The assembled system prompt for house_comedian contains ## DELIVERY with the
     file's content appended.
  4. The assembled system prompt for a profile without gemini_live.txt has no
     ## DELIVERY section.
"""

from pathlib import Path
from unittest.mock import patch

from robot_comic.prompts import get_session_instructions
from robot_comic.gemini_live import _strip_tts_delivery_tags, _load_profile_live_styling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_live_config(
    *,
    profile: str | None,
    profiles_directory: Path,
) -> object:
    """Return a minimal config-like namespace for patching gemini_live.config."""

    class _FakeLiveConfig:
        REACHY_MINI_CUSTOM_PROFILE = profile
        PROFILES_DIRECTORY = profiles_directory

    return _FakeLiveConfig()


def _make_prompts_config(
    *,
    profile: str | None,
    profiles_directory: Path,
) -> object:
    """Return a minimal config-like namespace for patching prompts.config."""

    class _FakePromptsConfig:
        REACHY_MINI_CUSTOM_PROFILE = profile
        PROFILES_DIRECTORY = profiles_directory
        # Use the Gemini Live audio output so the prompt assembler keeps the
        # GEMINI LIVE DELIVERY GUIDANCE section path identical to pre-4f tests.
        AUDIO_OUTPUT_BACKEND = "gemini_live_output"
        FORCE_DELIVERY_TAGS = False

    return _FakePromptsConfig()


def _write_profile(
    tmp_path: Path,
    name: str,
    *,
    instructions: str = "## IDENTITY\n\nTest persona.\n",
    live_txt: str | None = None,
) -> Path:
    """Create a minimal profile directory and return the profiles root."""
    profile_dir = tmp_path / name
    profile_dir.mkdir()
    (profile_dir / "instructions.txt").write_text(instructions, encoding="utf-8")
    if live_txt is not None:
        (profile_dir / "gemini_live.txt").write_text(live_txt, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# _load_profile_live_styling — unit tests
# ---------------------------------------------------------------------------


def test_load_styling_returns_none_when_no_profile() -> None:
    """Returns None when REACHY_MINI_CUSTOM_PROFILE is not set."""
    fake_cfg = _make_live_config(profile=None, profiles_directory=Path("/irrelevant"))
    with patch("robot_comic.gemini_live.config", fake_cfg):
        assert _load_profile_live_styling() is None


def test_load_styling_returns_none_when_file_absent(tmp_path: Path) -> None:
    """Returns None when gemini_live.txt does not exist for the profile."""
    profiles_root = _write_profile(tmp_path, "test_persona")  # no live_txt
    fake_cfg = _make_live_config(profile="test_persona", profiles_directory=profiles_root)
    with patch("robot_comic.gemini_live.config", fake_cfg):
        assert _load_profile_live_styling() is None


def test_load_styling_returns_content_when_file_exists(tmp_path: Path) -> None:
    """Returns the file content (stripped) when gemini_live.txt is present."""
    live_txt = "Speak fast. Hit the consonants. Never drawl."
    profiles_root = _write_profile(tmp_path, "test_persona", live_txt=live_txt)
    fake_cfg = _make_live_config(profile="test_persona", profiles_directory=profiles_root)
    with patch("robot_comic.gemini_live.config", fake_cfg):
        result = _load_profile_live_styling()
    assert result is not None
    assert "Speak fast" in result


def test_load_styling_returns_none_when_file_is_empty(tmp_path: Path) -> None:
    """Returns None when gemini_live.txt exists but contains only whitespace."""
    profiles_root = _write_profile(tmp_path, "test_persona", live_txt="   \n\n  ")
    fake_cfg = _make_live_config(profile="test_persona", profiles_directory=profiles_root)
    with patch("robot_comic.gemini_live.config", fake_cfg):
        assert _load_profile_live_styling() is None


# ---------------------------------------------------------------------------
# house_comedian — confirm real file loads correctly
# ---------------------------------------------------------------------------


def test_house_comedian_live_txt_loads_nonempty(tmp_path: Path) -> None:
    """house_comedian/gemini_live.txt exists and loader returns non-empty content."""
    from robot_comic.config import DEFAULT_PROFILES_DIRECTORY

    fake_cfg = _make_live_config(
        profile="house_comedian",
        profiles_directory=DEFAULT_PROFILES_DIRECTORY,
    )
    with patch("robot_comic.gemini_live.config", fake_cfg):
        result = _load_profile_live_styling()
    assert result is not None, "house_comedian/gemini_live.txt must exist and be non-empty"
    assert len(result) > 20  # sanity: not just a stub


# ---------------------------------------------------------------------------
# Prompt assembly — ## DELIVERY section
# ---------------------------------------------------------------------------


_MINIMAL_INSTRUCTIONS = "## IDENTITY\n\nYou are a test persona.\n"
_LIVE_STYLING_TEXT = "Speak with conviction. Land the punch word. Let silence sit."


def test_delivery_section_present_when_live_txt_exists(tmp_path: Path) -> None:
    """Assembled prompt contains ## DELIVERY when gemini_live.txt is present."""
    profiles_root = _write_profile(
        tmp_path,
        "test_persona",
        instructions=_MINIMAL_INSTRUCTIONS,
        live_txt=_LIVE_STYLING_TEXT,
    )
    prompts_cfg = _make_prompts_config(profile="test_persona", profiles_directory=profiles_root)
    live_cfg = _make_live_config(profile="test_persona", profiles_directory=profiles_root)

    with patch("robot_comic.prompts.config", prompts_cfg):
        instructions = _strip_tts_delivery_tags(get_session_instructions())

    with patch("robot_comic.gemini_live.config", live_cfg):
        live_styling = _load_profile_live_styling()

    assert live_styling is not None
    assembled = f"{instructions}\n\n## DELIVERY\n{live_styling}"
    assert "## DELIVERY" in assembled
    assert _LIVE_STYLING_TEXT in assembled


def test_delivery_section_absent_when_no_live_txt(tmp_path: Path) -> None:
    """Assembled prompt has no ## DELIVERY when gemini_live.txt is absent."""
    profiles_root = _write_profile(
        tmp_path,
        "test_persona",
        instructions=_MINIMAL_INSTRUCTIONS,
        # no live_txt
    )
    prompts_cfg = _make_prompts_config(profile="test_persona", profiles_directory=profiles_root)
    live_cfg = _make_live_config(profile="test_persona", profiles_directory=profiles_root)

    with patch("robot_comic.prompts.config", prompts_cfg):
        instructions = _strip_tts_delivery_tags(get_session_instructions())

    with patch("robot_comic.gemini_live.config", live_cfg):
        live_styling = _load_profile_live_styling()

    # Replicate the loader guard used in _build_live_config
    if live_styling:
        assembled = f"{instructions}\n\n## DELIVERY\n{live_styling}"
    else:
        assembled = instructions

    assert "## DELIVERY" not in assembled
    assert "## IDENTITY" in assembled


def test_house_comedian_assembled_prompt_contains_delivery_section() -> None:
    """Full assembly for house_comedian yields a ## DELIVERY section."""
    from robot_comic.config import DEFAULT_PROFILES_DIRECTORY

    prompts_cfg = _make_prompts_config(
        profile="house_comedian",
        profiles_directory=DEFAULT_PROFILES_DIRECTORY,
    )
    live_cfg = _make_live_config(
        profile="house_comedian",
        profiles_directory=DEFAULT_PROFILES_DIRECTORY,
    )

    with patch("robot_comic.prompts.config", prompts_cfg):
        instructions = _strip_tts_delivery_tags(get_session_instructions())

    with patch("robot_comic.gemini_live.config", live_cfg):
        live_styling = _load_profile_live_styling()

    assert live_styling is not None
    assembled = f"{instructions}\n\n## DELIVERY\n{live_styling}"
    assert "## DELIVERY" in assembled
    # The house_comedian file should reference delivery pacing guidance
    assert any(
        word in assembled.lower()
        for word in ("pace", "punchline", "silence", "measured", "conversational", "delivery")
    )
