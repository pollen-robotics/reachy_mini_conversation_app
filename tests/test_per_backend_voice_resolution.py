"""Tests for per-backend voice file resolution in prompts.get_session_voice.

Covers Issue #481 — optional ``<backend>_voice.txt`` files take precedence
over the shared ``voice.txt`` within a persona directory.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import robot_comic.prompts as prompts_mod
from robot_comic.prompts import get_session_voice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_config(profiles_dir: Path, profile: str) -> object:
    """Return a minimal config-like namespace for patching."""

    class _FakeConfig:
        PROFILES_DIRECTORY = profiles_dir
        REACHY_MINI_CUSTOM_PROFILE = profile
        AUDIO_OUTPUT_BACKEND = ""
        PIPELINE_MODE = ""

    return _FakeConfig()


def _make_profile(profiles_dir: Path, name: str) -> Path:
    """Create and return a profile directory inside *profiles_dir*."""
    profile_dir = profiles_dir / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


# ---------------------------------------------------------------------------
# Core resolution tests
# ---------------------------------------------------------------------------


def test_backend_file_takes_precedence(tmp_path: Path) -> None:
    """A per-backend voice file wins over voice.txt when both are present."""
    profile_dir = _make_profile(tmp_path, "test_persona")
    (profile_dir / "gemini_voice.txt").write_text("Puck", encoding="utf-8")
    (profile_dir / "voice.txt").write_text("Fenrir", encoding="utf-8")

    fake = _fake_config(tmp_path, "test_persona")
    with patch("robot_comic.prompts.config", fake):
        result = get_session_voice(backend="gemini")

    assert result == "Puck"


def test_falls_through_to_voice_txt_when_backend_missing(tmp_path: Path) -> None:
    """voice.txt is used when the requested backend file is absent."""
    profile_dir = _make_profile(tmp_path, "test_persona")
    (profile_dir / "voice.txt").write_text("john_mulaney", encoding="utf-8")

    fake = _fake_config(tmp_path, "test_persona")
    with patch("robot_comic.prompts.config", fake):
        result = get_session_voice(backend="xtts")

    assert result == "john_mulaney"


def test_empty_backend_file_falls_through_to_voice_txt(tmp_path: Path) -> None:
    """An empty per-backend file is treated as absent; voice.txt is used instead."""
    profile_dir = _make_profile(tmp_path, "test_persona")
    (profile_dir / "gemini_voice.txt").write_text("   ", encoding="utf-8")
    (profile_dir / "voice.txt").write_text("Fenrir", encoding="utf-8")

    fake = _fake_config(tmp_path, "test_persona")
    with patch("robot_comic.prompts.config", fake):
        result = get_session_voice(backend="gemini")

    assert result == "Fenrir"


def test_no_backend_arg_preserves_legacy_behavior(tmp_path: Path) -> None:
    """Calling get_session_voice() without backend reads only voice.txt, ignoring any backend files."""
    profile_dir = _make_profile(tmp_path, "test_persona")
    (profile_dir / "gemini_voice.txt").write_text("Puck", encoding="utf-8")
    (profile_dir / "voice.txt").write_text("Fenrir", encoding="utf-8")

    fake = _fake_config(tmp_path, "test_persona")
    with patch("robot_comic.prompts.config", fake):
        result = get_session_voice()

    assert result == "Fenrir"


def test_no_profile_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no custom profile is set, the default voice fallback is returned."""
    monkeypatch.setattr(prompts_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)

    result = get_session_voice(default="test_default_voice")

    assert result == "test_default_voice"


def test_malicious_backend_string_is_rejected(tmp_path: Path) -> None:
    """A backend string containing path separators is rejected; voice.txt is used instead."""
    profile_dir = _make_profile(tmp_path, "test_persona")
    (profile_dir / "voice.txt").write_text("safe_voice", encoding="utf-8")

    fake = _fake_config(tmp_path, "test_persona")
    with patch("robot_comic.prompts.config", fake):
        result = get_session_voice(backend="../etc/passwd")

    # Must not raise, must not attempt path traversal, must fall through to voice.txt
    assert result == "safe_voice"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_backend_whitespace_only_name_is_rejected(tmp_path: Path) -> None:
    """A backend string of only whitespace is rejected (not a valid identifier)."""
    profile_dir = _make_profile(tmp_path, "test_persona")
    (profile_dir / "voice.txt").write_text("safe_voice", encoding="utf-8")

    fake = _fake_config(tmp_path, "test_persona")
    with patch("robot_comic.prompts.config", fake):
        result = get_session_voice(backend="   ")

    assert result == "safe_voice"


def test_backend_with_newline_is_rejected(tmp_path: Path) -> None:
    """A backend string containing newlines is rejected."""
    profile_dir = _make_profile(tmp_path, "test_persona")
    (profile_dir / "voice.txt").write_text("safe_voice", encoding="utf-8")

    fake = _fake_config(tmp_path, "test_persona")
    with patch("robot_comic.prompts.config", fake):
        result = get_session_voice(backend="gemini\nevil")

    assert result == "safe_voice"


def test_multiple_backends_each_resolve_independently(tmp_path: Path) -> None:
    """Two backend files co-exist; each resolves to its own value."""
    profile_dir = _make_profile(tmp_path, "test_persona")
    (profile_dir / "gemini_voice.txt").write_text("Puck", encoding="utf-8")
    (profile_dir / "openai_voice.txt").write_text("alloy", encoding="utf-8")
    (profile_dir / "voice.txt").write_text("Fenrir", encoding="utf-8")

    fake = _fake_config(tmp_path, "test_persona")
    with patch("robot_comic.prompts.config", fake):
        gemini_result = get_session_voice(backend="gemini")
        openai_result = get_session_voice(backend="openai")
        legacy_result = get_session_voice()

    assert gemini_result == "Puck"
    assert openai_result == "alloy"
    assert legacy_result == "Fenrir"
