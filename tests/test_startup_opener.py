"""Tests for the canned-opener startup trigger (issue #290).

Covers:
- ``openers.load_openers`` reads ``profiles/<persona>/openers.txt`` lines.
- ``openers.get_canned_opener`` returns one of those lines, falls back to a
  default when the file is missing, and never falls back to the LLM path.
- ``_send_startup_trigger`` in mode=canned does NOT call the LLM and writes
  the canned line into ``_conversation_history`` as a model-role turn.
- ``_send_startup_trigger`` in mode=llm still routes through the legacy
  ``_dispatch_completed_transcript("[conversation started]")`` path.
"""

from __future__ import annotations
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from robot_comic import openers as openers_module
from robot_comic.config import config
from robot_comic.openers import (
    OPENERS_FILENAME,
    load_openers,
    get_canned_opener,
)


# --------------------------------------------------------------------------- #
# load_openers / get_canned_opener                                            #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def tmp_profiles_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point config.PROFILES_DIRECTORY at a tmp profiles root."""
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles_root)
    return profiles_root


def test_load_openers_reads_lines_from_file(tmp_profiles_dir: Path) -> None:
    profile = tmp_profiles_dir / "fake_comic"
    profile.mkdir()
    (profile / OPENERS_FILENAME).write_text(
        "Line one.\n# comment line\n\nLine two.\n",
        encoding="utf-8",
    )

    lines = load_openers("fake_comic")
    assert lines == ["Line one.", "Line two."]


def test_load_openers_falls_back_to_bundled_default_when_missing(
    tmp_profiles_dir: Path,
) -> None:
    # No profile directory at all — loader should still return something
    # by reading the bundled default profile's openers.txt.
    lines = load_openers("nonexistent_profile")
    assert lines, "Expected fallback to bundled default openers"


def test_get_canned_opener_returns_one_of_file_lines(tmp_profiles_dir: Path) -> None:
    profile = tmp_profiles_dir / "fake_comic"
    profile.mkdir()
    (profile / OPENERS_FILENAME).write_text("Only line.\n", encoding="utf-8")

    assert get_canned_opener("fake_comic") == "Only line."


def test_get_canned_opener_hardcoded_fallback_when_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If neither the profile nor the bundled default has openers, the
    loader returns the hardcoded neutral fallback — never the LLM path."""
    empty_profiles = tmp_path / "empty_profiles"
    empty_profiles.mkdir()
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", empty_profiles)
    monkeypatch.setattr(openers_module, "DEFAULT_PROFILES_DIRECTORY", empty_profiles)

    result = get_canned_opener("anything")
    assert result == openers_module._DEFAULT_FALLBACK_OPENER


def test_load_openers_skips_blank_and_comment_lines(tmp_profiles_dir: Path) -> None:
    profile = tmp_profiles_dir / "fake_comic"
    profile.mkdir()
    (profile / OPENERS_FILENAME).write_text(
        "\n   \n# a comment\nHello.\n   # indented comment kept (only leading-# stripped)\nWorld.\n",
        encoding="utf-8",
    )

    lines = load_openers("fake_comic")
    assert "Hello." in lines
    assert "World." in lines
    assert all(not line.startswith("#") for line in lines)


# --------------------------------------------------------------------------- #
# Bundled persona files exist and are non-empty                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("persona", ["house_comedian", "default"])
def test_bundled_persona_openers_files_exist(persona: str) -> None:
    """The personas mentioned in the issue body have openers shipped."""
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "profiles" / persona / OPENERS_FILENAME
    assert path.exists(), f"Missing openers file for persona {persona!r}"
    contents = path.read_text(encoding="utf-8").strip()
    assert contents, f"Openers file for {persona!r} is empty"


# --------------------------------------------------------------------------- #
# _send_startup_trigger behaviour (canned vs llm)                              #
# --------------------------------------------------------------------------- #


def _make_llama_handler() -> Any:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
    from robot_comic.tools.core_tools import ToolDependencies

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = ChatterboxTTSResponseHandler(deps)
    handler._http = AsyncMock()
    return handler


@pytest.mark.asyncio
async def test_canned_mode_does_not_call_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In mode=canned, _send_startup_trigger MUST NOT hit the LLM round-trip."""
    handler = _make_llama_handler()
    monkeypatch.setattr(config, "STARTUP_TRIGGER_MODE", "canned")
    monkeypatch.setattr(
        "robot_comic.llama_base.get_canned_opener",
        lambda: "Canned hello.",
    )

    # If the canned path is implemented correctly, _dispatch_completed_transcript
    # must NEVER be called from _send_startup_trigger.
    dispatch_mock = AsyncMock()
    monkeypatch.setattr(handler, "_dispatch_completed_transcript", dispatch_mock)
    # TTS is mocked so we don't hit a network.
    synth_mock = AsyncMock()
    monkeypatch.setattr(handler, "_synthesize_and_enqueue", synth_mock)

    await handler._send_startup_trigger()

    dispatch_mock.assert_not_awaited()
    synth_mock.assert_awaited()


@pytest.mark.asyncio
async def test_canned_mode_records_model_role_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _make_llama_handler()
    monkeypatch.setattr(config, "STARTUP_TRIGGER_MODE", "canned")
    monkeypatch.setattr(
        "robot_comic.llama_base.get_canned_opener",
        lambda: "Canned hello.",
    )
    monkeypatch.setattr(handler, "_synthesize_and_enqueue", AsyncMock())

    assert handler._conversation_history == []
    await handler._send_startup_trigger()

    # The llama-side history schema uses {role: "assistant", content: ...} —
    # which is the model-role-equivalent on that backend.
    assert any(
        msg.get("content") == "Canned hello." and msg.get("role") == "assistant"
        for msg in handler._conversation_history
    )


@pytest.mark.asyncio
async def test_llm_mode_routes_through_legacy_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode=llm preserves the legacy `[conversation started]` round-trip."""
    handler = _make_llama_handler()
    monkeypatch.setattr(config, "STARTUP_TRIGGER_MODE", "llm")

    dispatch_mock = AsyncMock()
    monkeypatch.setattr(handler, "_dispatch_completed_transcript", dispatch_mock)
    synth_mock = AsyncMock()
    monkeypatch.setattr(handler, "_synthesize_and_enqueue", synth_mock)

    await handler._send_startup_trigger()

    dispatch_mock.assert_awaited_once_with("[conversation started]")
    synth_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_speak_canned_opener_handles_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty canned text logs a warning and does not call TTS or update history."""
    handler = _make_llama_handler()
    synth_mock = AsyncMock()
    monkeypatch.setattr(handler, "_synthesize_and_enqueue", synth_mock)

    await handler._speak_canned_opener("")
    await handler._speak_canned_opener("   ")

    synth_mock.assert_not_awaited()
    assert handler._conversation_history == []
