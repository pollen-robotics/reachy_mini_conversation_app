"""Tests for the global companion-tool bundle."""

import importlib

import pytest

from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_mini_conversation_app.companion import COMPANION_TOOL_NAMES
from reachy_mini_conversation_app.personality import available_tool_catalog
from reachy_mini_conversation_app.tools.companion_status import CompanionStatus


def test_companion_status_guides_spoken_transitions() -> None:
    """Manual status checks preserve the spoken task lifecycle."""
    assert "background task finished" in CompanionStatus.description
    assert "background task failed" in CompanionStatus.description
    assert "ask the returned question" in CompanionStatus.description
    assert "call companion_answer" in CompanionStatus.description


def test_companion_tools_are_global_and_not_profile_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured bundle follows one global switch across personalities."""
    monkeypatch.setattr(config, "COMPANION_ENABLED", True)
    monkeypatch.setattr(config, "COMPANION_CONFIGURED", True)
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    companion_names = set(COMPANION_TOOL_NAMES)
    core_tools = importlib.import_module("reachy_mini_conversation_app.tools.core_tools")

    for profile in ("default", "mars_rover"):
        monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", profile)
        core_tools.initialize_tools(force=True)
        assert companion_names <= set(core_tools.get_tools())

    monkeypatch.setattr(config, "COMPANION_ENABLED", False)
    core_tools.initialize_tools(force=True)

    assert companion_names.isdisjoint(core_tools.get_tools())
    assert companion_names.isdisjoint(tool["id"] for tool in available_tool_catalog())
