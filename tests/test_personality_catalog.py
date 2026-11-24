from pathlib import Path

import pytest

from reachy_mini_conversation_app.tools import compose_song, speak_in_voice
from reachy_mini_conversation_app.personality import available_tool_catalog


def test_catalog_marks_built_in_tools_that_call_a_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Song and voice tools name their Space for the Tools page; the shared helper module is not a tool."""
    monkeypatch.chdir(tmp_path)

    catalog = {tool["id"]: tool for tool in available_tool_catalog()}

    assert catalog["compose_song"].get("space") == compose_song.SONG_SPACE
    assert catalog["speak_in_voice"].get("space") == speak_in_voice.VOICE_SPACE
    assert "space" not in catalog["play_song"]
    assert "_space_audio" not in catalog and "space_audio" not in catalog
