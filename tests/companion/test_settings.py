"""Tests for protected background-assistant settings."""

import os
import stat
from pathlib import Path

import pytest

from reachy_mini_conversation_app.companion.settings import (
    CompanionSettings,
    read_companion_settings,
    write_companion_settings,
    get_companion_settings_path,
)


API_URL = "https://alice-smolagents-assistant-reachy-mini.hf.space"
API_TOKEN = "a" * 32


def test_companion_settings_are_private_and_fail_closed(tmp_path: Path) -> None:
    """Saved credentials remain protected and invalid settings disable the feature."""
    settings = CompanionSettings(enabled=False, api_url=API_URL, api_token=API_TOKEN)

    settings_path = write_companion_settings(tmp_path, settings)

    assert settings_path == get_companion_settings_path(tmp_path)
    assert read_companion_settings(tmp_path) == settings
    if os.name == "posix":
        assert stat.S_IMODE(settings_path.stat().st_mode) == 0o600

    with pytest.raises(ValueError):
        CompanionSettings(api_url=API_URL, api_token="a" * 31 + "\x00")

    settings_path.write_text(
        f'{{"enabled": true, "api_url": "{API_URL}"}}\n',
        encoding="utf-8",
    )

    assert read_companion_settings(tmp_path) == CompanionSettings(enabled=False)
